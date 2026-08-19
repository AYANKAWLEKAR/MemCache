# Goal Hierarchy — Design

**Date:** 2026-08-19
**Status:** Approved, implementing
**Branch:** `multi-goal-stringing`
**Builds on:** `2026-08-14-task-goal-node-design.md`, `2026-08-14-proactive-retrieval-design.md`

## Problem

Task nodes are flat. `(:UserProfile)-[:PURSUES]->(:Task)` and
`(:Episode)-[:ADVANCES]->(:Task)` exist; nothing relates one Task to another.
Adjudication answers *same-or-new*, never *part-of*. So a user working "fix
duplicate column" under "migrate telemetry schema" under "ship telemetry v2"
gets three unrelated open Tasks, and every retrieval path that reads "the
active task" — the `Current task:` line, Known Failures scoping, the inherited
activation seeds — sees exactly one of them. The parent goal's failures,
decisions and episodes are invisible while its subgoal is being worked, and
the agent has no idea what larger objective it is serving.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | What should change for the agent? | Inherit parent-goal context, and show the agent where it sits in the plan. Not: task-list tidying, cascade-close. |
| 2 | Shape | Strict tree: at most one parent per Task, any depth. |
| 3 | Re-parenting | Both directions: a new Task may be placed under an existing one, **and** may adopt existing *root* Tasks as children. No synthetic umbrella Tasks — the model never mints a goal nobody stated. |
| 4 | Active task + activity | Active task is the most recently *directly* advanced Task (the leaf). Rendered with its ancestor path. Activity bubbles up so a parent never ages out of the adjudicator's candidate list while its subgoals are worked. |
| 5 | How the parent edge is found | Structural shortlist (graph evidence) → dedicated single-question Ollama placement call on ≤3 candidates → defensive parse → edge or nothing. The existing adjudication call is untouched. |

Precision beats coverage throughout: a wrong `SUBGOAL_OF` edge injects an
unrelated goal's failures into context; a missing one only costs the
inheritance this feature adds. Every ambiguous case resolves to "no edge".

## 1. Schema and invariants

```
(:Task)-[:SUBGOAL_OF]->(:Task)     // child -> parent
(:Task {id, title, status, created_at, updated_at, last_advanced_at, closed_at})
```

`last_advanced_at` is the one new property. Two timestamps because they now
serve two different questions:

- `updated_at` — bumped when an episode directly ADVANCES the task **and**
  when any descendant is advanced. Orders the adjudicator's candidate list.
  This is the bubbling: an umbrella goal stays visible in the top-20 while work
  proceeds beneath it, so the adjudicator keeps matching against it instead of
  minting a duplicate.
- `last_advanced_at` — bumped **only** on direct advancement. Selects the
  active task. Because episodes advance leaves, this resolves to the leaf
  without a tiebreak; without the split, parent and child would share an
  identical `updated_at` after bubbling and `ORDER BY … LIMIT 1` would pick
  between them arbitrarily.

Backfill: `last_advanced_at` is read as `coalesce(t.last_advanced_at,
t.updated_at)` everywhere, so pre-existing Tasks need no migration.

Invariants, all enforced in `TaskStore`, none by Neo4j (there is no
relationship-cardinality constraint):

1. **At most one outgoing `SUBGOAL_OF` per Task.** `set_parent` is a no-op
   for the edge that already exists and raises `TaskHierarchyError` for a
   *different* parent. Re-parenting is out of scope; the adjudicator only ever
   proposes placement for a Task that is currently a root.
2. **No cycles.** Guaranteed primarily by what may be proposed, and verified
   by one bounded descendant walk in `set_parent` so misuse from any caller
   fails loudly. Placement is only attempted for a *subject* Task that is
   currently a root (a fresh Task, or a matched Task with no parent). The
   candidate pool excludes the subject and every descendant of the subject.
   - `child_of X`: X is outside the subject's subtree, so `subject -> X`
     cannot close a loop.
   - `parent_of X`: X must be a root (checked in the store); a root's subtree
     cannot contain the subject because the subject is itself a root.
   `set_parent` re-checks both conditions and raises `TaskHierarchyError`
   otherwise; the worker logs and writes nothing.
3. **Single owner.** Child and parent must both be `PURSUES`'d by the same
   `UserProfile`. Unreachable by construction (candidates come from
   `list_placement_candidates(user_id, …)`), but asserted in the store: cross-user
   parentage would leak one person's failures into another's context, which is
   the one class of bug this codebase fails loudly on rather than degrades.
4. **Ancestor walks ignore `status`.** A closed parent with an open child is
   legitimate; filtering by status would silently truncate the path.
5. **No cascade on close.** `close_task` is unchanged. Closing a parent leaves
   children open. The path renders `(done)` on closed ancestors.

## 2. Store API (`task_store.py`)

New:

| Method | Contract |
|--------|----------|
| `set_parent(child_id, parent_id)` | MERGE `(child)-[:SUBGOAL_OF]->(parent)`. No-op if that exact edge already exists. Raises `TaskHierarchyError` if the child already has a *different* parent, if child == parent, if the parent is inside the child's subtree, or if the two are not pursued by the same profile. |
| `get_parent(task_id) -> TaskRow \| None` | One hop up. |
| `get_ancestors(task_id) -> list[TaskRow]` | Nearest-first, root last. Empty for a root. Bounded by `task_max_depth` (config, default 8) as a fetch cap — depth is not a semantic limit, the cap only bounds a runaway walk. |
| `get_children(task_id) -> list[TaskRow]` | Direct children only. |
| `get_lineage_ids(task_id) -> list[str]` | `[task_id, parent, grandparent, …]` — the id chain retrieval scopes by. Same bound as `get_ancestors`. |
| `get_descendant_ids(task_id) -> set[str]` | Every Task below this one, any depth (bounded by `task_max_depth`). Used to build the exclusion set for placement candidates. |
| `list_placement_candidates(user_id, *, subject_id, limit) -> list[PlacementCandidate]` | Open Tasks pursued by the user, excluding the subject and its descendants, most recently active first, capped at `limit`. Each row carries `id, title, is_root, updated_at, entities (set[str]), sessions (set[str])` — the evidence the shortlist scores. One Cypher round-trip. |
| `task_evidence(task_id) -> PlacementCandidate` | The same evidence row for the subject itself. |
| `active_task(user_id) -> TaskRow \| None` | Open Task with the greatest `coalesce(last_advanced_at, updated_at)`. Replaces the three `list_open_tasks(user_id, limit=1)` call sites in retrieval. |

Changed:

- `link_episode` — after MERGEing `ADVANCES`, sets `t.last_advanced_at = now`
  and `t.updated_at = now` on the target, then bubbles `updated_at = now` to
  every ancestor: `MATCH (t)-[:SUBGOAL_OF*1..8]->(a) SET a.updated_at = $now`.
  One Cypher statement, so bubbling and the direct bump are atomic.
- `TaskRow` gains `last_advanced_at: str | None`.
- `create_task` writes `last_advanced_at = now` too — a task is "active" the
  moment it is created; without this a just-created leaf that has not yet been
  linked (an ordering that does not happen in the worker, but could in a
  caller) would lose to a stale sibling.

## 3. Placement adjudication (`task_hierarchy.py`, new module)

Runs in the worker **after** `_write_task` returns a task id, only when the
resolved Task is currently a **root** (fresh, or matched with no parent). A
Task that already has a parent is never re-placed.

### 3a. Structural shortlist — `shortlist_candidates(...)`, pure

Input: the subject's evidence row, the candidate rows from
`list_placement_candidates`, and a `similarity(a: str, b: str) -> float`
callable for titles. In the worker that is MiniLM cosine (the embedder is
already loaded there); tests inject a fake. Score per candidate in `[0, 1]`:

```
score = 0.5 * similarity(subject.title, cand.title)
      + 0.3 * jaccard(subject.entities, cand.entities)
      + 0.2 * [subject.sessions ∩ cand.sessions ≠ ∅]
```

Ties break on `updated_at` (newer first). The top `task_placement_candidates`
(default 3) with `score >= task_placement_min_score` (default `0.0`, i.e. off)
go to the model. Zero candidates → no model call, no edge.

Why recency is a tiebreak and not a term: recency is not evidence that two
goals are related, and a recency term would push every recent Task past any
min-score cut. Why the min score defaults to off: with ≤3 candidates the
*model* is the precision gate, and the `unrelated_stays_flat` scenario
measures exactly that. If measurement shows over-linking, `min_score` is the
lever — it is there, it is tested, it is just not load-bearing by default.

Weights are declared, not calibrated. This is a shortlist; its one job is to
put the right parent in front of the model when a user has more open Tasks
than the model can compare.

Every score is symmetric. It says "these goals share territory," never which
is larger. Direction is the model's job.

### 3b. Placement call — `adjudicate_placement(...)`, Ollama

Prompt: the subject title, ≤3 candidate `(id, title)` pairs, one question,
fixed answer vocabulary:

```
Respond with ONLY a JSON object, no prose, no code fences:
{"relation": "child_of" | "parent_of" | "none",
 "task_id": <the id of the goal the relation is with, or null>}

child_of  = the new goal is a smaller step toward that goal.
parent_of = that goal is a smaller step toward the new goal.
none      = unrelated, siblings under some larger goal, or the same goal.
When unsure, answer none.
```

`temperature: 0`. `parse_placement(text, valid_ids)` mirrors
`parse_adjudication`: fenced / prose-wrapped JSON tolerated; missing field,
wrong type, unknown relation, `task_id` outside `valid_ids` → `None`.
`"none"` with a non-null `task_id` → `None` (contradiction).
`"child_of"`/`"parent_of"` with null `task_id` → `None`.

`parent_of` maps to adoption of **exactly one** candidate. Multi-adoption is a
straightforward extension of the same contract but multiplies the ways a 3B
model can be wrong, so it is deferred until measured.

### 3c. Worker wiring — `_write_hierarchy(...)` in `workers/tasks.py`

Called after `_write_task`, before `_claim_workbench`, only when
`resolved_task_id` is not None. Same additive contract as `_write_task`: any
exception logs and returns; the ingest stands. Sequence:

1. `get_parent(resolved)` is not None → return.
2. `list_placement_candidates` + `task_evidence`; shortlist. Empty → return.
3. Placement call. `None` → return.
4. `child_of X` → `set_parent(resolved, X)`.
   `parent_of X` → if X is not a root, log and return (adopting a parented
   Task would give it two parents); else `set_parent(X, resolved)`.
5. `TaskHierarchyError` → log, return. Never propagates.

L4 claim ordering: `_claim_workbench` stamps tool calls with
`resolved_task_id`, the leaf. That is correct — a failure belongs to the
subgoal being worked — and lineage-scoped retrieval (§4) makes it visible from
above.

Placement re-runs on every ingest whose resolved Task is still a root. That
is one extra Ollama call per such ingest, in an async worker, and it is what
lets an umbrella goal stated late adopt work that began earlier.

## 4. Retrieval: pulling context up the hierarchy

Three call sites currently do `list_open_tasks(user_id, limit=1)`. All three
change to `active_task(user_id)` and then widen to the **lineage**
(`get_lineage_ids(active.id)`).

### 4a. `Current task:` line (`_format_profile_facts`)

```
Current task: Fix duplicate column on episodes
  (under: Migrate telemetry schema ▸ Ship telemetry v2)
```

Rendered as one line when there are no ancestors, two when there are. Closed
ancestors render `Ship telemetry v2 (done)`. Source type `task` gains
`lineage: [ids…]` and `depth: int` in details.

### 4b. Known Failures (`_format_known_failures`)

`failed_calls` gains `task_ids: list[str] | None` (replacing the single
`task_id`; the old kwarg is kept as a shim for one release — a single id
becomes a one-element list). Scope union becomes: any failure whose
`task_id IN lineage` OR `user_id = user`. Ranking boost applies to the whole
lineage, leaf first:

```
ORDER BY CASE WHEN task_id = :leaf THEN 2
              WHEN task_id = ANY(:lineage) THEN 1
              ELSE 0 END DESC,
         created_at DESC, id DESC
```

So the subgoal's own failures rank first, then the parent chain's, then the
user's unrelated ones — the "never switch, always union" rule from the L4 spec
is preserved and extended.

### 4c. Structural seeds: the lineage Task nodes (`_format_proactive_context`)

Today the active Task contributes *topical* seeds — the entities its episodes
mention — at `task_seed` (0.6). That stays exactly as it is, leaf only.

New: the lineage Task nodes themselves become seeds, decaying up the tree:

```
seed(Task at depth d) = task_node_seed * task_depth_decay ** d      d=0 is the leaf
```

Defaults `proactive_task_node_seed = 0.2`, `proactive_task_depth_decay = 0.7`.
`build_seeds` gains `task_nodes: dict[str, float]` (already-computed
`Task:<id>` → activation) and merges them with max-wins.
`fetch_neighborhood` gains `task_ids: list[str] | None` — the start set
becomes `(seed:Entity AND seed.name IN $names) OR (seed:Task AND seed.id IN
$task_ids)`; the node id for a Task is `Task:<uuid>` via the existing
`toString(a.id)` fallback, which is what the renderer already handles.

Why Task nodes and not the ancestors' entities: the arithmetic. `MENTIONS`
is a *counted* edge — at count 1 it carries `0.9 · log2/log21 ≈ 0.205`, so an
ancestor's entity seeded at 0.42 reaches that ancestor's episode at 0.069 and
its tool call at **0.0496 — at the floor**. The feature would work by luck.
The Task→Episode→ToolCall route is all structural edges (0.9 each), so:

| depth | Task seed | its episodes (·0.72) | their tool calls (·0.72) |
|------:|----------:|---------------------:|-------------------------:|
| 0 (leaf) | 0.200 | 0.144 | 0.104 |
| 1 (parent) | 0.140 | 0.101 | 0.073 |
| 2 (grandparent) | 0.098 | 0.071 | 0.051 |
| 3 | 0.069 | 0.049 ✗ | — |

Live-mentioned entities still win: a live seed (1.0) reaches its episode at
`1.0 · 0.205 · 0.8 = 0.164 > 0.144`, so what was said in *this* conversation
outranks the goal's own history, which outranks the parent's, which outranks
the grandparent's — the ordering the existing task seed was designed for,
extended one axis. Great-grandparents fall under the floor by design; if a
user's tree is deeper than three, the near ancestors are the relevant ones.

These numbers are the design's *claim*. A deterministic test pins the
arithmetic on a hand-built graph, and `scripts/calibrate_activation.py` is
re-run on the live graph before merge with the result recorded in the
obstacles addendum. Both knobs are config.

**`SUBGOAL_OF` joins `EDGE_PRIORS` at `0.9`**, structural (uncounted), same
class as `PURSUES`/`ADVANCES`. Activation that reaches a Task by any route now
continues to its parent, and provenance paths through the tree read
`task A -SUBGOAL_OF-> task B -ADVANCES-> ep 12 -INVOKED-> call 9`. Because
lineage Tasks are seeds, ancestor Task nodes are *not* rendered in the
proactive section (seeds are skipped — restating them is noise); the
`Current task:` path line in §4a is where the agent reads its position. A
Task reached by activation that is **not** in the lineage still renders
`Related task: <title> (<status>)` as today.

## 5. Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `TASK_MAX_DEPTH` | `8` | Fetch cap on ancestor walks; not a semantic limit. |
| `TASK_PLACEMENT_CANDIDATES` | `3` | Shortlist size handed to the placement call. |
| `TASK_PLACEMENT_MIN_SCORE` | `0.0` | Structural score below which a candidate is not shown to the model. Off by default; the model is the precision gate. |
| `PROACTIVE_TASK_NODE_SEED` | `0.2` | Activation given to the active (leaf) Task node itself. |
| `PROACTIVE_TASK_DEPTH_DECAY` | `0.7` | Multiplier applied per level up the tree. |

## 6. Failure handling

Same contract as every additive tier: **an ingest never fails because
placement failed.** Ollama down, malformed JSON, contradictory verdict,
unknown id, hierarchy invariant violation — all log and leave the tree
unchanged. `TaskHierarchyError` is the only new exception type and is caught
at the worker boundary.

Retrieval degrades the same way it does today: if `active_task` or the
lineage walk raises, the profile/failure/proactive section that needed it is
dropped, `status: degraded`, and a warning is added.

## 7. Testing

The house rule holds: **LLM judgement is a reported metric, never a gate,
unless measurement earns it.**

### Deterministic gates (no model in the loop)

- `parse_placement`: clean / fenced / prose-wrapped / missing field / wrong
  type / unknown relation / hallucinated id / `none`+id contradiction /
  `child_of`+null id.
- `shortlist_candidates`: score formula with an injected similarity, entity
  Jaccard, session bonus, recency as tiebreak only, min-score cut, cap,
  subject and its descendants excluded, empty input.
- `TaskStore` against live Neo4j: `set_parent` happy path + idempotent;
  refuses a *different* second parent; refuses self; refuses a parent inside
  the child's subtree; refuses cross-user; `get_ancestors` order and depth
  cap; `get_children`; `get_descendant_ids`; `get_lineage_ids`;
  `list_placement_candidates` excludes subject + subtree and done, carries
  `is_root` and evidence; `active_task` picks the leaf after bubbling
  (the two-timestamp test — the exact tie the split exists to prevent);
  bubbling bumps every ancestor's `updated_at` and no ancestor's
  `last_advanced_at`.
- Worker wiring with adjudicator **and** placement mocked: `child_of` writes
  the edge under the named parent; `parent_of` adopts the named root;
  `parent_of` a non-root writes nothing; `none` writes nothing; malformed
  verdict writes nothing; `TaskHierarchyError` writes nothing and the ingest
  still returns 202/ok; already-parented Task is never re-placed; L4 calls
  are stamped with the leaf.
- Retrieval, hand-built graph: `Current task:` renders the path; closed
  ancestor renders `(done)`; Known Failures surfaces a parent's failure with
  the leaf's failures ranked first; lineage Task seeds decay by depth and the
  §4c table's arithmetic is pinned; a parent's ToolCall reachable only via its
  Task seed surfaces with the full path in provenance; ancestor Tasks are not
  re-rendered in the proactive section; a non-lineage Task still renders
  `Related task:`.
- Severed tests: (a) drop the lineage Task seeds and assert the parent's
  ToolCall no longer surfaces proactively; (b) set the `SUBGOAL_OF` prior to
  0 and assert a path that must cross the tree dies. Each proves its
  mechanism is the one doing the work.

### Agentic harness (real Ollama, real stack) — `tests/agentic/test_goal_hierarchy.py`

Three-session scenarios, each a *plan* in the `Scenario` style: anchors pin
the facts, the model chooses the words. Every scenario is scored two ways:
**structural** (did the graph get the tree right — measured, and gated only
where the deterministic path already proves the plumbing) and **behavioural**
(does session 3's retrieved context contain what only the parent chain could
have supplied — this is the gate that matters).

Scenarios (each run as its own test; `-k hierarchy` runs the set):

1. **top_down** — S1 states the umbrella goal ("ship telemetry v2"). S2 states
   a narrower goal ("migrate the telemetry schema to ClickHouse"), records a
   failing `alembic` tool call. S3 states an even narrower goal ("fix the
   duplicate column on episodes") and asks *"anything I should know before I
   start?"* — naming neither alembic nor the migration. **Gate:** context
   contains the alembic failure and the S1 umbrella title. **Metric:** tree
   depth == 3, edges point the right way.
2. **bottom_up** — the demo's own story, reversed: S1 states "fix duplicate
   column" and records the failure; S2 states "migrate telemetry schema" and
   the model must *adopt* S1's task; S3 continues the migration and asks what
   to watch out for. **Gate:** S1's failure surfaces in S3. **Metric:** S1
   task is a child of S2 task.
3. **unrelated_stays_flat** — S1 "ship telemetry v2"; S2 "plan the offsite in
   Lisbon" (disjoint entities); S3 continues the offsite. **Gate:** no
   `SUBGOAL_OF` between them (precision), and S3's context does **not** contain
   S1's telemetry content in the Known Failures or Current task lines. This is
   the false-positive check — without it a model that says `child_of` to
   everything would pass scenarios 1 and 2.
4. **sibling_subgoals** — S1 umbrella; S2 subgoal A with failure; S3 subgoal B
   under the same umbrella, asks for context. **Gate:** S3's `Current task:`
   line names the umbrella as an ancestor. **Metric:** does A's failure reach
   B via `U -SUBGOAL_OF- A -ADVANCES- ep -INVOKED- call` (the umbrella seed at
   depth 1 crossing down into a sibling)? Reported, since whether sibling
   failures *should* surface is a product judgement this spec does not make.

Each agentic test also prints the S3 context verbatim under a `[context]`
banner and the tree under `[tree]` so the qualitative read the user asked for
is in the pytest output, not reconstructed after the fact. Model verdicts are
printed under `[metric]`.

**Gate-promotion rule:** a behavioural assertion in 1–2 is a gate only if the
run-N trial during implementation passes on every run; otherwise it demotes to
`[metric]` and the deterministic path is the gate. The trial count and outcome
are recorded in the obstacles addendum, as was done for adjudication (40/40).

## 8. Scope

In: `SUBGOAL_OF` edge, two-timestamp activity, placement shortlist + call,
worker wiring, lineage-scoped retrieval in all three call sites, lineage Task
seeds + `fetch_neighborhood` task start-points, activation prior for the new
edge, config knobs, deterministic +
agentic tests, calibration addendum, README graph diagram update.

Out: cascade close, re-parenting an already-parented Task, multi-adoption per
verdict, synthetic umbrella Tasks, a `/tasks` API surface (retrieval is the
consumer; an API can follow once the tree exists), sibling-failure policy.
