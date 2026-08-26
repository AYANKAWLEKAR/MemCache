# Task Tier + L4 Workbench — Obstacles & Decisions Log

Session of 2026-08-14, branch `workbench`. Everything below was hit or decided
while implementing the two specs (`2026-08-14-task-goal-node-design.md`,
`2026-08-14-workbench-tool-log-design.md`). Final state: **203 tests, 5/5
stable full-suite runs** including every Ollama-driven agentic path.

## Obstacles hit, in order

### 1. FK on `episodes` broke a legacy fixture (real collision, two workstreams)

The workbench store was built in parallel by a delegated subagent while the
task tier was built here. The moment its `tool_calls` table (FK →
`episodes(id)`) existed in the live DB, `tests/test_l2_postgres.py`'s
`TRUNCATE TABLE episodes RESTART IDENTITY` began failing with
`FeatureNotSupported` — Postgres refuses plain TRUNCATE on a table referenced
by an FK. Both workstreams independently identified the same root cause.

**Fix:** `TRUNCATE … CASCADE` in that fixture, with a comment noting the blast
radius now includes test-DB `tool_calls` rows. **Residual risk documented:**
the L2 suite and workbench suite must not run concurrently against one
database. (That fixture's `RESTART IDENTITY` is also the episode-id-reuse
hazard that `EpisodeCollisionError` exists for — pre-existing, left as is.)

### 2. Task-node leak through test teardowns (self-inflicted, caught by diagnosis)

Wiring real adjudication into the worker meant every fixture that ingests with
a `user_id` silently began creating Task nodes — and none of their teardowns
deleted tasks. Found while diagnosing an unrelated one-off failure, not by a
test. **Fix:** every such teardown gained
`OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) … DETACH DELETE … t`, plus a sweep
query confirming zero already-leaked nodes.

### 3. One-off agentic failure: transient Ollama error during summarization

One full-suite run failed `test_scenario_builds_expected_memory` with "no
episode reached L2" — summarization returned `None`, which by design degrades
to a skipped episode. Immediate rerun and five subsequent full runs were clean
(observed rate ≈ 1 in ~12 full runs). **Decision:** documented rather than
patched. A summarize retry would mask real outages, and the skip path is the
designed behaviour. If the rate grows, the fix is retry-with-jitter in the
worker, not in tests.

### 4. Two-model-responsibility risk (design correction before code)

The original plan folded goal extraction into the summary prompt (one Ollama
call). Corrected during design: asking a 3B model for prose *and* strict JSON
in one response risks corrupting the summary — damaging L2 to save a round
trip an async worker doesn't need. Adjudication became a second, JSON-only
call at temperature 0.

## Decisions with evidence

### Gate vs. metric for real-Ollama task inference — measured, not assumed

House rule: LLM judgement is a reported metric, never a CI gate. Task
adjudication sits on the write path, so before writing the agentic test, the
model was measured directly: 10 trials × 4 behaviours (create-new,
match-paraphrase, mark-complete, null-goal) with `qwen2.5:3b` at temperature 0.

**Result: 40/40.** On that evidence the clear-cut end-to-end case (explicit
goal statement → open Task with ADVANCES edge) gates CI; paraphrase *merge
quality* stays a printed `[metric]` (observed merging correctly in the run
that shipped).

### Hallucinated task ids invalidate the whole verdict

If the model claims a `matches_task_id` that doesn't exist, the parser returns
`None` rather than falling back to "create new task" — minting duplicates from
the model's worst outputs would erode exactly the precision this tier exists
to provide.

### Regression-proof discipline for tests that pass first-run

Two agentic tests passed the moment they were written, which under TDD proves
nothing. Both were verified by severing the code under test and watching them
fail with the expected symptom, then restoring:
- Known Failures injection severed → "failure not surfaced unprompted".
- (Earlier precedent, same branch lineage: cross-session scoping reverted →
  "no episode source returned".)

### Delegation contract for the parallel subagent

The workbench store was specified to a subagent as *new files only* with exact
signatures, DDL, cap semantics, and scoped-teardown test rules; shared config
fields were committed **before** spawning to remove the one file both
workstreams needed. The returned code was reviewed line-by-line (parameterized
SQL only, byte-safe UTF-8 truncation, `COALESCE` task backfill) and its 18
tests re-run here before committing. One deviation it reported — `failed_calls`
filtering by task alone when `task_id` is given — matches the spec's literal
wording and was kept.

### Fixture hygiene as a first-class rule

Every new fixture cleans exactly what it created (uuid-scoped ids,
parameterized deletes). The one global wipe left in the codebase
(`test_l3_neo4j`'s `MATCH (n) DETACH DELETE n`) is the historical pattern that
caused the original episode-collision incident; new code never adds another.

## Addendum — post-review fixes (same day)

### 5. Known-Failures shadowing (live product bug, found by post-hoc audit)

Verified live before fixing: a failure stamped to task T1 vanished from
retrieval the moment any unrelated newer task became most-recently-active.
Root cause was the spec itself ("task-scoped when known, *else* user-scoped")
— implemented literally, wrong by design. Fix: the scopes union, active-task
failures ranked first. Spec corrected with the incident noted inline.

### 6. Postgres three-valued logic in the ranking boost

Inside the fix, the first boost expression evaluated to NULL (not false) for
untasked rows — and Postgres sorts NULLs FIRST under `DESC`, so untasked
failures outranked the active task's own. Diagnosed with a raw-SQL probe
printing the boost column, fixed with `COALESCE(... , false)` and a comment
naming the trap.

### 7. Demo control leaked the answer

The first behavior-delta question named "the duplicate user_id column" — so
the memoryless control answered correctly too, proving nothing. Rewritten to
contain no hint of the failure: anything the with-memory agent knows must have
come from MemCache. The delta is now attributable.

### 8. Ten-minute full-suite hang (environmental)

One suite run hung ~10 minutes with no failure; Ollama answered normally on
probe immediately after, no stray processes, immediate rerun green in 87s.
Second transient Ollama incident on record. If a third occurs, the fix is a
per-call retry-with-jitter in the worker's HTTP layer — not in tests.


## Addendum — weighted graph + proactive retrieval (same day)

### 9. Structural edges were being strangled by their own count

First live trace of activation from `clickhouse` reached the episode (0.164)
but the `ToolCall` two hops on died at **0.027** under the 0.05 floor. Cause:
`INVOKED`/`HAS_EPISODE`/etc. are structural — a tool call has exactly one
`INVOKED` edge forever — yet the log-count formula scaled them by
`log2/log21 ≈ 0.23`. Counting only makes sense for evidence that repeats
(`RELATED_TO`, `MENTIONS`). Fix: `COUNTED_EDGES`; structural edges carry the
full prior. RED test written from the measured numbers first.

### 10. `DetachedInstanceError` hidden by the graph block's bare `except`

The proactive builder read `Episode.summary` *after* `session_scope` closed;
the ORM had expired the instance on commit. Every upstream stage was correct
and the failure was invisible because the surrounding `except Exception`
degraded silently. Fixed by reading scalars in-session, and the except now
logs — a swallowed exception is how this bug survived one full green run.

### 11. Two unit tests encoded wrong math, not the algorithm

`weak_path_dies_at_hop_one` used a floor (0.1) that also killed the *first*
weak hop (0.091); `cycle_terminates` asserted an ordering between two nodes
that are symmetric by construction. Both fixed by working the arithmetic by
hand before touching code — the algorithm was right.

### Calibration, measured

`scripts/calibrate_activation.py` on the live demo graph, seed `clickhouse`,
13 edges: Episode 0.164 · Task 0.131 · ToolCall 0.118 · UserProfile 0.094 —
a clean 0.09–0.20 band above the 0.05 floor; second weak co-occurrence hop
≈0.008 dies. Recorded in `config.py` next to the values.


### 12. Provenance paths were reconstructed, and sometimes fabricated

`explain_path` originally walked backwards from a node choosing the
strongest-looking neighbour, because activation only returned scores. That is
not equivalent to the route activation took. A randomized search over 4000
graphs found **174 divergent paths**; the minimal case is three edges:

    a -DECIDED(1)-> c        sets c = 0.64 in one hop
    a -ADVANCES(20)-> b -ADVANCES(20)-> c   proposes exactly 0.64, never wins

The reconstruction sees the tie, picks `b`, and reports a two-hop path that
never happened — naming an intermediate entity the user was never connected
through. For an explainability feature that is worse than no path.

Fix: `spread_activation` returns `ActivationResult(scores, parents)`, recording
the winning parent at the update site; `explain_path` walks the record and the
backwards search is deleted. Verified by re-running the same 4000-graph search
against an independently written reference implementation: **19,781 nodes
checked, 0 disagreements**, scores identical — provenance-only change.

**Lesson:** deriving a fact after the fact is not the same as recording it when
it happens, even when the derivation looks equivalent. The tie case is where
they part.


## Addendum — goal hierarchy (2026-08-19, branch `multi-goal-stringing`)

### 13. Ancestor entity seeds land AT the floor; seed the Task nodes instead

The first design seeded ancestors' *entities* decayed by depth. Worked by
hand: `MENTIONS` is counted, so at count 1 it carries ~0.205, and a parent's
tool call landed at 0.0496 against a 0.05 floor — the feature would have
worked by luck. Seeding the lineage `Task` nodes rides all-structural edges
instead. Writing the pinning test then corrected the spec a second time:
`ADVANCES` carries prior 1.0 (not the assumed 0.9), and the `SUBGOAL_OF` hop
(0.9·0.8 = 0.72) out-propagates the per-depth seed decay (0.7), so every
ancestor's score is set by the *leaf* seed crossing the tree — scores are
identical with the leaf seeded alone. The per-depth seeds are kept for fetch
coverage (every lineage task is a neighborhood start point, so the radius cap
cannot cut a deep ancestor out of the pulled subgraph). Resulting band from a
0.2 leaf seed: episodes 0.160 / 0.115 / 0.083 by depth, tool calls 0.115 /
0.083 / 0.060, depth-3 dies; a live entity's episode (0.164) still outranks
the goal's own history. Verified on the live graph with
`calibrate_activation.py task:<leaf-id>` — identical numbers — which also
exposed that the script still read the pre-`ActivationResult` return shape
(`act.items()`); both modes now read `.scores`.

### 14. UserProfile is a hub, and the old PURSUES prior leaked through it

The severed-`SUBGOAL_OF` test came back with the *propagated* score instead
of the seed's own — the root's tool call was still reachable with the tree
edge dead. The route was `Task -PURSUES- UserProfile -PURSUES- Task`: at the
old 0.9 prior a 0.2 Task seed lit **every** goal the user pursues to 0.104 —
numerically identical to a real grandparent — and their failures to 0.06.
"Pull context up the lineage" had silently become "pull everything this user
ever did." PURSUES drops to 0.6: the hub route dies at 0.046 (< 0.05 floor)
while the legitimate alias→Profile→Task route keeps 0.48. A dedicated
precision test plants an unrelated goal with its own failure one hub-hop away
and asserts it stays out of proactive context while Known Failures (user
scope, union) still lists it last.

### 15. Placement prompting: measured, iterated, and partially reverted

A 16-case probe battery (8 adjudication, 8 placement) against live
qwen2.5:3b, temperature 0. The terse placement prompt scored 5/8 — it
inverted `parent_of` into `child_of` ("Ship telemetry v2" vs existing
"Migrate the telemetry schema") and picked the wrong candidate when the true
parent's sibling was present. A rewritten prompt (decide-which-is-broader
procedure + three worked examples) scored **worse** — 3/8, all misses
over-conservative `none` — so it was reverted; the terse prompt stands.
Adjudication started at 4/8: "as part of shipping telemetry v2, migrating the
schema" was *matched into* the umbrella task instead of becoming a new goal
(correct for the old flat tier, wrong once part-of is representable). One
added rule — a smaller step of an open task, or a bigger goal an open task is
one step of, is a NEW goal, not a match — took it to 6/8 without touching the
40/40-verified create/match/complete/null behaviour (suite still green).

### 16. Agentic verdict: plumbing proven, direction is model-bound

Five full runs of the four three-session scenarios (top_down, bottom_up,
unrelated_stays_flat, sibling_subgoals) with real Ollama on both sides:

| claim | result |
|-------|--------|
| behavioural gates (failure text reaches S3; no false ancestor line) | **20/20 pass** |
| correct SUBGOAL_OF edges | **0** across all runs |
| incorrect edges | 1 (bottom_up run 2: direction inverted) |
| false links between unrelated goals | 0 |

Deterministic tests prove every verdict lands correctly (mocked placement
builds the exact trees); the model simply cannot judge direction at 3B. So:
tree-shape claims stay `[metric]`, behavioural claims gate, and a
`task_placement_enabled` kill switch ships next to the recorded measurement.
The behavioural gates pass *without* the tree because Known Failures'
task-lineage scope unions with user scope and L2 recall is user-wide —
layered fallbacks, not dead code: with a stronger model the same wiring
upgrades failure ranking, the `(under: …)` line, and structural pull, for
free. The honest summary: **the hierarchy's retrieval machinery is proven
end-to-end with planted trees; the inference that builds trees from
conversation awaits a better local judge.**
