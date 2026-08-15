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
