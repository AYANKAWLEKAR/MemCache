# L4 Workbench (Tool-Call Log) — Design

**Date:** 2026-08-14
**Status:** Approved, implementing (after the Task tier)
**Branch:** `workbench`

## Value drivers (user-chosen)

1. **Failure memory** — an agent should already know which tool calls failed
   for a task and not re-run that workflow.
2. **Decision provenance** — attach the evidence (tool output) to decisions.
3. **Continuity** — "have I already tried this" survives context loss.

Persona-usage analytics were deliberately deferred; they are derivable later
from the same rows.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | What is stored per call? | Full row with hard byte caps: outputs truncate at 8 KB, errors keep 32 KB |
| 2 | How do calls attach to episodes? | Claim-unlinked: episode creation claims the session's unlinked calls (no client correlation, no time windows) |
| 3 | Graph representation | One `ToolCall` node per call, identity + outcome only; payloads never enter Neo4j |
| 4 | Retrieval | **Known Failures injected automatically** — the agent must already know, not have to ask |

Decision 4 reverses an earlier YAGNI call, on the user's clarified goal: recall
of failures must be proactive.

## Schema (Postgres)

```sql
CREATE TABLE IF NOT EXISTS tool_calls (
    id            BIGSERIAL PRIMARY KEY,
    session_id    VARCHAR(255) NOT NULL,
    user_id       VARCHAR(255),
    task_id       VARCHAR(64),                -- Task node UUID (graph-owned)
    episode_id    INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    tool_name     VARCHAR(255) NOT NULL,
    args          JSONB,
    status        VARCHAR(32)  NOT NULL,      -- 'ok' | 'error'
    output        TEXT,
    error         TEXT,
    output_bytes  INTEGER      NOT NULL,      -- true pre-truncation size
    truncated     BOOLEAN      NOT NULL,
    call_hash     CHAR(64)     NOT NULL,      -- sha256 over canonicalized call
    duration_ms   INTEGER,
    created_at    TIMESTAMPTZ  NOT NULL
);
```

Indexes: `(session_id, created_at DESC)`, `(user_id, created_at DESC)`,
`(call_hash)`, `(episode_id)`, `(task_id)`.

- `ON DELETE SET NULL`: deleting an episode must not destroy the record that
  work happened.
- `output_bytes` records true size even when truncated, so a fragment is always
  identifiable as one.
- `call_hash` canonicalizes args (sorted keys, stable separators) before
  hashing, so `{"a":1,"b":2}` equals `{"b":2,"a":1}` — that equality is what
  makes "have I already tried this" answerable.
- Caps are asymmetric by design: a successful 5,000-line read is worth 8 KB at
  most; a stack trace is the highest-value payload in the tier.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/workbench/tool-call` | Record one call; returns `id`, `truncated`, `call_hash` |
| `GET` | `/workbench/recent` | Filters: `session_id`, `user_id`, `task_id`, `status`, `tool_name`, `call_hash`, `limit` |

Dedup ("have I tried this?") is `GET /workbench/recent?call_hash=…`; failure
review is `?status=error`. One query shape, different filters.

## Episode + task linkage (worker)

After `insert_episode`:

1. `claim_tool_calls(session_id, episode_id)` — links every still-unlinked call
   in the session (`episode_id IS NULL`) to the new episode.
2. If the Task tier resolved a task for this episode, claimed rows also gain
   its `task_id`.
3. For each claimed call, Neo4j gains
   `(:Episode)-[:INVOKED]->(:ToolCall {id, tool_name, status, at})`, with a
   uniqueness constraint on `ToolCall.id`.

Task→failure traversal needs no new edge:
`(:Task)<-[:ADVANCES]-(:Episode)-[:INVOKED]->(:ToolCall {status:'error'})`.
Entity-scoped lookup is likewise free via existing `MENTIONS` edges.

Accepted consequences of claim-unlinked: calls in a never-ingested session stay
orphaned (still queryable by session/user, absent from the graph), and a burst
of calls before one short exchange all attach to that episode — lossy
attribution that errs toward keeping evidence.

**Resilience:** workbench graph writes and task backfill never fail an ingest;
failures log and move on, same contract as the profile and task tiers.

## Retrieval: Known Failures

When `user_id` is present, retrieval injects a capped **Known Failures**
section (`workbench_max_failures_in_context`, default 5): task scope and user
scope **union** — the active task's failures rank first, everything else
follows by recency. (Originally specced as task-scoped *else* user-scoped;
that switch was a live bug — a failure stamped to an older task vanished the
moment any unrelated newer task became active.) Each line
carries tool name and the first line of the error. Source type `tool_failure`,
tier `L4` — which extends the `MemorySource.tier` literal and the demo agent's
tier map.

Not included (own spec later): NER over tool outputs for direct
`ToolCall→Entity` edges; retention/TTL policy for old calls.

## Testing

- **Unit:** truncation at both caps; `output_bytes` accuracy; canonicalization
  key-order invariance; hash stability.
- **Integration (live Postgres):** record/filter round-trips; claim links only
  unlinked calls; orphans stay orphaned; task_id backfill.
- **Tri-tier:** one call verifiable in Postgres and as a Neo4j `ToolCall` node,
  ids agreeing, hanging off the same episode as the L2 row — the id-agreement
  discipline that caught real bugs twice before.
- **Agentic (Ollama):** session A records a failing call and ingests; session B
  retrieves and the Known Failures section carries the failure without being
  asked.
