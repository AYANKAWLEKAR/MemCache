# Multi-Session Recall — Design

**Date:** 2026-08-13
**Status:** Approved, implementing
**Branch:** `multi-session-recall` (off `api-testing`)

## Problem

L2 semantic recall is session-locked. `PostgresStore.search_episodes` filters
`WHERE session_id = :session_id`, so a new conversation can never retrieve an
earlier one.

The profile work (PR #1) fixed half of this: identity, decisions, and preferences
now travel across sessions through the graph. Episodes do not. A fresh session
knows *who you are and what you decided*, but cannot recall *the conversation
where you decided it*.

`episodes` has no `user_id` column at all — ownership currently exists only as a
`(:UserProfile)-[:PARTICIPATED_IN]->(:Session)` edge in Neo4j.

## Goal

Episode search spans a user's whole history, ranked so recent context wins
without discarding strong older matches.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | How does L2 learn ownership? | A dedicated nullable `user_id` column on `episodes` |
| 2 | Ranking across history | Similarity × exponential recency decay |
| 3 | Presentation | One merged `Relevant Past Episodes` section |

### Why a column rather than the graph

Ownership is a filter — "which episodes are mine" — and a column with an index is
the right tool. Deriving the session list from Neo4j would make every L2 read
depend on Neo4j being up, turn one query into two cross-database round-trips, and
ship an unbounded `session_id = ANY(...)` list.

This is deliberate redundancy. Neo4j remains the source of truth for *structure*
— entity co-occurrence, alias unification, traversal, decision provenance. The
column is a read-optimization so vector search needs no cross-database join.

## Schema

```sql
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes (user_id);
```

Nullable, written at insert from the `user_id` already threaded through
`process_conversation`. Pre-existing rows stay NULL — "unattributed" — and remain
reachable by `session_id` exactly as today. No backfill is required. The graph
retains the mapping if attribution is wanted later; that is a separate optional job.

## Scoping

```python
search_episodes(query_embedding, session_id, user_id=None, limit=5)
```

- `user_id` **absent** → `WHERE session_id = :session_id`. Byte-identical to today.
- `user_id` **present** → `WHERE user_id = :user_id OR session_id = :session_id`.

The `OR` is load-bearing: it keeps the current session visible even when its
episodes were ingested before a `user_id` existed, so enabling the feature
mid-history does not blind the live conversation.

## Ranking

```
score = similarity × 0.5 ^ (age_days / retrieval_recency_half_life_days)
```

`age_days` is measured from `end_time` to now. Half-life defaults to **30 days**.

### Two-step, and why

Computing decay in SQL and ordering by it would defeat the IVFFlat index, which
only accelerates `ORDER BY embedding <=> q`. A computed-expression sort forces a
full scan of the user's entire history — precisely what gets slow as history grows.

So the query **over-fetches by raw distance** (`limit × retrieval_overfetch_factor`,
default 4), and reranking by decayed score happens in Python. The index stays
useful, and the decay curve becomes a pure function testable with fixed timestamps.

The over-fetch factor is the accuracy/cost dial: too low and a recent-but-moderate
match never enters the candidate set to be promoted.

### Threshold interaction

`retrieval_similarity_threshold` (0.25, calibrated in the audit) continues to
filter on **raw** similarity. Decay affects ordering only. Filtering on the
decayed score would silently cut old-but-relevant episodes rather than merely
ranking them lower.

## Retrieval integration

`user_id` flows from `MemoryRetrieveRequest` (added in the profile work) through
`retrieve_context` into `search_episodes`. Output stays one merged
`Relevant Past Episodes` section.

Episode sources gain `age_days` and `decayed_score` in `details`, so provenance
still reveals that an episode came from another session even though the prose
does not label it.

## Index

`ensure_ivfflat_index(engine)` becomes conditional: it creates the index only when
row count ≥ `lists`, and no-ops otherwise. Called on worker startup alongside
`ensure_l2_schema`. Cross-session search makes this load-bearing rather than the
manual script it is today.

## Testing

### Deterministic unit tests (no ML, no Ollama)

- Decay: half-life boundary (age == half-life → exactly 0.5 × similarity), age 0
  → similarity unchanged, monotonic decrease with age.
- Rerank: a recent moderate match outranks an ancient strong one; an ancient
  match strong enough still appears.
- Over-fetch: candidate count is `limit × factor`; final result is `limit`.
- Threshold applies to raw similarity, not decayed score.

### Integration tests (live Postgres)

- `user_id` absent → session-locked, matching today.
- `user_id` present → episodes from a different session are returned.
- NULL-owner episodes in the current session remain reachable.
- Another user's episodes are never returned.

### Agentic tests (live stack + Ollama)

A cross-session scenario: session A ingests a distinctive fact, session B with the
same `user_id` retrieves the **episode** — not merely the graph fact. This is the
case that fails today.

### Tri-tier visibility

Every tier must be independently verifiable for the same ingest, asserted with
direct queries rather than through the API that wrote them:

| Tier | Assertion |
|------|-----------|
| **Redis (L1)** | `session:{id}` list holds the raw turns, with TTL set |
| **Postgres (L2)** | `episodes` row exists with non-null `embedding` **and** the expected `user_id` |
| **Neo4j (L3)** | `(:Session)-[:HAS_EPISODE]->(:Episode)-[:MENTIONS]->(:Entity)` path exists, and the episode id matches the L2 row id |

The L2↔L3 id agreement is the important one — it proves the tiers describe the
same episode rather than two unrelated records.

## Files

**Modified:** `scripts/init-postgres.sql`, `app/db/postgres.py` (model +
`ensure_l2_schema` + conditional `ensure_ivfflat_index`),
`app/services/postgres_store.py` (scoping + over-fetch),
`app/services/retrieval.py` (decay rerank, pass `user_id`),
`app/workers/tasks.py` (persist `user_id`), `app/config.py` (two settings).

**New tests:** `tests/test_recency_ranking.py` (pure functions),
`tests/test_multi_session_recall.py` (live Postgres),
plus agentic scenario and tri-tier probe additions.

## Scope boundary

L2 only. Entity-driven retrieval — surfacing episodes connected to `ClickHouse`
when the query never mentions it — is a graph traversal change and is excluded.
It is the natural follow-on and is what the L4 tool-call tier will also want.
