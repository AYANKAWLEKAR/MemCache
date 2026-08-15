# MemCache

Memory infrastructure for LLM agents that remembers **what was said, who was
involved, what you're trying to do, and what failed** — and hands it back
unprompted, so an agent never walks into the same wall twice.

Hand-built on boring, inspectable technology: FastAPI, Redis, PostgreSQL +
pgvector, Neo4j, Celery, spaCy, and local Ollama models. No memory framework
imported; every tier and every edge type is original design.

## The closed loop, on camera

`scripts/demo_closed_loop.py`, real output. Monday, session A: a migration
fails and the failure is recorded; the conversation is ingested. Thursday,
session B — a **brand-new conversation**:

```
retrieved context:
  | User Profile:
  | User: demo-54078c
  | Current task: Fix migration failure in ClickHouse
  |
  | Known Failures:
  | Failed action: alembic — DuplicateColumn: column user_id already exists on episodes
  |
  | Relevant Past Episodes:
  | Episode 39: ... migration of a telemetry schema to ClickHouse using Alembic has failed.

Q: You are resuming work on the telemetry schema migration.
   What is your very first action, and why?

WITH memory   : Review and address the 'DuplicateColumn: column user_id already
                exists on episodes' error ... to ensure a clean migration path.
WITHOUT memory: Review and document the current state of the existing telemetry
                schema to understand its structure and data flow.
```

The question contains no hint of the failure — anything the with-memory agent
knows, it learned from MemCache. Same model, same prompt, different behavior.

## Four tiers, one relation graph

| Tier | Store | Holds |
|------|-------|-------|
| **L1** | Redis | Raw recent turns per session, capped + TTL'd |
| **L2** | Postgres + pgvector | Summarized episodes with 384-d embeddings; user-scoped, recency-ranked semantic recall |
| **L3** | Neo4j | The relation graph: sessions, episodes, entities, decisions, preferences, user profiles, **tasks** |
| **L4** | Postgres | Tool calls with outcomes — byte-capped payloads, content-hash dedup, claimed into episodes |

What makes the system more than a RAG stack is the graph connecting them:

```
(:UserProfile)──PURSUES──▶(:Task)
      │                      ▲
   HAS_ALIAS              ADVANCES
      ▼                      │
  (:Entity)◀──MENTIONS──(:Episode)──INVOKED──▶(:ToolCall)
                             │
                 DECIDED / PREFERS ──▶ (:Decision) / (:Preference)
```

One Cypher traversal answers: *for this user's goal, which conversations
advanced it, which people/systems were involved, and which actions failed* —
episode, entity, goal, and action in a single query. Payloads never enter the
graph; Neo4j holds identity and relationships, Postgres holds the data, and
tests assert the two agree on ids.

## How memory gets built

`POST /memory/ingest` writes L1 immediately and enqueues a Celery job that:

1. summarizes the conversation (Ollama, `qwen2.5:3b` by default),
2. embeds the summary (MiniLM) and inserts the L2 episode with its owner,
3. extracts entities (spaCy NER, label-filtered) and decisions/preferences
   into the graph,
4. resolves the **user profile** — aliases (`Dana` ⊆ `Dana Whitfield`),
   name/title/location/gender from stated evidence only, explicit always
   beating inferred,
5. **adjudicates the task**: a second, JSON-only Ollama call decides whether
   this conversation starts a goal, continues one, or finishes one,
6. **claims tool calls**: unlinked L4 rows in the session attach to the new
   episode and inherit its task, then mirror into the graph.

Steps 4–6 are additive by contract: a dead Ollama, malformed JSON, or a
hallucinated task id degrades to "no attachment, logged" — an ingest can never
fail because inference failed. Identity ambiguity is the one thing that fails
loudly (`EpisodeCollisionError`, `ProfileAliasConflictError`, HTTP 409):
guessing is how two people get silently merged.

## Retrieval

`POST /memory/retrieve` returns one context document — recent conversation,
user profile with current task, **known failures** (task-scoped and user-scoped,
unioned), semantically relevant episodes from the user's *whole history*
(similarity × recency decay), and graph facts — plus per-line provenance:
every source carries its tier, ids, and scores, so nothing in the context is
unattributable.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Redis/Postgres/Neo4j connectivity |
| `POST` | `/memory/ingest` | Store turns, enqueue processing (`user_id` optional — enables profile/task/cross-session tiers) |
| `POST` | `/memory/retrieve` | Hybrid context + provenance |
| `GET` | `/profile/{user_id}` | Resolved identity: attributes with provenance, aliases, decisions, preferences |
| `PATCH` | `/profile/{user_id}` | Set attributes explicitly (always beats inference) |
| `POST` | `/profile/{user_id}/alias` | Register an alias; 409 on conflict |
| `POST` | `/workbench/tool-call` | Record a tool invocation (L4) |
| `GET` | `/workbench/recent` | Filterable call log; dedup via `call_hash` |

All routes authenticate via `X-API-Key`.

## Quick start

```bash
cp .env.example .env
docker compose up -d redis postgres neo4j
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:3b
```

Run the API and (in another terminal) the worker:

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

Or skip straight to the demo (self-contained, runs the worker inline):

```bash
.venv/bin/python scripts/demo_closed_loop.py
```

## Testing

205 tests; the full suite runs against the live stack and holds green across
repeated runs.

```bash
pytest -m "not integration"   # fast, no Docker
pytest                        # full: live Redis/Postgres/Neo4j + Ollama
```

The parts worth stealing:

- **An Ollama-driven agent harness** (`tests/agentic/`) generates realistic
  traffic under a fixed contract — *the model chooses the words, the scenario
  chooses the facts*: planned turns carry anchor strings that must survive
  generation, with deterministic fallback, so model variance degrades realism
  but never test correctness.
- **Independent probes**: integration tests assert via their own SQL/Cypher,
  never through the API that wrote the data, and tri-tier tests require L2 and
  L3 to agree on episode ids — a check that has caught real corruption twice.
- **Measured, not asserted**: the similarity threshold was calibrated from
  real embedding distributions (the original 0.7 sat above the maximum
  achievable score — the whole semantic tier was dead while every test was
  green); NER reliability was quantified per-name and demoted to a reported
  metric; the real-LLM task-inference gate was justified by a measured 40/40
  trial. LLM judgement is never a CI gate unless measurement earns it.

## Design history

Decisions, trade-offs, and the audit trail live in the repo:

- `docs/superpowers/specs/` — one design doc per subsystem (profile,
  multi-session recall, task/goal nodes, L4 workbench), each recording the
  roads not taken.
- `steps taken/` — the original ten-defect audit, an obstacles-and-decisions
  log, and an honest portfolio review including the open weaknesses: extraction
  is spaCy-small + regex with measured limits, the recency half-life and
  over-fetch factor are admitted guesses, IVFFlat creation is still manual, and
  claim-based attribution assumes sequential sessions.

## Configuration

Everything is env-driven (`app/config.py`). The interesting knobs:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Summarization + task adjudication |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | `0.25` | L2 recall floor (calibrated; retune if you change the embedding model) |
| `RETRIEVAL_RECENCY_HALF_LIFE_DAYS` | `30` | Episode ranking half-life |
| `TASK_CANDIDATE_LIMIT` | `20` | Open tasks shown to the adjudicator |
| `WORKBENCH_OUTPUT_MAX_BYTES` / `WORKBENCH_ERROR_MAX_BYTES` | `8192` / `32768` | Asymmetric caps: stack traces are the tier's most valuable bytes |
| `WORKBENCH_MAX_FAILURES_IN_CONTEXT` | `5` | Known Failures cap in retrieval |

## License

Add a `LICENSE` file if you distribute this project publicly.
