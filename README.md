# MemCache

[![CI](https://github.com/AYANKAWLEKAR/MemCache/actions/workflows/ci.yml/badge.svg)](https://github.com/AYANKAWLEKAR/MemCache/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tri-tier episodic memory for AI agents** — not a cache. MemCache gives long-running agents durable, queryable memory: it ingests conversation turns, summarizes and embeds them asynchronously, builds a knowledge graph of entities and decisions, and serves back hybrid context on demand — including facts recalled from *previous* sessions and a resolved user identity profile.

- **L1 — Redis**: recent raw turns per session (capped, TTL'd lists).
- **L2 — PostgreSQL + pgvector**: summarized episodes with 384-d embeddings for semantic recall, ranked with recency decay.
- **L3 — Neo4j**: sessions, episodes, entities, co-occurrence edges, decisions/preferences — plus a **canonical user profile** with provenance-tracked attributes, alias resolution with conflict detection, and explicit-beats-inferred precedence.

A **FastAPI** service handles ingest/retrieve; a **Celery** worker does summarization (Ollama), embedding (SentenceTransformers), NER (spaCy), and graph updates off the request path.

## How it works

```
POST /memory/ingest                        POST /memory/retrieve
        │                                            │
        ▼                                            ▼
   Redis (L1) ──► Celery worker ──► summarize ─► hybrid retrieval:
   raw turns        (async)         embed        L1 recent turns
        │                           extract      + L2 semantic episodes
        └── 202 + task_id           entities     (recency-decayed, cross-session)
                                       │         + L3 profile facts & decisions
                                       ▼                │
                              Postgres (L2)             ▼
                              Neo4j (L3)         token-budgeted context
                                                 + per-source provenance
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | Service name and status. |
| `GET` | `/health` | `X-API-Key` | Redis, PostgreSQL, and Neo4j checks; `200` if all OK, `503` if degraded. |
| `POST` | `/memory/ingest` | `X-API-Key` | Append turns to L1 and enqueue async processing; returns `202` with `task_id`. Optional `user_id` triggers profile resolution. |
| `POST` | `/memory/retrieve` | `X-API-Key` | Hybrid context for a query: recent turns + semantically similar episodes (recency-ranked, across sessions) + profile facts when `user_id` is given. Returns `context`, per-tier `sources`, and degradation `warnings`. |
| `GET` | `/profile/{user_id}` | `X-API-Key` | Resolved canonical profile: attributes with provenance (`explicit`/`inferred`, confidence, evidence), aliases, decisions, preferences. |
| `PATCH` | `/profile/{user_id}` | `X-API-Key` | Explicitly set attributes/display name. Explicit values always beat inferred ones. |
| `POST` | `/profile/{user_id}/alias` | `X-API-Key` | Manually register an alias; conflicts return `409`, never silent guesses. |

Example round trip:

```bash
curl -s -X POST http://localhost:8000/memory/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dummy-api-key-123" \
  -d '{"session_id":"demo","user_id":"ayan","messages":[{"role":"user","content":"I decided to use Postgres for the billing service."}]}'
```

```bash
curl -s -X POST http://localhost:8000/memory/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dummy-api-key-123" \
  -d '{"session_id":"demo-2","user_id":"ayan","query":"what database did I pick for billing?"}'
```

The second call runs in a *different session* and still recalls the decision — that's the point.

Default API key in `.env.example` is `dummy-api-key-123` (comma-separated list in **`API_KEYS`**).

## Quick start

Requires **Python 3.11+**, **Docker**, and **[Ollama](https://ollama.com/)** (for the worker's summarization).

```bash
git clone https://github.com/AYANKAWLEKAR/MemCache.git
cd MemCache
cp .env.example .env

# 1. Data stores (Redis, Postgres+pgvector, Neo4j)
docker compose up -d redis postgres neo4j

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Ollama model (matches OLLAMA_MODEL in .env, default llama2)
ollama pull llama2

# 4. Celery worker (required for ingest post-processing)
celery -A app.workers.celery_app worker --loglevel=info

# 5. API — docs at http://localhost:8000/docs
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The worker can also run in Docker: `docker compose --profile worker up worker` (point `OLLAMA_BASE_URL` at the host, e.g. `http://host.docker.internal:11434` on macOS/Windows).

## Configuration

All settings are environment variables (see **`app/config.py`**). Common ones:

| Variable | Purpose |
|----------|---------|
| `API_KEYS` | Comma-separated valid API keys for `X-API-Key`. |
| `REDIS_URL` | L1 Redis and Celery broker/backend host. |
| `POSTGRES_URL` | L2 PostgreSQL connection string. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | L3 Bolt connection. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` | Ollama generate API; Bearer token optional for local Ollama. |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Celery broker and result backend (defaults align with Redis URL). |
| `EMBEDDING_MODEL` | SentenceTransformers model id (default MiniLM 384-d). |
| `SPACY_MODEL` | spaCy model for NER in the worker (default `en_core_web_sm`). |

## Testing

145 tests across three layers:

```bash
# Fast unit tests (no Docker) — this is what CI runs
pytest -m "not integration and not agentic"

# Integration tests (need the docker compose stack)
pytest -m integration

# Agentic end-to-end: a local Ollama agent converses across sessions
# and the suite verifies recall, identity collapse, and graph state
pytest -m agentic
```

Lint with `ruff check app tests`.

## Project layout

```
MemCache/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── api/                 # Routes, request/response models, deps, service clients
│   ├── db/                  # Postgres engine/models, Neo4j driver
│   ├── services/            # Redis/Postgres/Neo4j stores, retrieval, summarization,
│   │                        #   graph + profile extraction, profile store
│   └── workers/             # Celery app and process_conversation task
├── docs/superpowers/        # Design specs (profile node, multi-session recall)
├── scripts/                 # Postgres init + ivfflat index SQL
├── tests/                   # Unit, integration, and agentic suites
└── docker-compose.yml
```

Design specs live under **`docs/superpowers/specs/`**.

## License

[MIT](LICENSE)
