# MemCache

Episodic memory infrastructure for long-running agents: a **FastAPI** service that stores recent conversation turns in **Redis**, durable summarized episodes with vectors in **PostgreSQL (pgvector)**, and entities and relationships in **Neo4j**. A **Celery** worker performs summarization (Ollama), embeddings, and graph updates asynchronously after ingest.

## Features

- **L1 — Redis**: Append capped, TTL’d message lists per session (`session:{session_id}`).
- **L2 — PostgreSQL + pgvector**: Episode summaries and 384-dimensional embeddings for semantic recall.
- **L3 — Neo4j**: Sessions, episodes, entities, co-occurrence edges, and decision/preference nodes (see `app/services/neo4j_store.py`).
- **API**: `X-API-Key` authentication, **`GET /health`**, **`POST /memory/ingest`** (Redis + Celery enqueue). **`POST /memory/retrieve`** is planned; `retrieve_context` in `app/services/retrieval.py` is not implemented yet.
- **Worker**: `process_conversation` summarizes via Ollama, embeds with SentenceTransformer, writes L2 and L3.

## Requirements

- **Python** 3.11 or newer
- **Docker** (recommended) for Redis, PostgreSQL with pgvector, and Neo4j
- **Ollama** (host or remote) for summarization when running the worker
- **spaCy** English model: `en_core_web_sm` (used by the worker for NER)

## Quick start

### 1. Clone and environment

```bash
git clone <repository-url> MemCache
cd MemCache
cp .env.example .env
```

Edit `.env` if your passwords, URLs, or API keys differ from the defaults.

### 2. Start data stores

```bash
docker compose up -d redis postgres neo4j
```

Wait until services are healthy. PostgreSQL is initialized with `scripts/init-postgres.sql` (pgvector extension and `episodes` table).

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run Ollama (for the worker)

Install [Ollama](https://ollama.com/), then pull a model matching **`OLLAMA_MODEL`** in `.env` (default `llama2`):

```bash
ollama pull llama2
```

### 5. Run Celery worker

The worker must be running for **`POST /memory/ingest`** to complete background processing (L2/L3). In one terminal:

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

Optional: run the worker in Docker (after bringing up Redis, Postgres, and Neo4j):

```bash
docker compose --profile worker up worker
```

Point **`OLLAMA_BASE_URL`** at Ollama. From inside the worker container, `http://host.docker.internal:11434` is the default on macOS/Windows when Ollama runs on the host.

### 6. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Root** (no API key): [http://localhost:8000/](http://localhost:8000/) — simple JSON banner.
- **OpenAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)

## API overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | Service name and status. |
| `GET` | `/health` | `X-API-Key` | Redis, PostgreSQL, and Neo4j checks; `200` if all OK, `503` if degraded. |
| `POST` | `/memory/ingest` | `X-API-Key` | Append messages to L1 Redis and enqueue `process_conversation`; returns **202** with `task_id`. |

Example ingest:

```bash
curl -s -X POST http://localhost:8000/memory/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dummy-api-key-123" \
  -d '{"session_id":"demo","messages":[{"role":"user","content":"Hello"}]}'
```

Default API key in `.env.example` is `dummy-api-key-123` (comma-separated list in **`API_KEYS`**).

## Configuration

All settings support environment variables (see **`app/config.py`**). Common variables:

| Variable | Purpose |
|----------|---------|
| `API_KEYS` | Comma-separated valid API keys for `X-API-Key`. |
| `REDIS_URL` | L1 Redis and often Celery broker/backend host. |
| `POSTGRES_URL` | L2 PostgreSQL connection string. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | L3 Bolt connection. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY` | Ollama generate API; Bearer token optional for local Ollama. |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Celery broker and result backend (defaults align with Redis URL). |
| `EMBEDDING_MODEL` | SentenceTransformers model id (default MiniLM 384-d). |
| `SPACY_MODEL` | spaCy model for NER in the worker (default `en_core_web_sm`). |

Copy **`.env.example`** to **`.env`** and adjust values for production.

## Testing

Fast tests (no Docker):

```bash
pytest -m "not integration"
```

Integration tests need the stack (Redis, Postgres, Neo4j, and sometimes more):

```bash
pytest -m integration
```

## Project layout (abbreviated)

```
MemCache/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── api/                 # Routes, models, deps, service clients
│   ├── db/                  # Postgres engine/models, Neo4j driver
│   ├── services/            # Redis, Postgres, Neo4j, retrieval, summarization, graph extraction
│   └── workers/             # Celery app and process_conversation task
├── scripts/init-postgres.sql
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── tests/
```

Design notes and step-by-step write-ups live under **`steps taken/`** (for example `STEP_5_PLAN.md` for the Celery worker and `api_implementation.md` for the HTTP slice).

## License

Add a `LICENSE` file if you distribute this project publicly.
