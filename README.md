# MemCache

[![CI](https://github.com/AYANKAWLEKAR/MemCache/actions/workflows/ci.yml/badge.svg)](https://github.com/AYANKAWLEKAR/MemCache/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

MemCache is a memory service for LLM agents. An agent sends it conversations as they happen. MemCache stores what was said, who was involved, what the user is trying to accomplish, and which actions failed. When the agent later asks for context, MemCache returns a single document containing the relevant history, including facts the current question never mentions. The purpose is to let an agent resume work in a new session without repeating mistakes it has already made.

The system is built directly on FastAPI, Redis, PostgreSQL with pgvector, Neo4j, Celery, spaCy, and local Ollama models. It does not import a memory framework. Every storage tier and every relationship type described below is original design.

## The four memory tiers

Different kinds of memory have different access patterns, so each kind lives in the store suited to it.

| Tier | Store | Contents |
|------|-------|----------|
| L1 | Redis | The raw recent turns of each conversation |
| L2 | PostgreSQL with pgvector | Summaries of past conversations, with embeddings for similarity search |
| L3 | Neo4j | A graph of people, entities, goals, decisions, and preferences, and the relationships between them |
| L4 | PostgreSQL | A log of tool calls and their outcomes, including error messages |

**L1 holds the present.** Redis is an in-memory database with very fast reads and writes. It holds the last turns of each active conversation in a capped list with an expiry time. This tier answers the question of what was just said, and it needs no durability beyond the session.

**L2 holds the past in searchable form.** Each finished conversation is condensed into a short summary called an episode. The summary is converted into an embedding, a fixed-length vector of numbers in which similar meanings produce nearby vectors. pgvector is a PostgreSQL extension that stores these vectors and finds the nearest ones to a query vector. This is what allows a question about a schema migration to retrieve an episode about that migration even when the wording differs. Results are ranked by a combination of similarity and recency, so newer episodes outrank older ones of equal relevance.

**L3 holds structure.** Neo4j is a graph database: it stores records as nodes and relationships as edges, and it is efficient at following chains of relationships. This tier records that a user pursues a goal, that a goal is a subgoal of another, that an episode mentioned an entity, and that the user made a decision or stated a preference. Structure is what plain text search cannot provide: it lets the system reason about how facts relate, not only about what they say.

**L4 holds evidence of failure.** Every recorded tool call is stored with its arguments, status, and error output. Error payloads are capped by size, with a larger cap for errors than for normal output, because a stack trace is the most useful content this tier holds. Duplicate calls are detected by a content hash. When an agent is about to retry an action, this tier is what tells it precisely how that action failed before.

The graph connects the tiers:

```
(:UserProfile)──PURSUES──▶(:Task)──SUBGOAL_OF──▶(:Task)
      │                      ▲
   HAS_ALIAS              ADVANCES
      ▼                      │
  (:Entity)◀──MENTIONS──(:Episode)──INVOKED──▶(:ToolCall)
                             │
                 DECIDED / PREFERS ──▶ (:Decision) / (:Preference)
```

Payloads never enter the graph. Neo4j stores identity and relationships; PostgreSQL stores the data; tests assert that the two agree on ids.

## Methodology

### How memory is written

The agent calls `POST /memory/ingest` with the turns of a conversation. The API writes them to L1 immediately and returns. A Celery background worker (Celery is a task queue that runs jobs outside the request path) then processes the conversation:

1. An Ollama model (`qwen2.5:3b` by default, running locally) writes a summary of the conversation.
2. The summary is embedded with a MiniLM sentence-transformer model and inserted into L2 as an episode owned by the user.
3. spaCy, a natural language processing library, extracts named entities from the text. Entities, decisions, and preferences become nodes and edges in the L3 graph.
4. The user profile is resolved. Names, titles, and locations are recorded only from stated evidence, and a value the user set explicitly always overrides an inferred one. Aliases are linked by a subset rule, so that "Dana" can attach to "Dana Whitfield".
5. A second, JSON-only model call decides whether the conversation starts a new goal, continues an existing one, or completes one.
6. Tool calls recorded in the session that are not yet attached to an episode are attached to the new one and inherit its goal.

Steps 4 through 6 are additive by contract. If the model is unreachable, returns malformed JSON, or names a goal that does not exist, the step is skipped and logged, and the ingest still succeeds. The single exception is identity ambiguity. If an alias could belong to two different people, the API returns HTTP 409 rather than guessing, because a wrong guess would silently merge two users' histories.

### How memory is read

The agent calls `POST /memory/retrieve` with a query. The response is one context document assembled from all four tiers: the recent turns, the user profile with the current goal, known failures, semantically similar episodes from the user's full history, and related graph facts. Every line carries provenance: the tier it came from, the ids involved, and the scores that selected it, so nothing in the document is unattributable.

Two mechanisms distinguish this from a standard retrieval pipeline.

**Activation spreading.** Retrieval does not only match the query. The entities that appear in the recent turns and the query are located in the graph and given an activation score of 1.0. Entities already connected to the user's current goal are given 0.6. Activation then flows outward along edges, diminishing with the weight of each edge crossed, and every node that ends above a measured threshold is fetched from the tier that owns it. There is no fixed hop limit; how far activation reaches is a consequence of edge weights. The effect is that mentioning a topic in passing surfaces its recorded consequences. Each result carries the chain of edges that reached it:

```
clickhouse -RELATED_TO(6)-> alembic -MENTIONS(1)-> episode 41 -INVOKED-> tool call 49
```

Edge weights encode evidence quality. Edges that accumulate observations, such as co-occurrence, carry a count, and their weight grows with the logarithm of that count up to a cap. Structural edges, such as the link between an episode and a tool call, carry a fixed prior, because such an edge exists exactly once and repetition is not evidence. An edge type adjudicated by a model is assigned a higher prior than one produced by co-occurrence alone. The spreading computation is a pure Python function over a single fetched neighborhood, which makes it testable on hand-built graphs and replaceable behind the same interface.

**Goal lineage.** Goals form a tree through `SUBGOAL_OF` edges. A model call decides whether a new goal is a step toward an existing one, choosing among at most three structurally shortlisted candidates, and any ambiguous case resolves to no edge at all. Retrieval walks up the tree from the current goal: the context names the goal together with its ancestors, failures attached to the current goal rank ahead of failures attached to its ancestors, and the lineage nodes seed activation so that a parent goal's failed tool call can surface in a fresh session. One measured limitation is documented rather than hidden: across 20 scripted scenario runs, the default 3B model could not judge parent-child direction correctly, so tree shape is reported as a metric rather than asserted, and the `TASK_PLACEMENT_ENABLED` flag disables the mechanism entirely.

### Measurement over assertion

Numeric choices in the system were calibrated rather than picked. The similarity threshold for L2 recall was derived from real embedding score distributions after the initial value of 0.7 turned out to sit above the highest score the embedder could produce, which had silently disabled the semantic tier while all tests passed. The activation threshold and per-hop decay were set from measured distributions on a live graph (`scripts/calibrate_activation.py`). The reliability of spaCy's entity extraction was quantified per name and reported as a metric. A model's judgement is never used as a test gate unless a measurement has first shown the model capable of the judgement.

## Example: the same agent with and without memory

`scripts/demo_closed_loop.py` runs the full loop in one process. On Monday, in session A, a database migration fails and the failure is recorded and ingested. On Thursday, in session B, a new conversation begins and the agent is asked how to proceed. The question contains no reference to the earlier failure. The transcript below is condensed from the script's output:

```
retrieved context:
  | User Profile:
  | User: demo-54078c
  | Current task: Fix migration failure in ClickHouse
  |
  | Known Failures:
  | Failed action: alembic (DuplicateColumn: column user_id already exists on episodes)
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

Everything the first agent knows about the failure came from MemCache. The model and the prompt are identical in both runs.

## Demo frontend

A one-page Streamlit application contains six clickable demos. Each demo seeds the tiers through the real ingest pipeline, then shows the same `qwen3:4b` agent answering the same question with and without MemCache context, together with a table of the episode, entity, goal, and tool-call ids that entered the context.

Two demos follow a student persona through a semester:

- **Student companion.** In a session a month old, a student introduces themselves as a computer science student at Berkeley, records a decision to target backend and infrastructure internships, and states a preference for demanding work after 9pm. In a session today, the student adds one line: a behavioral interview with a fintech startup is next Friday. The retrieval query is only "what should I be getting ready for?". The agent with memory identifies the student, the roles they are targeting, the upcoming interview, and the evening hours in which to schedule preparation. The agent without memory can use none of this.

- **Recruiting roadmap.** Five sessions spread across a semester define a goal tree. The root goal is a summer software engineering internship; coursework, interview preparation, and side projects are recorded as subgoals, one of which is shipping this repository. The final session records a real pytest failure (the similarity-threshold bug described above, found during this project's own development). Weeks later, the query "Sitting down to work. Where was I, and what should I not waste time on?" produces a context naming the current subgoal, the root goal it serves, and the exact recorded failure. The demo's own description notes that the goal tree is planted by the demo, because measurement showed the 3B judge cannot construct it; every other step runs the live pipeline.

The remaining four demos each isolate a single mechanism: recall of a failure across sessions, goal-hierarchy retrieval, identity and preference resolution, and the surfacing of a past failure from a passing mention.

```bash
ollama pull qwen3:4b
.venv/bin/python -m streamlit run frontend/demo_app.py
```

The frontend requires the docker-compose stack and Ollama, the same dependencies as the API.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Redis, PostgreSQL, and Neo4j connectivity |
| `POST` | `/memory/ingest` | Store turns and enqueue processing; `user_id` is optional and enables the profile, goal, and cross-session tiers |
| `POST` | `/memory/retrieve` | The assembled context document with provenance |
| `GET` | `/profile/{user_id}` | The resolved identity: attributes with provenance, aliases, decisions, preferences |
| `PATCH` | `/profile/{user_id}` | Set attributes explicitly; explicit values always override inference |
| `POST` | `/profile/{user_id}/alias` | Register an alias; returns 409 on conflict |
| `POST` | `/workbench/tool-call` | Record a tool invocation in L4 |
| `GET` | `/workbench/recent` | Filterable call log with hash-based deduplication |

All routes authenticate with an `X-API-Key` header.

## Quick start

```bash
cp .env.example .env
docker compose up -d redis postgres neo4j
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:3b
```

Run the API, and run the worker in a second terminal:

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

Alternatively, run the closed-loop demo directly; it is self-contained and runs the worker inline:

```bash
.venv/bin/python scripts/demo_closed_loop.py
```

## Testing

The suite contains 312 tests. The full suite runs against the live stack and holds green across repeated runs.

```bash
pytest -m "not integration"   # fast, no Docker required
pytest                        # full: live Redis, PostgreSQL, Neo4j, and Ollama
```

CI runs the fast suite and `ruff check app tests` on every push and pull request.

Aspects of the test design worth noting:

- An Ollama-driven agent harness (`tests/agentic/`) generates realistic conversation traffic under a fixed contract: the model chooses the wording while the scenario fixes the facts. Planned turns carry anchor strings that must survive generation, with a deterministic fallback, so model variance can degrade realism but cannot change test correctness.
- Integration tests verify state through their own SQL and Cypher queries rather than through the API that wrote the data, and cross-tier tests require L2 and L3 to agree on episode ids. This check has caught real corruption twice.
- A test that passes on its first run is deliberately broken once to prove it can fail. The proactive-retrieval test was verified by disabling activation spreading and observing the expected failure before the test was kept.

## Design history

The repository records its decisions and their trade-offs:

- `docs/superpowers/specs/` contains one design document per subsystem, each recording the alternatives that were rejected.
- `steps taken/` contains the original ten-defect audit, a log of obstacles and decisions, and a review that lists the open weaknesses: entity extraction is a small spaCy model plus regexes with measured limits, the recency half-life and over-fetch factor are acknowledged guesses, IVFFlat index creation is still manual, and claim-based tool attribution assumes sequential sessions.

## Configuration

All settings are environment variables (`app/config.py`). The consequential ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model for summarization and goal adjudication |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | `0.25` | Minimum similarity for L2 recall; calibrated, and must be retuned if the embedding model changes |
| `RETRIEVAL_RECENCY_HALF_LIFE_DAYS` | `30` | Half-life for episode recency ranking |
| `TASK_CANDIDATE_LIMIT` | `20` | Open goals shown to the adjudicator |
| `TASK_PLACEMENT_ENABLED` | on | Kill switch for goal-tree placement |
| `WORKBENCH_OUTPUT_MAX_BYTES` / `WORKBENCH_ERROR_MAX_BYTES` | `8192` / `32768` | Payload caps; the error cap is larger because stack traces are the tier's most valuable content |
| `WORKBENCH_MAX_FAILURES_IN_CONTEXT` | `5` | Maximum failures included in a retrieved context |

## License

[MIT](LICENSE)
