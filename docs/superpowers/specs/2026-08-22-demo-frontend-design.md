# Demo Frontend — Design

**Date:** 2026-08-22
**Status:** Approved, implementing
**Branch:** `multi-goal-stringing`
**Builds on:** `scripts/demo_closed_loop.py`, all four tier specs

## Problem

MemCache's proof lives in terminal output: the closed-loop demo prints a
behavior delta, the agentic tests print `[context]` blocks, and nothing lets a
person *click* a story and watch memory change an agent. There is no surface
that shows, side by side, the same model answering the same question with and
without retrieved memory — or that shows *which* episode, entity, and goal ids
actually entered the context.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | UI stack | Streamlit, one page. Chosen for simplicity by request. |
| 2 | Agent model | `qwen3:4b` for the side-by-side answers only. MemCache internals (summarization, adjudication, placement) stay on the measured `qwen2.5:3b` — every calibration in the repo still describes what runs. |
| 3 | Seeding | Live pipeline, cached: clicking a demo runs the REAL ingest path (Ollama summarization, task adjudication, L4 claiming) with progress feedback; the seeded user persists so subsequent runs are instant. A reset control rebuilds from scratch. |
| 4 | Demos | All four: failure recall, goal hierarchy, identity & preferences, passing mention. |
| 5 | API hosting | In-process: Streamlit drives the real ASGI app through `TestClient` with Celery eager — the repo's proven demo/harness pattern. One process, no uvicorn, no worker, no polling. |
| 6 | Verification | Implementation ends with the frontend running in the browser pane, screenshot-reviewed demo by demo. Pure logic (registry, source-table builder, think-stripper) gets unit tests. |

## 1. Architecture

```
streamlit run frontend/demo_app.py
        │
        ├── frontend/demo_app.py       UI only: layout, buttons, rendering
        ├── frontend/demo_runtime.py   stack bootstrap, seed/retrieve/ask/reset
        └── frontend/demos.py          the four preset demo definitions
                │
                └── TestClient(app) + celery eager  ──►  Redis / Postgres+pgvector / Neo4j
                                                          Ollama: qwen2.5:3b (internals)
                                                                  qwen3:4b  (demo agents)
```

`app/` is untouched — the frontend is purely additive. External requirements:
the docker-compose stack and Ollama with both models pulled. The app checks
all four dependencies at startup and renders exact remediation commands
(`docker compose up -d …`, `ollama pull qwen3:4b`) instead of stack traces.

## 2. Demo definitions (`frontend/demos.py`)

```python
@dataclass(frozen=True)
class DemoSession:
    label: str                      # "Monday — session A"
    messages: list[dict[str, str]]  # scripted user/assistant turns
    tool_failures: list[dict]       # recorded to L4 BEFORE the ingest, like the demo script

@dataclass(frozen=True)
class Demo:
    key: str            # slug; user_id = f"demo-ui-{key}" (deterministic → cacheable)
    title: str
    blurb: str          # 2-3 sentences: what this demo proves and what to watch for
    sessions: list[DemoSession]
    plant_hierarchy: bool           # goal-hierarchy demo only, see §4
    retrieval_query: str            # query for the fresh "today" session
    agent_question: str             # the question both agents answer
```

Conversations are scripted (synthetic), not model-generated — the frontend
shows MemCache, not traffic realism; the agentic harness already covers that.
The four demos and their stories:

1. **failure-recall** — the closed-loop script's story verbatim: Monday's
   `alembic` failure recorded to L4 + one ingest; question: "What is your very
   first action resuming the migration, and why?"
2. **goal-hierarchy** — three sessions state "ship telemetry v2", "migrate the
   telemetry schema to ClickHouse" (+ alembic failure), "fix the duplicate
   user_id column"; question: "What are you working on right now, what larger
   goal does it serve, and what should you not repeat?"
3. **identity-preferences** — "I'm Dana Whitfield … Northwind Robotics",
   "we decided to use Rust", "I prefer async standups"; question: "Who are you
   speaking with and how should you run their standup update?"
4. **passing-mention** — session A: ClickHouse+alembic failure ingested;
   session B ingests only an offhand "ClickHouse ingest looked slow";
   retrieval query names *nothing* ("anything else I should keep in mind?");
   question: "Anything the user should know before continuing?"

## 3. Runtime (`frontend/demo_runtime.py`)

All side effects live here; `demo_app.py` never touches a store or a socket.

- `bootstrap()` — `@st.cache_resource`-wrapped: sets Celery eager, opens the
  `TestClient`, returns a handle with client + auth headers. Also
  `stack_status()` returning per-dependency health (API `/health`, Ollama tags
  for both models) for the sidebar.
- `is_seeded(demo) -> bool` — one Postgres probe: any L2 episode owned by the
  demo's user_id.
- `seed(demo, progress_cb)` — for each session: record its `tool_failures`
  via `POST /workbench/tool-call`, then `POST /memory/ingest` (eager → returns
  when L1–L4 are written). Calls `progress_cb(i, n, label)` per session.
  Then, if `plant_hierarchy`, §4. Idempotent by construction: callers check
  `is_seeded` first; `seed` itself starts with a `reset(demo)` so a partial
  earlier seed can never double-write.
- `retrieve(demo) -> dict` — `POST /memory/retrieve` from a fresh session id
  (`demo-ui-<key>-today-<nonce>`; L1 empty, so context is purely cross-session
  memory), `max_tokens=1200`. Returns the raw response (context + sources).
- `ask_agent(question, context | None) -> AgentAnswer(text, seconds)` — one
  `qwen3:4b` generation at temperature 0, the demo script's prompt shape:
  with-memory gets "Context retrieved from your memory system:\n…", without
  gets "You have no memory of previous sessions.". Qwen3 emits
  `<think>…</think>` blocks; `strip_think(text)` removes them (and a
  `thinking` response field is ignored) before display. Model name from env
  `DEMO_AGENT_MODEL`, default `qwen3:4b`.
- `reset(demo)` / `reset_all()` — the demo script's `_cleanup` pattern,
  scoped to `demo-ui-*` ids: Redis session keys, Postgres episodes +
  tool_calls by user_id, Neo4j profile/tasks/sessions/episodes/invoked
  tool-call nodes. Never a global delete.
- `build_source_rows(sources) -> list[dict]` — pure; see §5.

## 4. The planted tree (goal-hierarchy demo)

Measured on this branch (obstacles §16): qwen2.5:3b builds 0 correct
`SUBGOAL_OF` edges from conversation. The demo does not pretend otherwise.
After live seeding, `plant_hierarchy` fetches the demo user's open tasks in
creation order and chains task N+1 under task N via `TaskStore.set_parent`
(each guarded — `TaskHierarchyError` skips that link). The demo's blurb states
the tree is planted by the demo while everything else ran the live pipeline.
If adjudication produced fewer than two tasks, planting is skipped and the
demo still runs (the path line simply has no `under:`). Retrieval and
side-by-side behavior on the planted tree are exactly the machinery the
deterministic suite proves.

## 5. UI (`frontend/demo_app.py`)

**Sidebar** — stack status (four ✅/❌ rows: Redis/Postgres/Neo4j via
`/health`, Ollama with per-model presence), the two model names, demo picker
(radio over the four demos, each with a seeded/unseeded badge), `Reset this
demo` and `Reset all demo data` buttons.

**Main column, top to bottom:**

1. Demo title + blurb, then the scripted conversations: one expander per
   session (`st.chat_message` bubbles), tool failures rendered as a red
   monospace line inside the session that recorded them.
2. One primary button: **Run demo**. On click: seed if `not is_seeded`
   (st.status with per-session progress), then retrieve, then both agent
   calls, storing results in `st.session_state` keyed by demo so switching
   demos doesn't lose results.
3. **Side by side** — `st.columns(2)`: left "🧠 With MemCache", right
   "🚫 Without memory", each showing the agent's answer and its latency; the
   question shown once above both.
4. **What the agent was given** — the retrieved context verbatim in a code
   block, then **Retrieved memory, structured**: `st.dataframe` of
   `build_source_rows`, columns `Tier | Type | ID | Detail | Score | Path`:
   - episode sources → `ID = episode {episode_id}`
   - entity sources → `ID = entity {name}` (Entity nodes are keyed by
     normalized name — that IS the entity id in this graph)
   - task sources → `ID = goal {task_id}`, Detail carries title + status,
     and the active task's row carries its full `lineage` list
   - tool-call sources → `ID = tool_call {tool_call_id}`
   - recent_message/profile rows keep their tier and a short detail
   - `Score` = similarity/decayed score or activation where present;
     `Path` = the proactive `via` chain, else empty
   A caption under the table totals the ids: "Retrieved: N episodes, M
   entities, K goals, J tool calls."

**Error handling:** every button handler wraps in try/except → `st.error`
with the exception text; startup dependency failures render remediation
commands and disable Run. A missing `qwen3:4b` disables only the side-by-side
(seeding still works) with the pull command shown.

## 6. Configuration

| Env | Default | Meaning |
|-----|---------|---------|
| `DEMO_AGENT_MODEL` | `qwen3:4b` | Model for the side-by-side answers. |
| `DEMO_AGENT_TIMEOUT` | `120` | Seconds per agent generation. |

Everything else comes from the existing `app/config.py` settings. Streamlit
is added to `requirements.txt` (`streamlit>=1.37`). A `.claude/launch.json`
entry (`demo-frontend`, port 8501, headless) makes the app previewable in the
browser pane.

## 7. Testing & verification

**Unit (gates, no Streamlit import):** `tests/test_demo_frontend.py`
- registry sanity: 4 demos, unique keys/user_ids, every session non-empty,
  tool_failures carry tool_name+error, exactly one demo plants hierarchy;
- `build_source_rows` over a fixture `sources` list containing every source
  type the retrieval layer emits (episode, task, proactive_task,
  proactive_entity, proactive_tool_failure, tool_failure, profile_identity,
  recent_message) — asserts tier/ID/Score/Path extraction, including lineage
  on the task row;
- `strip_think` on nested/unclosed/absent think blocks.

**Interactive (the user-requested protocol, not CI):** launch via the browser
pane, then for each of the four demos: screenshot the idle state, click Run,
screenshot seeding progress, screenshot the finished side-by-side + table, and
review that (a) the with-memory answer uses the seeded facts, (b) the table
shows the expected id kinds for that demo (goal ids + lineage in
goal-hierarchy, proactive path in passing-mention, entity rows in identity),
(c) reset returns the badge to unseeded. Findings fixed on the spot; the
final state of each demo screenshot-verified once more after fixes.

## 8. Scope

In: the three `frontend/` files, unit tests, requirements + launch entry,
README run instructions, browser verification.

Out: authentication (localhost demo), deployment/packaging, streaming token
output, letting users type their own conversations, multi-user isolation
beyond the fixed `demo-ui-*` namespace, any change under `app/`.
