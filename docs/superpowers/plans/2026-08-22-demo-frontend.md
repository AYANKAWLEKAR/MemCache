# Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-page Streamlit app with four clickable preset demos that seed L2/L3/L4 through the real pipeline and show, side by side, the same qwen3:4b agent answering with and without MemCache context — plus a structured table of the episode/entity/goal ids that entered that context.

**Architecture:** Three new files under `frontend/`, nothing under `app/` changes. `demos.py` holds the four scripted demo definitions; `demo_runtime.py` holds every side effect (in-process `TestClient` + eager Celery bootstrap, seed/retrieve/ask/reset) and the pure helpers (`strip_think`, `build_source_rows`) — it never imports Streamlit, so its logic is unit-testable; `demo_app.py` is UI only. Verification is unit tests for the pure parts plus browser-pane screenshot review of all four demos.

**Tech Stack:** Streamlit ≥1.37, FastAPI `TestClient`, Celery eager, httpx → Ollama (`qwen3:4b` agents, `qwen2.5:3b` internals), live Redis/Postgres/Neo4j.

**Spec:** `docs/superpowers/specs/2026-08-22-demo-frontend-design.md`

## Global Constraints

- `app/` is untouched. The frontend is purely additive.
- Demo agent model: env `DEMO_AGENT_MODEL`, default `qwen3:4b`; timeout env `DEMO_AGENT_TIMEOUT`, default `120`. MemCache internals stay on `settings.ollama_model` (`qwen2.5:3b`).
- Demo user ids are `demo-ui-<key>`; every seeded session id starts with that user id, so cleanup can scope by prefix. Never a global delete.
- `demo_runtime.py` and `demos.py` must import without Streamlit installed-or-not mattering (no `import streamlit` outside `demo_app.py`).
- Deviation from spec §3, recorded here: `bootstrap()` caches via a module-global singleton instead of `@st.cache_resource` — same lifetime (Streamlit re-runs share the process), keeps the runtime Streamlit-free.
- Run tests with `.venv/bin/python -m pytest`. Stack up: `docker compose up -d redis postgres neo4j`; Ollama serving both models.
- Commit after every task, `git add -f` for gitignored `*.md`/config, trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|------|----------------|
| `frontend/__init__.py` (new, empty) | Package marker so tests import `frontend.*`. |
| `frontend/demos.py` (new) | `DemoSession`, `Demo`, `DEMOS` registry (4 entries). Pure data. |
| `frontend/demo_runtime.py` (new) | Bootstrap, health, seed/is_seeded/retrieve/ask_agent/reset(+all), `strip_think`, `build_source_rows`. No Streamlit. |
| `frontend/demo_app.py` (new) | Streamlit page. |
| `tests/test_demo_frontend.py` (new) | Registry sanity, `build_source_rows`, `strip_think` (pure); one integration seed/retrieve/reset round-trip. |
| `requirements.txt` (modify) | `streamlit>=1.37`. |
| `.claude/launch.json` (new) | `demo-frontend` entry, port 8501. |
| `README.md` (modify) | "Demo frontend" run section. |

---

### Task 1: Demo registry + pure helpers, with tests

**Files:**
- Create: `frontend/__init__.py`, `frontend/demos.py`
- Create: `frontend/demo_runtime.py` (pure parts only in this task)
- Test: `tests/test_demo_frontend.py`

**Interfaces:**
- Produces: `DemoSession(label, messages, tool_failures)`, `Demo(key, title, blurb, sessions, plant_hierarchy, retrieval_query, agent_question)` with property `user_id -> f"demo-ui-{key}"` and `session_id(i) -> f"demo-ui-{key}-s{i}"`; `DEMOS: list[Demo]`; `strip_think(text) -> str`; `build_source_rows(sources) -> list[dict]` with keys `Tier, Type, ID, Detail, Score, Path`; `count_kinds(rows) -> dict` with keys `episodes, entities, goals, tool_calls`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_demo_frontend.py`:

```python
"""Demo-frontend logic: registry sanity, source-table building, think-stripping.

Pure — no Streamlit, no stores, no model. The one integration round-trip
(seed → probe → retrieve → reset) lives at the bottom under the integration
marker.
"""

from __future__ import annotations

import pytest

from frontend.demo_runtime import build_source_rows, count_kinds, strip_think
from frontend.demos import DEMOS


# ------------------------------------------------------------- registry


def test_registry_has_four_well_formed_demos():
    assert len(DEMOS) == 4
    keys = [d.key for d in DEMOS]
    assert len(set(keys)) == 4
    for d in DEMOS:
        assert d.user_id == f"demo-ui-{d.key}"
        assert d.sessions, d.key
        assert d.retrieval_query and d.agent_question and d.blurb
        for i, s in enumerate(d.sessions):
            assert d.session_id(i).startswith(d.user_id)
            assert s.messages, f"{d.key} session {i} empty"
            for m in s.messages:
                assert m["role"] in {"user", "assistant"} and m["content"].strip()
            for tf in s.tool_failures:
                assert tf["tool_name"] and tf["error"]
    assert sum(1 for d in DEMOS if d.plant_hierarchy) == 1


def test_every_demo_seeds_a_failure_or_a_fact_the_question_needs():
    """Each demo's script must contain the anchor its side-by-side hinges on."""
    text = {d.key: " ".join(m["content"].lower() for s in d.sessions for m in s.messages)
            for d in DEMOS}
    assert "alembic" in text["failure-recall"]
    assert "telemetry v2" in text["goal-hierarchy"]
    assert "dana whitfield" in text["identity-preferences"]
    assert "clickhouse" in text["passing-mention"]


# ---------------------------------------------------------- strip_think


def test_strip_think_removes_closed_and_unclosed_blocks():
    assert strip_think("<think>hmm</think>Answer.") == "Answer."
    assert strip_think("A<think>x</think>B<think>y</think>C") == "ABC"
    assert strip_think("<think>never closed... Answer buried") == ""
    assert strip_think("no think here") == "no think here"
    assert strip_think("") == ""


# ---------------------------------------------------- build_source_rows


FIXTURE_SOURCES = [
    {"type": "recent_message", "tier": "L1", "details": {"session_id": "s", "index": 0}},
    {"type": "profile_identity", "tier": "L3", "details": {"user_id": "u", "display_name": "Dana"}},
    {"type": "task", "tier": "L3", "details": {"task_id": "T-LEAF", "title": "Fix column",
     "status": "open", "lineage": ["T-LEAF", "T-MID", "T-ROOT"], "depth": 2}},
    {"type": "episode", "tier": "L2", "details": {"episode_id": 41, "session_id": "old",
     "similarity": 0.61, "decayed_score": 0.55}},
    {"type": "tool_failure", "tier": "L4", "details": {"tool_call_id": 9, "tool_name": "alembic",
     "task_id": "T-ROOT", "error_head": "DuplicateColumn: boom"}},
    {"type": "proactive_episode", "tier": "L3", "details": {"episode_id": 42, "session_id": "old2",
     "activation": 0.144, "via": "Task:T-ROOT -ADVANCES(1)-> 42", "path": [["Task:T-ROOT", "ADVANCES", 1, "Episode:42"]], "is_seed": False}},
    {"type": "proactive_entity", "tier": "L3", "details": {"name": "clickhouse",
     "activation": 0.4, "via": "alembic -RELATED_TO(6)-> clickhouse", "path": [], "is_seed": False}},
    {"type": "proactive_task", "tier": "L3", "details": {"task_id": "T-OTHER",
     "activation": 0.09, "via": "x", "path": [], "is_seed": False}},
    {"type": "proactive_tool_failure", "tier": "L4", "details": {"tool_call_id": 10,
     "tool_name": "alembic", "status": "error", "activation": 0.06, "via": "p", "path": [], "is_seed": False}},
    {"type": "decision", "tier": "L3", "details": {"text": "use Rust"}},
]


def test_build_source_rows_extracts_ids_scores_and_paths():
    rows = build_source_rows(FIXTURE_SOURCES)
    assert len(rows) == len(FIXTURE_SOURCES)
    by_id = {r["ID"]: r for r in rows}

    assert by_id["episode 41"]["Tier"] == "L2"
    assert by_id["episode 41"]["Score"] == 0.55          # decayed beats raw
    assert by_id["episode 42"]["Score"] == 0.144         # activation
    assert by_id["episode 42"]["Path"] == "Task:T-ROOT -ADVANCES(1)-> 42"

    goal = by_id["goal T-LEAF"]
    assert "Fix column" in goal["Detail"] and "open" in goal["Detail"]
    assert "T-LEAF ▸ T-MID ▸ T-ROOT" in goal["Detail"]   # lineage surfaces

    assert by_id["entity clickhouse"]["Score"] == 0.4
    assert "DuplicateColumn" in by_id["tool_call 9"]["Detail"]
    assert by_id["tool_call 10"]["Tier"] == "L4"
    assert by_id["goal T-OTHER"]["ID"] == "goal T-OTHER"
    # Rows without a natural id still render.
    assert any(r["Type"] == "recent_message" for r in rows)
    assert any("use Rust" in r["Detail"] for r in rows)


def test_count_kinds_totals_by_id_kind():
    rows = build_source_rows(FIXTURE_SOURCES)
    counts = count_kinds(rows)
    assert counts == {"episodes": 2, "entities": 1, "goals": 2, "tool_calls": 2}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_demo_frontend.py -q`
Expected: collection error — `ModuleNotFoundError: frontend`.

- [ ] **Step 3: Implement**

`frontend/__init__.py`: empty file.

`frontend/demos.py`:

```python
"""The four preset demos: scripted synthetic conversations + seeded failures.

Pure data. Conversations are scripted, not model-generated — this surface
shows MemCache, not traffic realism (the agentic harness covers that). Each
demo's `blurb` is user-facing copy and states honestly what is scripted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DemoSession:
    label: str
    messages: list[dict[str, str]]
    tool_failures: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    blurb: str
    sessions: list[DemoSession]
    retrieval_query: str
    agent_question: str
    plant_hierarchy: bool = False

    @property
    def user_id(self) -> str:
        return f"demo-ui-{self.key}"

    def session_id(self, index: int) -> str:
        return f"{self.user_id}-s{index + 1}"


_ALEMBIC_FAILURE = {
    "tool_name": "alembic",
    "status": "error",
    "args": {"command": "upgrade", "revision": "0042"},
    "error": "DuplicateColumn: column user_id already exists on episodes",
    "duration_ms": 412,
}

FAILURE_RECALL = Demo(
    key="failure-recall",
    title="Failure recall",
    blurb=(
        "Monday, session A: a migration fails (scripted, recorded to the L4 "
        "workbench for real) and the conversation is ingested. Thursday, a "
        "brand-new session resumes the work. Watch the with-memory agent "
        "address the exact DuplicateColumn error while the other starts blind."
    ),
    sessions=[
        DemoSession(
            label="Monday — session A",
            messages=[
                {"role": "user", "content": (
                    "I'm trying to migrate the telemetry schema to ClickHouse, "
                    "but the alembic migration just failed."
                )},
                {"role": "assistant", "content": "Noted — the migration errored."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
    ],
    retrieval_query="Picking the telemetry migration back up. What should I know?",
    agent_question=(
        "You are resuming work on the telemetry schema migration. "
        "What is your very first action, and why? Answer in one short sentence."
    ),
)

GOAL_HIERARCHY = Demo(
    key="goal-hierarchy",
    title="Goal hierarchy",
    blurb=(
        "Three sessions state a goal, a subgoal, and a sub-subgoal; the "
        "SUBGOAL_OF tree is planted by the demo (a 3B judge cannot infer "
        "direction — measured), while ingestion, retrieval, and ranking all "
        "run the live pipeline. Watch the Current-task path line and the "
        "parent goal's failure surface in the leaf's context."
    ),
    sessions=[
        DemoSession(
            label="Two weeks ago — session A",
            messages=[
                {"role": "user", "content": "My overall goal this quarter is to ship telemetry v2."},
                {"role": "assistant", "content": "Understood — telemetry v2 is the objective."},
            ],
        ),
        DemoSession(
            label="Last week — session B",
            messages=[
                {"role": "user", "content": (
                    "My goal is to migrate the telemetry schema to ClickHouse, "
                    "and the alembic migration just failed."
                )},
                {"role": "assistant", "content": "Recorded the failed migration."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
        DemoSession(
            label="Yesterday — session C",
            messages=[
                {"role": "user", "content": (
                    "My goal is to fix the duplicate user_id column on the episodes table."
                )},
                {"role": "assistant", "content": "On it — the duplicate column fix."},
            ],
        ),
    ],
    retrieval_query="Getting back to work. Where was I and what should I avoid?",
    agent_question=(
        "What are you working on right now, what larger goal does it serve, "
        "and what must you not repeat? Answer in at most three short sentences."
    ),
    plant_hierarchy=True,
)

IDENTITY_PREFERENCES = Demo(
    key="identity-preferences",
    title="Identity & preferences",
    blurb=(
        "Session A introduces Dana Whitfield of Northwind Robotics, a decision "
        "(Rust) and a preference (async standups). A fresh session then has to "
        "know who it is talking to — aliases collapse to one profile."
    ),
    sessions=[
        DemoSession(
            label="Session A",
            messages=[
                {"role": "user", "content": (
                    "Hi, I'm Dana Whitfield and I work at Northwind Robotics."
                )},
                {"role": "assistant", "content": "Nice to meet you, Dana."},
                {"role": "user", "content": "We decided to use Rust for the control backend."},
                {"role": "assistant", "content": "Rust for the backend — noted."},
                {"role": "user", "content": "I prefer async standups over daily video calls."},
                {"role": "assistant", "content": "Async standups it is."},
            ],
        ),
    ],
    retrieval_query="Who am I and how do we work together?",
    agent_question=(
        "Who are you speaking with, where do they work, and how should their "
        "standup update be run? Answer in two short sentences."
    ),
)

PASSING_MENTION = Demo(
    key="passing-mention",
    title="Passing mention",
    blurb=(
        "Session A ties ClickHouse to a failed alembic migration. Session B "
        "only mentions ClickHouse offhand, and the retrieval query names "
        "nothing at all — the failure must arrive through the weighted graph, "
        "with its activation path shown in the table below."
    ),
    sessions=[
        DemoSession(
            label="Earlier — session A",
            messages=[
                {"role": "user", "content": (
                    "The alembic migration for the ClickHouse telemetry schema just failed."
                )},
                {"role": "assistant", "content": "Logged the ClickHouse migration failure."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
        DemoSession(
            label="Today — session B",
            messages=[
                {"role": "user", "content": "Also, ClickHouse ingest looked slow yesterday."},
                {"role": "assistant", "content": "Noted about the ingest speed."},
            ],
        ),
    ],
    retrieval_query="anything else I should keep in mind before I continue?",
    agent_question=(
        "Anything the user should know before continuing their work? "
        "Answer in one short sentence."
    ),
)

DEMOS: list[Demo] = [FAILURE_RECALL, GOAL_HIERARCHY, IDENTITY_PREFERENCES, PASSING_MENTION]
```

`frontend/demo_runtime.py` (pure parts; side effects arrive in Task 2):

```python
"""Runtime for the demo frontend: every side effect, plus the pure helpers.

Deliberately Streamlit-free — `demo_app.py` owns rendering; this module owns
doing. The pure helpers (`strip_think`, `build_source_rows`, `count_kinds`)
are unit-tested without any store or model.
"""

from __future__ import annotations

import re
from typing import Any

#: <think>…</think> (qwen3 thinking traces), including an unclosed trailing block.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove qwen3 thinking blocks; an unclosed block swallows to the end."""
    return _THINK.sub("", text or "").strip()


def _score(details: dict) -> float | None:
    for key in ("activation", "decayed_score", "similarity"):
        if details.get(key) is not None:
            return details[key]
    return None


def build_source_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provenance sources → table rows: Tier | Type | ID | Detail | Score | Path.

    The ID column is the point of the table: `episode <L2 row id>`,
    `entity <normalized name>` (Entity nodes are keyed by name — that IS the
    entity id in this graph), `goal <task uuid>`, `tool_call <L4 row id>`.
    """
    rows: list[dict[str, Any]] = []
    for src in sources:
        d = src.get("details", {}) or {}
        stype = src.get("type", "?")
        id_, detail = "—", ""
        if "episode_id" in d:
            id_ = f"episode {d['episode_id']}"
            detail = f"session {d.get('session_id', '?')}"
        elif "task_id" in d and stype in {"task", "proactive_task"}:
            id_ = f"goal {d['task_id']}"
            detail = f"{d.get('title', '')} ({d.get('status', '?')})".strip()
            if d.get("lineage"):
                detail += f" — lineage: {' ▸ '.join(d['lineage'])}"
        elif "tool_call_id" in d:
            id_ = f"tool_call {d['tool_call_id']}"
            detail = f"{d.get('tool_name', '?')}: {d.get('error_head') or d.get('status', '')}"
        elif "name" in d:
            id_ = f"entity {d['name']}"
            detail = d.get("display_name") or ""
        elif "text" in d:
            detail = d["text"]
        elif "display_name" in d or "user_id" in d:
            id_ = f"profile {d.get('user_id', '?')}"
            detail = d.get("display_name") or d.get("key", "")
        else:
            detail = ", ".join(f"{k}={v}" for k, v in d.items() if k not in {"path", "via"})
        rows.append({
            "Tier": src.get("tier", "?"),
            "Type": stype,
            "ID": id_,
            "Detail": detail,
            "Score": _score(d),
            "Path": d.get("via", "") or "",
        })
    return rows


def count_kinds(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Caption totals: how many episode/entity/goal/tool-call ids were retrieved."""
    prefixes = {"episodes": "episode ", "entities": "entity ",
                "goals": "goal ", "tool_calls": "tool_call "}
    return {
        kind: sum(1 for r in rows if str(r["ID"]).startswith(prefix))
        for kind, prefix in prefixes.items()
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_demo_frontend.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/__init__.py frontend/demos.py frontend/demo_runtime.py tests/test_demo_frontend.py
git commit -m "feat: demo registry and pure frontend helpers (source table, think-strip)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Runtime side effects + one integration round-trip

**Files:**
- Modify: `frontend/demo_runtime.py` (append)
- Modify: `requirements.txt` (add `streamlit>=1.37`)
- Test: `tests/test_demo_frontend.py` (append)

**Interfaces:**
- Consumes: `Demo`/`DEMOS` from Task 1; `TaskStore`, `app.main.app`, `app.config.settings`.
- Produces: `Handle(client, auth)`, `bootstrap() -> Handle`, `stack_status() -> dict[str, tuple[bool, str]]`, `is_seeded(demo) -> bool`, `seed(demo, progress_cb=None) -> None`, `retrieve(demo) -> dict`, `AgentAnswer(text, seconds)`, `ask_agent(question, context) -> AgentAnswer`, `agent_model_available() -> bool`, `reset(demo) -> None`, `reset_all() -> None`, `DEMO_AGENT_MODEL: str`.

- [ ] **Step 1: Install streamlit and pull the model**

```bash
.venv/bin/pip install "streamlit>=1.37"
ollama pull qwen3:4b
```

Append `streamlit>=1.37` to `requirements.txt` (with a `# demo frontend` comment). Verify: `curl -s localhost:11434/api/tags | grep -o 'qwen3:4b'` prints `qwen3:4b`.

- [ ] **Step 2: Write the failing integration test**

Append to `tests/test_demo_frontend.py`:

```python
# ------------------------------------------------- integration round-trip


@pytest.mark.integration
def test_seed_retrieve_reset_round_trip_failure_recall():
    """The cheapest demo (one session) through the REAL pipeline: seed writes
    all tiers, retrieve surfaces the failure with provenance ids, reset wipes.
    Asserts via the runtime's own API — the tiers themselves are covered by
    the main suite; this is the frontend's contract."""
    from frontend.demo_runtime import bootstrap, is_seeded, reset, retrieve, seed
    from frontend.demos import FAILURE_RECALL as demo

    bootstrap()
    reset(demo)
    assert not is_seeded(demo)

    calls: list[str] = []
    seed(demo, progress_cb=lambda i, n, label: calls.append(label))
    assert calls, "progress callback never fired"
    assert is_seeded(demo)

    result = retrieve(demo)
    assert "duplicatecolumn" in result["context"].lower()
    from frontend.demo_runtime import build_source_rows, count_kinds
    counts = count_kinds(build_source_rows(result["sources"]))
    assert counts["tool_calls"] >= 1, counts
    assert counts["episodes"] >= 1, counts

    reset(demo)
    assert not is_seeded(demo)
```

Run: `.venv/bin/python -m pytest tests/test_demo_frontend.py -k round_trip -q`
Expected: FAIL — `ImportError: cannot import name 'bootstrap'`.

- [ ] **Step 3: Implement the side-effect half**

Append to `frontend/demo_runtime.py`:

```python
# ---------------------------------------------------------------- runtime

import os
import time
from dataclasses import dataclass

import httpx

DEMO_AGENT_MODEL = os.environ.get("DEMO_AGENT_MODEL", "qwen3:4b")
DEMO_AGENT_TIMEOUT = float(os.environ.get("DEMO_AGENT_TIMEOUT", "120"))
_USER_PREFIX = "demo-ui-"


@dataclass(frozen=True)
class Handle:
    client: Any
    auth: dict[str, str]


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    seconds: float


_HANDLE: Handle | None = None


def bootstrap() -> Handle:
    """In-process API: real ASGI app, Celery eager — the repo's demo pattern.

    Module-global singleton: Streamlit reruns share the process, so this is
    the same lifetime @st.cache_resource would give without importing
    Streamlit here.
    """
    global _HANDLE
    if _HANDLE is None:
        from app.workers.celery_app import celery_app

        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = True

        from fastapi.testclient import TestClient

        from app.config import settings
        from app.main import app

        client = TestClient(app)
        client.__enter__()  # lifespan startup; process exit tears it down
        _HANDLE = Handle(
            client=client,
            auth={"X-API-Key": next(iter(settings.get_valid_api_keys()))},
        )
    return _HANDLE


def stack_status() -> dict[str, tuple[bool, str]]:
    """Dependency health for the sidebar: API tiers + both Ollama models."""
    from app.config import settings

    out: dict[str, tuple[bool, str]] = {}
    try:
        h = bootstrap()
        resp = h.client.get("/health").json()
        for name in ("redis", "postgres", "neo4j"):
            state = str(resp.get(name, {}).get("status", "unknown"))
            out[name] = (state == "ok", state)
    except Exception as exc:  # stack down: report, never crash the page
        for name in ("redis", "postgres", "neo4j"):
            out[name] = (False, f"unreachable: {exc}")
    try:
        tags = httpx.get(
            settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5
        ).json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        for model in (settings.ollama_model, DEMO_AGENT_MODEL):
            out[f"ollama {model}"] = (model in names, "pulled" if model in names else "missing")
    except Exception as exc:
        out["ollama"] = (False, f"unreachable: {exc}")
    return out


def agent_model_available() -> bool:
    return stack_status().get(f"ollama {DEMO_AGENT_MODEL}", (False, ""))[0]


def is_seeded(demo) -> bool:
    """One probe: does this demo's user own any L2 episode?"""
    from app.api import services as api_services

    with api_services.get_postgres_engine().connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT 1 FROM episodes WHERE user_id = %s LIMIT 1", (demo.user_id,)
        ).fetchone()
    return row is not None


def seed(demo, progress_cb=None) -> None:
    """Run the demo's sessions through the REAL pipeline, oldest first.

    Starts with a reset so a partial earlier seed can never double-write.
    Eager Celery makes each ingest synchronous: when the POST returns, L1-L4
    for that session are written.
    """
    h = bootstrap()
    reset(demo)
    n = len(demo.sessions)
    for i, session in enumerate(demo.sessions):
        if progress_cb:
            progress_cb(i, n, session.label)
        sid = demo.session_id(i)
        for tf in session.tool_failures:
            r = h.client.post(
                "/workbench/tool-call",
                headers=h.auth,
                json={"session_id": sid, "user_id": demo.user_id, **tf},
            )
            r.raise_for_status()
        r = h.client.post(
            "/memory/ingest",
            headers=h.auth,
            json={"session_id": sid, "user_id": demo.user_id, "messages": session.messages},
        )
        assert r.status_code == 202, r.text
    if demo.plant_hierarchy:
        _plant_hierarchy(demo)


def _plant_hierarchy(demo) -> None:
    """Chain the seeded tasks oldest→newest via SUBGOAL_OF.

    The honest workaround, stated in the demo copy: measured on this branch,
    the 3B judge builds 0 correct edges, so the demo plants the tree and lets
    the proven retrieval machinery do the rest. Guarded per link — a merged
    or missing task skips its link rather than failing the seed.
    """
    from app.api import services as api_services
    from app.services.task_store import TaskHierarchyError, TaskStore

    store = TaskStore(api_services.get_neo4j_driver())
    tasks = sorted(
        store.list_open_tasks(demo.user_id, limit=20), key=lambda t: t.created_at
    )
    for parent, child in zip(tasks, tasks[1:]):
        try:
            store.set_parent(child.id, parent.id)
        except TaskHierarchyError:
            pass


def retrieve(demo) -> dict:
    """Retrieve from a FRESH session: empty L1, so the context is purely
    cross-session memory."""
    h = bootstrap()
    fresh = f"{demo.user_id}-today-{int(time.time())}"
    r = h.client.post(
        "/memory/retrieve",
        headers=h.auth,
        json={
            "session_id": fresh,
            "user_id": demo.user_id,
            "query": demo.retrieval_query,
            "max_tokens": 1200,
        },
    )
    r.raise_for_status()
    return r.json()


def ask_agent(question: str, context: str | None) -> AgentAnswer:
    """One qwen3:4b generation, the closed-loop demo's prompt shape."""
    from app.config import settings

    if context:
        prompt = f"Context retrieved from your memory system:\n{context}\n\n{question}"
    else:
        prompt = f"You have no memory of previous sessions.\n\n{question}"
    start = time.monotonic()
    resp = httpx.post(
        settings.ollama_base_url.rstrip("/") + "/api/generate",
        json={
            "model": DEMO_AGENT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=DEMO_AGENT_TIMEOUT,
    )
    resp.raise_for_status()
    text = strip_think((resp.json().get("response") or "").strip())
    return AgentAnswer(text=text, seconds=time.monotonic() - start)


def reset(demo) -> None:
    _reset_user(demo.user_id)


def reset_all() -> None:
    for key_user in [d.user_id for d in __import__("frontend.demos", fromlist=["DEMOS"]).DEMOS]:
        _reset_user(key_user)


def _reset_user(uid: str) -> None:
    """Scoped cleanup across all tiers — the demo script's pattern, by prefix.

    Every session id starts with the user id, so `STARTS WITH uid` and
    `session:uid*` bound the deletes to this demo's data.
    """
    assert uid.startswith(_USER_PREFIX), uid  # never wipe non-demo data
    import redis as redis_lib

    from app.api import services as api_services
    from app.config import settings

    r = redis_lib.from_url(settings.redis_url, decode_responses=True)
    try:
        for key in r.scan_iter(f"session:{uid}*"):
            r.delete(key)
    finally:
        r.close()
    with api_services.get_postgres_engine().begin() as conn:
        conn.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (uid,))
        conn.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    driver = api_services.get_neo4j_driver()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            DETACH DELETE p, t, a
            """,
            uid=uid,
        )
        s.run(
            """
            MATCH (se:Session) WHERE se.id STARTS WITH $uid
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall)
            DETACH DELETE se, ep, dp, tc
            """,
            uid=uid,
        )
        s.run(
            "MATCH (:UserProfile)-[r:HAS_ALIAS]->(:Entity {name: 'dana whitfield'}) DELETE r"
        )
        s.run("MATCH (:UserProfile)-[r:HAS_ALIAS]->(:Entity {name: 'dana'}) DELETE r")
```

Note the alias release: the identity demo claims "dana whitfield", and alias
conflicts are permanent by design — without the release, a reseed after reset
would 409. Releasing only the HAS_ALIAS edges mirrors the test fixture
`release_person_names`.

Fix the clumsy `reset_all` import while implementing: use a plain

```python
def reset_all() -> None:
    from frontend.demos import DEMOS

    for d in DEMOS:
        _reset_user(d.user_id)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_demo_frontend.py -q`
Expected: all PASS (round-trip takes ~15-40s: one real Ollama summarization + adjudication).

- [ ] **Step 5: Commit**

```bash
git add frontend/demo_runtime.py tests/test_demo_frontend.py requirements.txt
git commit -m "feat: demo runtime — in-process seed/retrieve/ask/reset with scoped cleanup

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The Streamlit page

**Files:**
- Create: `frontend/demo_app.py`
- Create: `.claude/launch.json`
- Modify: `README.md` (run section)

**Interfaces:**
- Consumes: everything Task 2 produced, `DEMOS` from Task 1.
- Produces: `streamlit run frontend/demo_app.py` serves the page on :8501.

- [ ] **Step 1: Write the app**

`frontend/demo_app.py`:

```python
"""MemCache demo frontend. UI only — every side effect lives in demo_runtime.

    .venv/bin/python -m streamlit run frontend/demo_app.py
"""

from __future__ import annotations

import streamlit as st

from frontend import demo_runtime as rt
from frontend.demos import DEMOS

st.set_page_config(page_title="MemCache Demos", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------- sidebar

st.sidebar.title("🧠 MemCache")
st.sidebar.caption("Memory infrastructure for LLM agents — live demos")

status = rt.stack_status()
stack_ok = all(ok for ok, _ in status.values())
with st.sidebar.expander("Stack status", expanded=not stack_ok):
    for name, (ok, detail) in status.items():
        st.write(("✅" if ok else "❌") + f" {name} — {detail}")
    if not stack_ok:
        st.code("docker compose up -d redis postgres neo4j\nollama pull qwen3:4b", language="bash")

agent_ok = status.get(f"ollama {rt.DEMO_AGENT_MODEL}", (False, ""))[0]
st.sidebar.caption(
    f"agents: `{rt.DEMO_AGENT_MODEL}` · internals: `qwen2.5:3b`"
)

titles = {d.title: d for d in DEMOS}
seeded_now = {d.key: rt.is_seeded(d) for d in DEMOS} if stack_ok else {}
choice = st.sidebar.radio(
    "Demo",
    list(titles),
    format_func=lambda t: f"{t} {'●' if seeded_now.get(titles[t].key) else '○'}",
)
demo = titles[choice]
st.sidebar.caption("● seeded — reruns are instant · ○ will seed on first run")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Reset this demo", use_container_width=True, disabled=not stack_ok):
    rt.reset(demo)
    st.session_state.pop(f"result-{demo.key}", None)
    st.rerun()
if col_b.button("Reset all", use_container_width=True, disabled=not stack_ok):
    rt.reset_all()
    for d in DEMOS:
        st.session_state.pop(f"result-{d.key}", None)
    st.rerun()

# -------------------------------------------------------------------- main

st.title(demo.title)
st.write(demo.blurb)

st.subheader("The scripted conversations")
for i, session in enumerate(demo.sessions):
    with st.expander(session.label, expanded=False):
        for m in session.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        for tf in session.tool_failures:
            st.error(f"$ {tf['tool_name']} — recorded to L4: `{tf['error']}`")

run = st.button(
    "▶ Run demo",
    type="primary",
    disabled=not stack_ok,
    help=None if stack_ok else "Fix the stack status in the sidebar first",
)

if run:
    try:
        if not rt.is_seeded(demo):
            with st.status("Seeding memory through the real pipeline…", expanded=True) as box:
                def cb(i, n, label):
                    box.write(f"({i + 1}/{n}) {label}: Ollama summarize → embed → graph → claim")
                rt.seed(demo, progress_cb=cb)
                box.update(label="Seeded — L2, L3, and L4 written", state="complete")
        with st.spinner("Retrieving memory for a brand-new session…"):
            retrieved = rt.retrieve(demo)
        answers = {}
        if agent_ok:
            with st.spinner(f"Asking {rt.DEMO_AGENT_MODEL} twice — with and without memory…"):
                answers["with"] = rt.ask_agent(demo.agent_question, retrieved["context"])
                answers["without"] = rt.ask_agent(demo.agent_question, None)
        st.session_state[f"result-{demo.key}"] = {"retrieved": retrieved, "answers": answers}
        st.rerun()
    except Exception as exc:  # never a stack trace on the page
        st.error(f"Demo run failed: {exc}")

result = st.session_state.get(f"result-{demo.key}")
if result:
    retrieved, answers = result["retrieved"], result["answers"]

    st.subheader("Same model, same question")
    st.markdown(f"**Q:** {demo.agent_question}")
    if answers:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🧠 With MemCache")
            st.info(answers["with"].text or "(empty answer)")
            st.caption(f"{answers['with'].seconds:.1f}s · {rt.DEMO_AGENT_MODEL}")
        with right:
            st.markdown("#### 🚫 Without memory")
            st.warning(answers["without"].text or "(empty answer)")
            st.caption(f"{answers['without'].seconds:.1f}s · {rt.DEMO_AGENT_MODEL}")
    else:
        st.warning(
            f"`{rt.DEMO_AGENT_MODEL}` is not pulled — side-by-side skipped. "
            f"Run `ollama pull {rt.DEMO_AGENT_MODEL}` and rerun."
        )

    st.subheader("What the with-memory agent was given")
    st.code(retrieved["context"], language=None)

    st.subheader("Retrieved memory, structured")
    rows = rt.build_source_rows(retrieved["sources"])
    st.dataframe(rows, use_container_width=True, hide_index=True)
    c = rt.count_kinds(rows)
    st.caption(
        f"Retrieved: {c['episodes']} episodes · {c['entities']} entities · "
        f"{c['goals']} goals · {c['tool_calls']} tool calls — every context "
        "line above is attributable to one of these rows."
    )
```

- [ ] **Step 2: Launch config + README**

`.claude/launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "demo-frontend",
      "runtimeExecutable": ".venv/bin/python",
      "runtimeArgs": ["-m", "streamlit", "run", "frontend/demo_app.py",
                       "--server.port", "8501", "--server.headless", "true"],
      "port": 8501
    }
  ]
}
```

README, after the Quick start section:

```markdown
## Demo frontend

A one-page Streamlit app with four clickable demos — each seeds L2/L3/L4
through the real pipeline, then shows the same `qwen3:4b` agent answering the
same question with and without MemCache context, plus a table of exactly which
episode / entity / goal / tool-call ids entered the context.

​```bash
ollama pull qwen3:4b
.venv/bin/python -m streamlit run frontend/demo_app.py
​```

(Requires the docker-compose stack and Ollama, same as the API.)
```

- [ ] **Step 3: Smoke-import**

Run: `.venv/bin/python -c "import ast; ast.parse(open('frontend/demo_app.py').read())" && .venv/bin/python -c "from frontend import demo_runtime, demos; print('imports ok')"`
Expected: `imports ok` (demo_app itself only imports under Streamlit).

- [ ] **Step 4: Commit**

```bash
git add frontend/demo_app.py .claude/launch.json
git add -f README.md
git commit -m "feat: Streamlit demo page — side-by-side memory delta + structured retrieval table

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Browser verification (the user-requested protocol)

**Files:** none new — fixes land where the review finds them.

- [ ] **Step 1: Launch in the browser pane**

Use `preview_start` with name `demo-frontend`. Wait for Streamlit's "You can now view" log line (via `preview_logs`), then screenshot the initial page.

- [ ] **Step 2: Review each demo**

For each of the four demos, in order (failure-recall, goal-hierarchy,
identity-preferences, passing-mention):
1. Select it in the sidebar; screenshot the idle state (blurb + transcript expanders).
2. Click **Run demo**; screenshot during seeding (st.status visible) when timing allows.
3. When results render, screenshot the full page (side-by-side + context + table) — scroll and take a second screenshot if the table is below the fold.
4. Check against the spec's protocol: (a) the with-memory answer uses the seeded facts (DuplicateColumn / path line + failure / Dana + Rust + async / the passing-mention failure); (b) the table shows that demo's expected id kinds — goal ids with lineage for goal-hierarchy, a `Path` chain for passing-mention, entity rows for identity; (c) the demo's radio badge flipped to ●.
5. `Reset this demo`; verify the badge returns to ○.

- [ ] **Step 3: Fix what the review finds**

Each finding: fix in the source file, `preview_logs` for errors, re-run that demo, re-screenshot. Commit fixes as they land (`fix: <what the screenshot showed>`).

- [ ] **Step 4: Final verification**

Run the unit + integration suite once more and confirm no regression to the main suite:

```bash
.venv/bin/python -m pytest tests/test_demo_frontend.py -q
.venv/bin/python -m pytest -m "not agentic" -q
```

Expected: all PASS. Re-screenshot each demo's finished state once (four final screenshots).

- [ ] **Step 5: Commit**

```bash
git add -A frontend tests/test_demo_frontend.py
git commit -m "fix: demo frontend polish from browser review

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

(Skip the commit if the review found nothing.)

---

## Self-review against the spec

- §1 architecture (in-process, three files, dependency checks) → Tasks 1-3.
- §2 demo definitions, all four stories with anchors → Task 1 (registry test pins the anchors).
- §3 runtime contract (bootstrap/status/is_seeded/seed/retrieve/ask/reset, cached seeding, reset-first idempotency, think-stripping, env config) → Task 2. `@st.cache_resource` deviation recorded in Global Constraints.
- §4 planted tree, guarded, honest copy → Task 1 (blurb), Task 2 (`_plant_hierarchy`).
- §5 UI (sidebar status/picker/badges/resets; transcript expanders; Run; columns; context block; dataframe + caption; error handling; missing-model degradation) → Task 3.
- §6 config (`DEMO_AGENT_MODEL`, `DEMO_AGENT_TIMEOUT`, requirements, launch.json) → Tasks 2-3.
- §7 testing (registry, source rows incl. lineage, strip_think, integration round-trip; browser protocol) → Tasks 1, 2, 4.
- Alias-conflict reseed hazard (identity demo) handled in `_reset_user` — spec §3 cleanup, made concrete.
- Names consistent across tasks: `bootstrap/stack_status/is_seeded/seed/retrieve/ask_agent/agent_model_available/reset/reset_all/strip_think/build_source_rows/count_kinds`, `Handle`, `AgentAnswer`, `Demo.user_id/session_id`.
