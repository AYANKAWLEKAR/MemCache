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
    from frontend.demos import DEMOS

    for d in DEMOS:
        _reset_user(d.user_id)


def _reset_user(uid: str) -> None:
    """Scoped cleanup across all tiers — the demo script's pattern, by prefix.

    Every session id starts with the user id, so `STARTS WITH uid` and
    `session:uid*` bound the deletes to this demo's data. The alias release at
    the end mirrors the test fixture `release_person_names`: alias conflicts
    are permanent by design, so without it a reseed of the identity demo would
    409 against the previous run's profile.
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
