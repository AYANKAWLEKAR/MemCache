"""L4 wiring: API endpoints, worker claim into episodes + graph, Known Failures.

Live stack. The Ollama-dependent pieces (summarization, task adjudication) are
mocked so these gates test plumbing, not the model.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema
from app.main import app
from app.services.workbench_store import ensure_l4_schema, record_tool_call

pytestmark = pytest.mark.integration

AUTH = {"X-API-Key": "dummy-api-key-123"}


@pytest.fixture(scope="module", autouse=True)
def _eager():
    from app.workers.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def engine():
    eng = create_engine_from_settings()
    ensure_l2_schema(eng)
    ensure_l4_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def scoped(engine, driver):
    """user_id + session_id with full L2/L4/graph cleanup."""
    uid = f"wb-{uuid.uuid4().hex[:10]}"
    sid = f"{uid}-s1"
    yield uid, sid
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM tool_calls WHERE session_id = %s", (sid,))
        conn.exec_driver_sql("DELETE FROM episodes WHERE session_id = %s", (sid,))
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            DETACH DELETE p, a, e, t
            """,
            uid=uid,
        )
        s.run(
            """
            MATCH (se:Session {id: $sid})
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall)
            DETACH DELETE se, ep, dp, tc
            """,
            sid=sid,
        )


def _mock_summary(monkeypatch, text="Dana worked on the pipeline."):
    monkeypatch.setattr(
        "app.workers.tasks.summarize_conversation_ollama",
        lambda messages, settings=None: text,
    )


def _mock_adjudicator(monkeypatch, verdict=None):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: verdict,
    )


# ------------------------------------------------------------------- API


def test_post_tool_call_records_and_returns_receipt(client, engine, scoped):
    uid, sid = scoped
    response = client.post(
        "/workbench/tool-call",
        headers=AUTH,
        json={
            "session_id": sid,
            "tool_name": "psql",
            "status": "error",
            "args": {"query": "ALTER TABLE x"},
            "error": "permission denied for table x",
            "user_id": uid,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] > 0
    assert len(body["call_hash"]) == 64
    assert body["truncated"] is False


def test_post_tool_call_rejects_bad_status(client, scoped):
    _uid, sid = scoped
    response = client.post(
        "/workbench/tool-call",
        headers=AUTH,
        json={"session_id": sid, "tool_name": "psql", "status": "exploded"},
    )
    assert response.status_code == 422


def test_workbench_requires_api_key(client, scoped):
    _uid, sid = scoped
    assert (
        client.post(
            "/workbench/tool-call",
            json={"session_id": sid, "tool_name": "x", "status": "ok"},
        ).status_code
        == 401
    )
    assert client.get("/workbench/recent").status_code == 401


def test_recent_filters_by_hash_for_dedup(client, scoped):
    """Continuity: 'have I already tried this exact call?'"""
    uid, sid = scoped
    first = client.post(
        "/workbench/tool-call",
        headers=AUTH,
        json={
            "session_id": sid,
            "tool_name": "migrate",
            "status": "error",
            "args": {"b": 2, "a": 1},
            "error": "boom",
            "user_id": uid,
        },
    ).json()

    # Same call, different key order — must dedup to the same hash.
    response = client.get(
        "/workbench/recent",
        headers=AUTH,
        params={"call_hash": first["call_hash"], "user_id": uid},
    )
    assert response.status_code == 200
    calls = response.json()["calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == first["id"]
    assert calls[0]["status"] == "error"
    assert calls[0]["error"] == "boom"


# ------------------------------------------------------- worker claim + graph


def test_ingest_claims_tool_calls_and_writes_graph_nodes(
    client, engine, driver, scoped, monkeypatch
):
    """The tri-tier contract for L4: Postgres row, ToolCall node, ids agreeing."""
    uid, sid = scoped
    _mock_summary(monkeypatch)
    _mock_adjudicator(monkeypatch, None)

    recorded = record_tool_call(
        engine,
        session_id=sid,
        tool_name="pytest",
        status="error",
        error="1 failed",
        user_id=uid,
    )
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": sid,
            "user_id": uid,
            "messages": [{"role": "user", "content": "Tests are failing."}],
        },
    )
    assert response.status_code == 202

    with engine.begin() as conn:
        row = conn.exec_driver_sql(
            "SELECT episode_id FROM tool_calls WHERE id = %s", (recorded.id,)
        ).fetchone()
    assert row is not None and row[0] is not None, "call was not claimed by the episode"
    episode_id = row[0]

    with driver.session() as s:
        record = s.run(
            """
            MATCH (:Episode {id: $eid})-[:INVOKED]->(tc:ToolCall {id: $tcid})
            RETURN tc.tool_name AS tool_name, tc.status AS status
            """,
            eid=episode_id,
            tcid=recorded.id,
        ).single()
    assert record is not None, "no INVOKED edge in the graph"
    assert record["tool_name"] == "pytest"
    assert record["status"] == "error"


def test_claimed_calls_inherit_the_resolved_task(client, engine, driver, scoped, monkeypatch):
    """Failure memory scoped to the task, not just the session."""
    from app.services.task_inference import TaskAdjudication

    uid, sid = scoped
    _mock_summary(monkeypatch)
    _mock_adjudicator(
        monkeypatch,
        TaskAdjudication(goal="Fix the CI pipeline", matches_task_id=None, task_complete=False),
    )

    recorded = record_tool_call(
        engine, session_id=sid, tool_name="ci-run", status="error", error="timeout", user_id=uid
    )
    client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": sid,
            "user_id": uid,
            "messages": [{"role": "user", "content": "CI keeps timing out."}],
        },
    )

    with engine.begin() as conn:
        row = conn.exec_driver_sql(
            "SELECT task_id FROM tool_calls WHERE id = %s", (recorded.id,)
        ).fetchone()
    assert row is not None and row[0], "claimed call did not inherit the task"

    with driver.session() as s:
        record = s.run(
            "MATCH (t:Task {id: $tid}) RETURN t.title AS title", tid=row[0]
        ).single()
    assert record is not None and record["title"] == "Fix the CI pipeline"


# ------------------------------------------------------------ Known Failures


def test_known_failures_injected_into_retrieval(client, engine, scoped):
    uid, sid = scoped
    record_tool_call(
        engine,
        session_id=sid,
        tool_name="alembic",
        status="error",
        error="DuplicateColumn: column user_id already exists\n  at migration 0042",
        user_id=uid,
    )

    response = client.post(
        "/memory/retrieve",
        headers=AUTH,
        json={"session_id": sid, "user_id": uid, "query": "What should I do next?"},
    )
    assert response.status_code == 200
    body = response.json()

    assert "known failures" in body["context"].lower()
    assert "alembic" in body["context"].lower()
    assert "duplicatecolumn" in body["context"].lower()
    # Only the first line of the error belongs in context.
    assert "migration 0042" not in body["context"]

    failures = [s for s in body["sources"] if s["type"] == "tool_failure"]
    assert failures and failures[0]["tier"] == "L4"
    assert failures[0]["details"]["tool_name"] == "alembic"


def test_no_failures_section_without_user_id(client, engine, scoped):
    uid, sid = scoped
    record_tool_call(
        engine, session_id=sid, tool_name="alembic", status="error", error="x", user_id=uid
    )
    response = client.post(
        "/memory/retrieve",
        headers=AUTH,
        json={"session_id": sid, "query": "What should I do next?"},
    )
    assert "known failures" not in response.json()["context"].lower()


def test_successful_calls_do_not_appear_as_failures(client, engine, scoped):
    uid, sid = scoped
    record_tool_call(
        engine, session_id=sid, tool_name="ls", status="ok", output="fine", user_id=uid
    )
    response = client.post(
        "/memory/retrieve",
        headers=AUTH,
        json={"session_id": sid, "user_id": uid, "query": "Anything to know?"},
    )
    assert "known failures" not in response.json()["context"].lower()
