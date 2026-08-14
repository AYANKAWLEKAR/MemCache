"""Worker task-inference wiring: adjudicator mocked, everything else real.

Mocking the adjudicator here is deliberate and follows the house rule: LLM
judgement is a metric, never a gate. These tests gate the *plumbing* — that a
verdict, whatever produces it, lands in the graph correctly and that no verdict
failure can fail an ingest.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.main import app
from app.services.task_inference import TaskAdjudication
from app.services.task_store import TaskStore

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
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def scoped_user(driver):
    """User id plus full cleanup of profile, tasks, episodes, and L2 rows."""
    from app.db.postgres import create_engine_from_settings

    uid = f"tw-{uuid.uuid4().hex[:10]}"
    yield uid
    engine = create_engine_from_settings()
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    engine.dispose()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(al:Entity)
            DETACH DELETE p, t, a, al
            """,
            uid=uid,
        )
        s.run(
            """
            MATCH (se:Session) WHERE se.id STARTS WITH $uid
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            DETACH DELETE se, ep, dp
            """,
            uid=uid,
        )


def _ingest(client, session_id, user_id, content="We kept working on the migration."):
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "Understood."},
            ],
        },
    )
    assert response.status_code == 202, response.text


def test_new_goal_creates_task_and_advances_edge(client, driver, scoped_user, monkeypatch):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal="Migrate telemetry to ClickHouse", matches_task_id=None, task_complete=False
        ),
    )
    _ingest(client, f"{scoped_user}-s1", scoped_user)

    store = TaskStore(driver)
    tasks = store.list_open_tasks(scoped_user, limit=20)
    assert [t.title for t in tasks] == ["Migrate telemetry to ClickHouse"]

    with driver.session() as s:
        record = s.run(
            """
            MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task)<-[:ADVANCES]-(e:Episode)
            RETURN count(e) AS c
            """,
            uid=scoped_user,
        ).single()
    assert record is not None and record["c"] == 1


def test_matched_goal_links_existing_task_instead_of_creating(client, driver, scoped_user, monkeypatch):
    store = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $uid})", uid=scoped_user)
    existing = store.create_task(scoped_user, "Migrate telemetry to ClickHouse")

    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal="finish the ClickHouse migration",
            matches_task_id=existing,
            task_complete=False,
        ),
    )
    _ingest(client, f"{scoped_user}-s2", scoped_user)

    tasks = store.list_open_tasks(scoped_user, limit=20)
    assert [t.id for t in tasks] == [existing], "a duplicate task was created"


def test_task_complete_closes_the_task(client, driver, scoped_user, monkeypatch):
    store = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $uid})", uid=scoped_user)
    existing = store.create_task(scoped_user, "Ship the release")

    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal="ship the release", matches_task_id=existing, task_complete=True
        ),
    )
    _ingest(client, f"{scoped_user}-s3", scoped_user, "The release shipped, we're done.")

    row = store.get_task(existing)
    assert row is not None and row.status == "done"
    assert store.list_open_tasks(scoped_user, limit=20) == []


def test_null_goal_creates_nothing(client, driver, scoped_user, monkeypatch):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal=None, matches_task_id=None, task_complete=False
        ),
    )
    _ingest(client, f"{scoped_user}-s4", scoped_user, "Nice weather today.")

    assert TaskStore(driver).list_open_tasks(scoped_user, limit=20) == []


def test_adjudicator_returning_none_does_not_fail_ingest(client, driver, scoped_user, monkeypatch):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: None,
    )
    _ingest(client, f"{scoped_user}-s5", scoped_user)
    assert TaskStore(driver).list_open_tasks(scoped_user, limit=20) == []


def test_adjudicator_raising_does_not_fail_ingest(client, driver, scoped_user, monkeypatch):
    """The contract: no task-tier failure may fail an ingest."""

    def _boom(summary, open_tasks, settings=None):
        raise RuntimeError("adjudication exploded")

    monkeypatch.setattr("app.workers.tasks.adjudicate_task", _boom)
    _ingest(client, f"{scoped_user}-s6", scoped_user)  # asserts 202 internally


def test_no_user_id_means_no_task_inference(client, driver, monkeypatch):
    calls = []

    def _spy(summary, open_tasks, settings=None):
        calls.append(summary)
        return None

    monkeypatch.setattr("app.workers.tasks.adjudicate_task", _spy)
    sid = f"tw-nouser-{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": sid,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 202
    assert calls == [], "task inference ran without a user_id"
