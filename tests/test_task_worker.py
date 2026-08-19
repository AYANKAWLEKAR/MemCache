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


# ------------------------------------------------------------ hierarchy


def _mock_adjudicator(monkeypatch, goal, matches=None):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal=goal, matches_task_id=matches, task_complete=False
        ),
    )


def _mock_placement(monkeypatch, fn):
    """fn(subject_title, candidates) -> PlacementVerdict | None."""
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_placement",
        lambda subject_title, candidates, settings=None: fn(subject_title, candidates),
    )


def _tree(driver, uid):
    with driver.session() as s:
        return {
            r["t"]: r["p"]
            for r in s.run(
                "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task) "
                "OPTIONAL MATCH (t)-[:SUBGOAL_OF]->(p:Task) RETURN t.title AS t, p.title AS p",
                uid=uid,
            )
        }


def test_child_of_places_new_task_under_named_parent(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Ship telemetry v2")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "Goal: ship telemetry v2.")

    _mock_adjudicator(monkeypatch, "Migrate telemetry schema")
    _mock_placement(
        monkeypatch,
        lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]) if c else None,
    )
    _ingest(client, f"{scoped_user}-s2", scoped_user, "Now migrating the telemetry schema.")

    assert _tree(driver, scoped_user) == {
        "Ship telemetry v2": None,
        "Migrate telemetry schema": "Ship telemetry v2",
    }


def test_parent_of_adopts_the_named_root(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Fix duplicate column")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "Fixing the duplicate column.")

    _mock_adjudicator(monkeypatch, "Migrate telemetry schema")
    _mock_placement(
        monkeypatch,
        lambda t, c: PlacementVerdict(relation="parent_of", task_id=c[0][0]) if c else None,
    )
    _ingest(client, f"{scoped_user}-s2", scoped_user, "The bigger job is migrating the schema.")

    assert _tree(driver, scoped_user) == {
        "Fix duplicate column": "Migrate telemetry schema",
        "Migrate telemetry schema": None,
    }


def test_parent_of_a_non_root_writes_nothing(client, driver, scoped_user, monkeypatch):
    """Adopting an already-parented task would give it two parents."""
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "root")
    _mock_adjudicator(monkeypatch, "Child")
    _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
    _ingest(client, f"{scoped_user}-s2", scoped_user, "child")

    # Third task tries to adopt "Child" (which has parent Root).
    _mock_adjudicator(monkeypatch, "Interloper")

    def adopt_child(t, c):
        cid = next(i for i, title in c if title == "Child")
        return PlacementVerdict(relation="parent_of", task_id=cid)

    _mock_placement(monkeypatch, adopt_child)
    _ingest(client, f"{scoped_user}-s3", scoped_user, "interloper")

    assert _tree(driver, scoped_user) == {"Root": None, "Child": "Root", "Interloper": None}


def test_none_and_malformed_verdicts_write_nothing(client, driver, scoped_user, monkeypatch):
    _mock_adjudicator(monkeypatch, "A")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "a")
    _mock_adjudicator(monkeypatch, "B")
    _mock_placement(monkeypatch, lambda t, c: None)  # model said none / parse failed
    _ingest(client, f"{scoped_user}-s2", scoped_user, "b")
    assert _tree(driver, scoped_user) == {"A": None, "B": None}


def test_already_parented_task_is_never_replaced(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "root")
    _mock_adjudicator(monkeypatch, "Other root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s2", scoped_user, "other")
    _mock_adjudicator(monkeypatch, "Child")

    def under_root(t, c):
        rid = next(i for i, title in c if title == "Root")
        return PlacementVerdict(relation="child_of", task_id=rid)

    _mock_placement(monkeypatch, under_root)
    _ingest(client, f"{scoped_user}-s3", scoped_user, "child")

    # Continue "Child" (matched); placement is not even consulted.
    child_id = next(t.id for t in TaskStore(driver).list_open_tasks(scoped_user, limit=20) if t.title == "Child")
    _mock_adjudicator(monkeypatch, "Child", matches=child_id)
    called = {"n": 0}

    def spy(t, c):
        called["n"] += 1
        return PlacementVerdict(
            relation="child_of", task_id=next(i for i, title in c if title == "Other root")
        )

    _mock_placement(monkeypatch, spy)
    _ingest(client, f"{scoped_user}-s4", scoped_user, "more child work")

    assert called["n"] == 0, "placement must not run for an already-parented task"
    assert _tree(driver, scoped_user)["Child"] == "Root"


def test_hierarchy_error_does_not_fail_ingest(client, driver, scoped_user, monkeypatch):
    """set_parent raising must be swallowed at the worker boundary."""
    from app.services import task_store as ts
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "A")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "a")

    def boom(self, child, parent):
        raise ts.TaskHierarchyError("simulated")

    monkeypatch.setattr(ts.TaskStore, "set_parent", boom)
    _mock_adjudicator(monkeypatch, "B")
    _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
    _ingest(client, f"{scoped_user}-s2", scoped_user, "b")  # asserts 202 inside

    assert _tree(driver, scoped_user) == {"A": None, "B": None}


def test_tool_calls_are_stamped_with_the_leaf(client, driver, scoped_user, monkeypatch):
    from app.db.postgres import create_engine_from_settings
    from app.services.task_hierarchy import PlacementVerdict
    from app.services.workbench_store import recent_tool_calls, record_tool_call

    engine = create_engine_from_settings()
    try:
        _mock_adjudicator(monkeypatch, "Root")
        _mock_placement(monkeypatch, lambda t, c: None)
        _ingest(client, f"{scoped_user}-s1", scoped_user, "root")

        sid = f"{scoped_user}-s2"
        record_tool_call(
            engine, session_id=sid, user_id=scoped_user, tool_name="alembic", status="error", error="boom"
        )
        _mock_adjudicator(monkeypatch, "Leaf")
        _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
        _ingest(client, sid, scoped_user, "leaf work")

        assert _tree(driver, scoped_user) == {"Root": None, "Leaf": "Root"}
        leaf_id = next(t.id for t in TaskStore(driver).list_open_tasks(scoped_user, limit=20) if t.title == "Leaf")
        rows = recent_tool_calls(engine, session_id=sid, limit=10)
        assert rows and rows[0].task_id == leaf_id
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (scoped_user,))
        engine.dispose()
