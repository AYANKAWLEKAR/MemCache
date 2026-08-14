"""L3 Task nodes: creation, candidate listing, episode linkage, completion.

Live Neo4j. Teardown is scoped to the ids each test created — never global.
"""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.task_store import TaskStore

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def user_id(driver):
    uid = f"tsk-{uuid.uuid4().hex[:10]}"
    yield uid
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            DETACH DELETE p, t
            """,
            uid=uid,
        )


@pytest.fixture
def store(driver):
    return TaskStore(driver)


def _seed_profile(driver, uid):
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $uid})", uid=uid)


def test_create_and_get_roundtrip(store, driver, user_id):
    _seed_profile(driver, user_id)
    task_id = store.create_task(user_id, "Migrate telemetry to ClickHouse")

    row = store.get_task(task_id)
    assert row is not None
    assert row.id == task_id
    assert row.title == "Migrate telemetry to ClickHouse"
    assert row.status == "open"


def test_create_links_pursues_from_profile(store, driver, user_id):
    _seed_profile(driver, user_id)
    task_id = store.create_task(user_id, "Ship the release")

    with driver.session() as s:
        record = s.run(
            "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task {id: $tid}) "
            "RETURN count(t) AS c",
            uid=user_id,
            tid=task_id,
        ).single()
    assert record is not None and record["c"] == 1


def test_get_task_returns_none_when_absent(store):
    assert store.get_task("no-such-task") is None


def test_list_open_tasks_orders_by_recent_activity_and_caps(store, driver, user_id):
    _seed_profile(driver, user_id)
    first = store.create_task(user_id, "First goal")
    second = store.create_task(user_id, "Second goal")

    # Touching the older task must move it to the front of the candidate list.
    store.link_episode(first, episode_id=-930001)

    rows = store.list_open_tasks(user_id, limit=20)
    assert [r.id for r in rows][:2] == [first, second]

    capped = store.list_open_tasks(user_id, limit=1)
    assert [r.id for r in capped] == [first]


def test_list_open_tasks_excludes_done(store, driver, user_id):
    _seed_profile(driver, user_id)
    task_id = store.create_task(user_id, "Finish the audit")
    store.close_task(task_id)

    assert store.list_open_tasks(user_id, limit=20) == []
    row = store.get_task(task_id)
    assert row is not None and row.status == "done"


def test_link_episode_creates_advances_edge(store, driver, user_id):
    _seed_profile(driver, user_id)
    task_id = store.create_task(user_id, "Fix the auth bug")
    episode_id = -930002
    try:
        with driver.session() as s:
            s.run("MERGE (:Episode {id: $eid})", eid=episode_id)
        store.link_episode(task_id, episode_id=episode_id)
        store.link_episode(task_id, episode_id=episode_id)  # idempotent

        with driver.session() as s:
            record = s.run(
                "MATCH (:Episode {id: $eid})-[r:ADVANCES]->(:Task {id: $tid}) "
                "RETURN count(r) AS c",
                eid=episode_id,
                tid=task_id,
            ).single()
        assert record is not None and record["c"] == 1
    finally:
        with driver.session() as s:
            s.run("MATCH (e:Episode {id: $eid}) DETACH DELETE e", eid=episode_id)


def test_tasks_are_scoped_to_their_user(store, driver, user_id):
    _seed_profile(driver, user_id)
    other = f"{user_id}-other"
    _seed_profile(driver, other)
    try:
        store.create_task(user_id, "Mine")
        store.create_task(other, "Theirs")

        titles = {r.title for r in store.list_open_tasks(user_id, limit=20)}
        assert titles == {"Mine"}
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (p:UserProfile {user_id: $uid})
                OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
                DETACH DELETE p, t
                """,
                uid=other,
            )
