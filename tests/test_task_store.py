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


# ---------------------------------------------------------------- hierarchy


def test_set_parent_creates_subgoal_edge_and_is_idempotent(store, driver, user_id):
    _seed_profile(driver, user_id)
    parent = store.create_task(user_id, "Ship telemetry v2")
    child = store.create_task(user_id, "Migrate telemetry schema")

    store.set_parent(child, parent)
    store.set_parent(child, parent)  # same edge again: no-op

    with driver.session() as s:
        n = s.run(
            "MATCH (:Task {id: $c})-[r:SUBGOAL_OF]->(:Task {id: $p}) RETURN count(r) AS n",
            c=child, p=parent,
        ).single()["n"]
    assert n == 1
    got = store.get_parent(child)
    assert got is not None and got.id == parent
    assert store.get_parent(parent) is None


def test_set_parent_refuses_a_second_different_parent(store, driver, user_id):
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    a = store.create_task(user_id, "A")
    b = store.create_task(user_id, "B")
    child = store.create_task(user_id, "child")
    store.set_parent(child, a)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(child, b)
    assert store.get_parent(child).id == a  # unchanged


def test_set_parent_refuses_self(store, driver, user_id):
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    t = store.create_task(user_id, "loop")
    with pytest.raises(TaskHierarchyError):
        store.set_parent(t, t)


def test_set_parent_refuses_a_parent_inside_the_childs_subtree(store, driver, user_id):
    """root <- mid <- leaf; then root under leaf would be a cycle."""
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    mid = store.create_task(user_id, "mid")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(mid, root)
    store.set_parent(leaf, mid)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(root, leaf)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(root, mid)
    assert store.get_parent(root) is None


def test_set_parent_refuses_cross_user(store, driver, user_id):
    """Cross-user parentage would leak one person's failures into another's
    context — this must fail loudly, never degrade."""
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    other = f"{user_id}-other"
    _seed_profile(driver, other)
    try:
        mine = store.create_task(user_id, "mine")
        theirs = store.create_task(other, "theirs")
        with pytest.raises(TaskHierarchyError):
            store.set_parent(mine, theirs)
        with pytest.raises(TaskHierarchyError):
            store.set_parent(theirs, mine)
    finally:
        with driver.session() as s:
            s.run(
                "MATCH (p:UserProfile {user_id: $uid}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) "
                "DETACH DELETE p, t",
                uid=other,
            )


def test_ancestors_children_descendants_and_lineage(store, driver, user_id):
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    mid = store.create_task(user_id, "mid")
    leaf = store.create_task(user_id, "leaf")
    sib = store.create_task(user_id, "sibling")
    store.set_parent(mid, root)
    store.set_parent(leaf, mid)
    store.set_parent(sib, root)

    assert [t.id for t in store.get_ancestors(leaf)] == [mid, root]  # nearest first
    assert store.get_ancestors(root) == []
    assert {t.id for t in store.get_children(root)} == {mid, sib}
    assert store.get_children(leaf) == []
    assert store.get_descendant_ids(root) == {mid, leaf, sib}
    assert store.get_descendant_ids(leaf) == set()
    assert store.get_lineage_ids(leaf) == [leaf, mid, root]
    assert store.get_lineage_ids(root) == [root]


def test_ancestor_walk_ignores_status(store, driver, user_id):
    """A closed parent with an open child is legitimate; the walk must not
    truncate at it."""
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(leaf, root)
    store.close_task(root)
    anc = store.get_ancestors(leaf)
    assert [t.id for t in anc] == [root]
    assert anc[0].status == "done"


def test_ancestor_walk_is_bounded_by_max_depth(store, driver, user_id, monkeypatch):
    from app.config import settings

    _seed_profile(driver, user_id)
    monkeypatch.setattr(settings, "task_max_depth", 3)
    ids = [store.create_task(user_id, f"t{i}") for i in range(6)]
    for child, parent in zip(ids[1:], ids[:-1]):
        store.set_parent(child, parent)  # t5 -> t4 -> ... -> t0
    anc = store.get_ancestors(ids[5])
    assert len(anc) == 3, "depth cap must bound the walk"
    assert [t.id for t in anc] == [ids[4], ids[3], ids[2]]
