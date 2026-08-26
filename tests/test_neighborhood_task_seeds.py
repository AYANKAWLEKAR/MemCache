"""fetch_neighborhood can start from Task ids as well as Entity names."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.neo4j_store import Neo4jStore
from app.services.task_store import TaskStore

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


def test_neighborhood_from_task_ids_reaches_parent_episode_and_tool_call(driver):
    uid = f"nb-{uuid.uuid4().hex[:8]}"
    ts = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $u})", u=uid)
    root = ts.create_task(uid, "root")
    leaf = ts.create_task(uid, "leaf")
    ts.set_parent(leaf, root)
    try:
        with driver.session() as s:
            s.run(
                "MATCH (t:Task {id: $root}) MERGE (e:Episode {id: -940001}) MERGE (e)-[:ADVANCES]->(t) "
                "MERGE (tc:ToolCall {id: -940002}) SET tc.tool_name='alembic', tc.status='error' "
                "MERGE (e)-[:INVOKED]->(tc)",
                root=root,
            )
        nb = Neo4jStore(driver).fetch_neighborhood([], task_ids=[leaf, root], radius=4)
        ids = {e.src for e in nb.edges} | {e.dst for e in nb.edges}
        assert f"Task:{leaf}" in ids and f"Task:{root}" in ids
        assert "Episode:-940001" in ids and "ToolCall:-940002" in ids
        assert nb.labels[f"Task:{root}"] == "Task"
        rels = {e.rel for e in nb.edges}
        assert {"SUBGOAL_OF", "ADVANCES", "INVOKED"} <= rels
        # Still empty when given nothing at all.
        assert Neo4jStore(driver).fetch_neighborhood([], task_ids=[]).edges == []
    finally:
        with driver.session() as s:
            s.run("MATCH (n) WHERE n:Episode AND n.id = -940001 DETACH DELETE n")
            s.run("MATCH (n) WHERE n:ToolCall AND n.id = -940002 DETACH DELETE n")
            s.run("MATCH (p:UserProfile {user_id: $u}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) DETACH DELETE p, t", u=uid)
