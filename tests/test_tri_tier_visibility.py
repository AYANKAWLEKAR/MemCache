"""One ingest must be independently verifiable in Redis, Postgres, and Neo4j.

Every assertion queries the store directly rather than reading back through the
API that wrote it — otherwise a bug in the write path could hide behind the same
bug in the read path.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema
from app.main import app

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
def stores():
    import redis

    engine = create_engine_from_settings()
    ensure_l2_schema(engine)
    driver = create_driver_from_settings()
    ensure_constraints(driver)
    r = redis.from_url(settings.redis_url, decode_responses=True)
    yield {"pg": engine, "neo": driver, "redis": r}
    r.close()
    driver.close()
    engine.dispose()


@pytest.fixture
def ingested(client, stores):
    """One real ingest carrying a user_id, cleaned up afterwards."""
    user_id = f"tri-{uuid.uuid4().hex[:10]}"
    session_id = f"{user_id}-s1"
    messages = [
        {"role": "user", "content": "I'm Dana Whitfield at Northwind Robotics."},
        {"role": "assistant", "content": "Good to meet you, Dana."},
    ]
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={"session_id": session_id, "user_id": user_id, "messages": messages},
    )
    assert response.status_code == 202, response.text

    yield {"user_id": user_id, "session_id": session_id, "messages": messages}

    stores["redis"].delete(f"session:{session_id}")
    with stores["pg"].begin() as conn:
        conn.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (user_id,))
    with stores["neo"].session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=user_id,
        )
        s.run(
            """
            MATCH (se:Session {id: $sid})
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            DETACH DELETE se, ep, dp
            """,
            sid=session_id,
        )


def test_l1_redis_holds_the_raw_turns_with_ttl(stores, ingested):
    key = f"session:{ingested['session_id']}"
    raw = stores["redis"].lrange(key, 0, -1)
    assert len(raw) == len(ingested["messages"])

    contents = {json.loads(item)["content"] for item in raw}
    assert "I'm Dana Whitfield at Northwind Robotics." in contents

    ttl = stores["redis"].ttl(key)
    assert 0 < ttl <= settings.redis_session_ttl_seconds


def test_l2_postgres_holds_an_embedded_episode_owned_by_the_user(stores, ingested):
    with stores["pg"].begin() as conn:
        rows = conn.exec_driver_sql(
            "SELECT id, user_id, session_id, (embedding IS NOT NULL) FROM episodes "
            "WHERE session_id = %s",
            (ingested["session_id"],),
        ).fetchall()

    assert rows, "no episode reached L2"
    for episode_id, user_id, session_id, has_embedding in rows:
        assert user_id == ingested["user_id"], "episode not attributed to the user"
        assert session_id == ingested["session_id"]
        assert has_embedding, "episode stored without an embedding"


def test_l3_neo4j_holds_the_entity_episode_path(stores, ingested):
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(ep:Episode)-[:MENTIONS]->(e:Entity)
    RETURN ep.id AS episode_id, collect(DISTINCT e.name) AS entities
    """
    with stores["neo"].session() as s:
        records = list(s.run(q, sid=ingested["session_id"]))

    assert records, "no Session->Episode->Entity path in L3"
    entities = {name for r in records for name in r["entities"]}
    assert "northwind robotics" in entities, f"expected entity missing: {entities}"


def test_l2_and_l3_agree_on_the_episode_id(stores, ingested):
    """The tiers must describe the same episode, not two unrelated records."""
    with stores["pg"].begin() as conn:
        pg_ids = {
            r[0]
            for r in conn.exec_driver_sql(
                "SELECT id FROM episodes WHERE session_id = %s",
                (ingested["session_id"],),
            ).fetchall()
        }

    with stores["neo"].session() as s:
        neo_ids = {
            r["id"]
            for r in s.run(
                "MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(e:Episode) RETURN e.id AS id",
                sid=ingested["session_id"],
            )
        }

    assert pg_ids, "no L2 episode to compare"
    assert pg_ids == neo_ids, f"L2 {pg_ids} and L3 {neo_ids} disagree"


def test_profile_links_the_session_in_the_graph(stores, ingested):
    """L3 also records that this user participated in the session."""
    with stores["neo"].session() as s:
        record = s.run(
            "MATCH (:UserProfile {user_id: $uid})-[:PARTICIPATED_IN]->(se:Session) "
            "RETURN collect(se.id) AS sessions",
            uid=ingested["user_id"],
        ).single()

    assert record is not None
    assert ingested["session_id"] in record["sessions"]
