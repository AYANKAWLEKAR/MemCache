"""Proactive retrieval end-to-end against the live graph.

The contract: something *mentioned in passing* in the conversation lights up
its neighborhood — episodes and tool failures reachable through weighted edges
surface without the query naming them. Assertions read the graph and the
returned provenance directly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.neo4j_store import Neo4jStore
from app.services.postgres_store import PostgresStore
from app.services.retrieval import retrieve_context
from app.services.workbench_store import claim_tool_calls, ensure_l4_schema, record_tool_call
from tests.conftest import unit_embedding_384

pytestmark = pytest.mark.integration


@pytest.fixture
def stack():
    engine = create_engine_from_settings()
    ensure_l2_schema(engine)
    ensure_l4_schema(engine)
    driver = create_driver_from_settings()
    ensure_constraints(driver)
    yield engine, driver
    driver.close()
    engine.dispose()


@pytest.fixture
def world(stack, monkeypatch):
    """A small graph where ClickHouse links, via a counted RELATED_TO edge, to an
    episode that INVOKED a failing alembic call. Nothing in the *query* will
    name any of it; only recent L1 messages mention ClickHouse."""
    engine, driver = stack
    uid = f"pr-{uuid.uuid4().hex[:8]}"
    old_session = f"{uid}-old"
    live_session = f"{uid}-live"
    now = datetime.now(UTC)

    with session_scope(engine) as s:
        episode_id = PostgresStore(s).insert_episode(
            session_id=old_session,
            summary="Ran the alembic migration for the ClickHouse telemetry schema; it failed.",
            embedding=unit_embedding_384(primary_axis=300),  # far from any query axis
            start_time=now,
            end_time=now,
            metadata=None,
            user_id=uid,
        )
    call = record_tool_call(
        engine,
        session_id=old_session,
        user_id=uid,
        tool_name="alembic",
        status="error",
        error="DuplicateColumn: user_id already exists",
    )
    claim_tool_calls(engine, session_id=old_session, episode_id=episode_id)

    graph = Neo4jStore(driver)
    graph.upsert_session(old_session)
    graph.upsert_episode(old_session, episode_id, "alembic migration failed")
    graph.merge_entities(["ClickHouse", "alembic"], episode_id=episode_id)
    graph.create_relationships([("ClickHouse", "alembic")] * 6)  # a well-worn link
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $u})", u=uid)
        s.run(
            "MATCH (e:Episode {id: $eid}) MERGE (t:ToolCall {id: $tid}) "
            "SET t.tool_name = 'alembic', t.status = 'error' MERGE (e)-[:INVOKED]->(t)",
            eid=episode_id, tid=call.id,
        )
        s.run(
            "MATCH (p:UserProfile {user_id: $u}) MERGE (se:Session {id: $sid}) "
            "MERGE (p)-[:PARTICIPATED_IN]->(se)", u=uid, sid=live_session,
        )

    # Live L1: the user mentions ClickHouse in passing. Redis is real.
    from app.api import services as api_services

    api_services.get_redis_store().append_messages(
        live_session,
        [{"role": "user", "content": "Also, ClickHouse ingest looked slow yesterday."}],
    )
    # The query embedding must not accidentally match the episode via L2, so
    # the only route to it is the graph. Point the embedder at an orthogonal axis.
    monkeypatch.setattr(
        "app.api.services.get_query_embedder",
        lambda: type("E", (), {"encode": lambda self, t, normalize_embeddings=True: type(
            "V", (), {"tolist": lambda s: unit_embedding_384(primary_axis=7)}
        )()})(),
    )

    yield {"uid": uid, "live": live_session, "old": old_session,
           "episode_id": episode_id, "call_id": call.id}

    api_services.get_redis_client().delete(f"session:{live_session}")
    with engine.begin() as c:
        c.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (uid,))
        c.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    with driver.session() as s:
        s.run("MATCH (p:UserProfile {user_id:$u}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) DETACH DELETE p, t", u=uid)
        for sid in (old_session, live_session):
            s.run(
                "MATCH (se:Session {id:$s}) OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep) "
                "OPTIONAL MATCH (ep)-[:INVOKED]->(tc) DETACH DELETE se, ep, tc", s=sid,
            )
        s.run("MATCH (e:Entity) WHERE e.name IN ['clickhouse','alembic'] DETACH DELETE e")


def test_passing_mention_surfaces_linked_failure_and_episode(world):
    """Query names nothing; ClickHouse in L1 lights the alembic failure two hops away."""
    r = retrieve_context(world["live"], "anything else I should keep in mind?", 1500,
                         user_id=world["uid"])
    types = {s["type"] for s in r["sources"]}
    assert "proactive_tool_failure" in types or "proactive_episode" in types, (
        f"nothing surfaced proactively; sources={sorted(types)}\n{r['context']}"
    )
    lower = r["context"].lower()
    assert "alembic" in lower and "duplicatecolumn" in lower, r["context"]


def test_proactive_sources_carry_activation_and_a_path(world):
    r = retrieve_context(world["live"], "ok", 1500, user_id=world["uid"])
    pro = [s for s in r["sources"] if s["type"].startswith("proactive_")]
    assert pro, "no proactive sources"
    for s in pro:
        d = s["details"]
        assert 0.0 < d["activation"] <= 1.0
        assert isinstance(d["path"], list)
        # A non-seed node's path must start at a seed and end at the node.
        if not d.get("is_seed"):
            assert d["path"], f"activated node with empty path: {s}"


def test_no_user_id_means_no_proactive_section(world):
    r = retrieve_context(world["live"], "ok", 1500)
    assert not any(s["type"].startswith("proactive_") for s in r["sources"])
    assert "Proactive Context" not in r["context"]


def test_severed_spreading_removes_the_surface(world, monkeypatch):
    """Regression guard: with activation disabled the same call surfaces nothing.
    Proves the section is *earned* by spreading, not by some other route."""
    monkeypatch.setattr("app.services.retrieval.spread_activation", lambda *a, **k: {})
    r = retrieve_context(world["live"], "ok", 1500, user_id=world["uid"])
    assert not any(s["type"].startswith("proactive_") for s in r["sources"])
