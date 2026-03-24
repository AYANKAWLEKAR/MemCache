"""L3 Neo4j: constraints, upserts, MENTIONS, RELATED_TO, decisions/preferences, queries.

Requires docker-compose Neo4j. Deselect with: pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.neo4j_store import Neo4jStore, normalize_entity_name
from tests.conftest import (
    SYNTHETIC_L3_DECISIONS,
    SYNTHETIC_L3_ENTITIES_ACME,
    SYNTHETIC_L3_PREFERENCES,
    SYNTHETIC_L3_RELATED_PAIRS,
)


def test_normalize_entity_name_unit():
    """No database: normalization matches PRD-style MERGE keys."""
    assert normalize_entity_name("  Acme Corp  ") == "acme corp"
    assert normalize_entity_name("John\n\tDoe") == "john doe"
    assert normalize_entity_name("") == ""


@pytest.fixture
def neo4j_driver():
    driver = create_driver_from_settings()
    ensure_constraints(driver)
    yield driver
    driver.close()


@pytest.fixture
def l3_store(neo4j_driver):
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    return Neo4jStore(neo4j_driver)


@pytest.mark.integration
def test_constraints_and_empty_graph(l3_store, neo4j_driver):
    """Driver session works; graph can be cleared."""
    with neo4j_driver.session() as session:
        n = session.run("MATCH (n) RETURN count(n) AS c").single()
        assert n is not None
        assert n["c"] == 0


@pytest.mark.integration
def test_upsert_session_and_episode_roundtrip(l3_store):
    sid = "sess_l3_roundtrip"
    l3_store.upsert_session(sid)
    l3_store.upsert_episode(sid, episode_id=42, summary="Synthetic episode summary.")
    ents = l3_store.query_session_entities(sid)
    assert ents == []


@pytest.mark.integration
def test_merge_entities_creates_mentions(l3_store):
    sid = "sess_mentions"
    l3_store.upsert_session(sid)
    l3_store.upsert_episode(sid, episode_id=100, summary="Discussed John and Acme.")
    norms = l3_store.merge_entities(
        ["John", "Acme Corp"],
        episode_id=100,
    )
    assert "john" in norms
    assert "acme corp" in norms
    rows = l3_store.query_session_entities(sid)
    names = {r.name for r in rows}
    assert names == {"acme corp", "john"}


@pytest.mark.integration
def test_create_relationships_two_hop_traversal(l3_store):
    sid = "sess_rel"
    l3_store.upsert_session(sid)
    l3_store.upsert_episode(sid, episode_id=200, summary="Link entities.")
    l3_store.merge_entities(["alpha", "beta", "gamma"])
    l3_store.create_relationships([("alpha", "beta"), ("beta", "gamma")])
    related_from_alpha = set(l3_store.query_related_entities("alpha", max_hops=2))
    assert related_from_alpha == {"beta", "gamma"}


@pytest.mark.integration
def test_record_decisions_and_query_decisions_preferences(l3_store):
    sid = "sess_dp"
    l3_store.upsert_session(sid)
    l3_store.upsert_episode(sid, episode_id=300, summary="Decided and preferred.")
    l3_store.record_decisions_and_preferences(
        300,
        decisions=["Ship MVP in Q1"],
        preferences=["Typed Python"],
    )
    out = l3_store.query_decisions_preferences(sid)
    assert out["decisions"] == ["Ship MVP in Q1"]
    assert out["preferences"] == ["Typed Python"]


@pytest.mark.integration
def test_synthetic_acme_session_end_to_end(l3_store):
    """PRD-aligned synthetic path: session → episode → entities, rels, decisions."""
    sid = "sess_synthetic_acme"
    eid = 9001
    l3_store.upsert_session(sid)
    l3_store.upsert_episode(
        sid,
        episode_id=eid,
        summary="John at Acme chose Python and prefers dark mode.",
    )
    l3_store.merge_entities(SYNTHETIC_L3_ENTITIES_ACME, episode_id=eid)
    pairs = [(a, b) for a, b in SYNTHETIC_L3_RELATED_PAIRS]
    l3_store.create_relationships(pairs)
    l3_store.record_decisions_and_preferences(
        eid,
        decisions=list(SYNTHETIC_L3_DECISIONS),
        preferences=list(SYNTHETIC_L3_PREFERENCES),
    )

    ent_rows = l3_store.query_session_entities(sid)
    ent_names = {r.name for r in ent_rows}
    assert "john" in ent_names
    assert "acme corp" in ent_names
    assert "python" in ent_names

    related_john = set(l3_store.query_related_entities("John", max_hops=2))
    assert "acme corp" in related_john
    assert "python" in related_john

    dp = l3_store.query_decisions_preferences(sid)
    assert SYNTHETIC_L3_DECISIONS[0] in dp["decisions"]
    assert SYNTHETIC_L3_PREFERENCES[0] in dp["preferences"]


@pytest.mark.integration
def test_session_boundary_entities(l3_store):
    """Entities from another session are not returned for the queried session."""
    l3_store.upsert_session("sess_a")
    l3_store.upsert_session("sess_b")
    l3_store.upsert_episode("sess_a", 1, summary="A")
    l3_store.upsert_episode("sess_b", 2, summary="B")
    l3_store.merge_entities(["OnlyA"], episode_id=1)
    l3_store.merge_entities(["OnlyB"], episode_id=2)
    a_entities = {r.name for r in l3_store.query_session_entities("sess_a")}
    b_entities = {r.name for r in l3_store.query_session_entities("sess_b")}
    assert a_entities == {"onlya"}
    assert b_entities == {"onlyb"}
