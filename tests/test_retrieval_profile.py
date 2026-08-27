"""Retrieval resolves identity through the canonical profile."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_store import ProfileStore
from app.services.retrieval import retrieve_context

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def seeded(driver):
    """A profile with identity and a decision, plus an unrelated empty session."""
    from app.services.neo4j_store import Neo4jStore

    uid = f"rp-{uuid.uuid4().hex[:10]}"
    session_a = f"{uid}-a"
    session_b = f"{uid}-b"
    episode_id = -920001

    graph = Neo4jStore(driver)
    store = ProfileStore(driver)
    graph.upsert_session(session_a)
    graph.upsert_episode(session_a, episode_id, "Chose Rust for motion control.")
    graph.record_decisions_and_preferences(
        episode_id, ["use Rust for motion control"], ["async standups"]
    )
    store.upsert_profile(uid, display_name="Dana Whitfield")
    store.set_attribute(uid, "title", "Staff Engineer", source="explicit", confidence=1.0)
    store.link_alias(uid, "Dana Whitfield", source="explicit", confidence=1.0)
    store.link_session(uid, session_a)
    store.promote_episode_facts(uid, episode_id)

    yield uid, session_b

    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )
        s.run(
            """
            MATCH (ep:Episode {id: $eid})
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            DETACH DELETE ep, dp
            """,
            eid=episode_id,
        )
        for sid in (session_a, session_b):
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=sid)


def test_profile_facts_reach_a_different_session(seeded):
    """The payoff: session B knows who the user is and what they decided in A."""
    uid, session_b = seeded

    result = retrieve_context(session_b, "What do you know about me?", 1500, user_id=uid)
    context = result["context"].lower()

    assert "dana whitfield" in context
    assert "staff engineer" in context
    assert "use rust for motion control" in context

    types = {s["type"] for s in result["sources"]}
    assert "profile_identity" in types
    assert {
        s["tier"] for s in result["sources"] if s["type"] == "profile_identity"
    } == {"L3"}


def test_retrieval_without_user_id_is_unchanged(seeded):
    """Omitting user_id must not surface profile facts."""
    _uid, session_b = seeded

    result = retrieve_context(session_b, "What do you know about me?", 1500)
    assert "staff engineer" not in result["context"].lower()
    assert all(s["type"] != "profile_identity" for s in result["sources"])


def test_current_task_line_appears_for_user(driver, seeded):
    """The most recently active open task surfaces as one profile line."""
    from app.services.task_store import TaskStore

    uid, session_b = seeded
    store = TaskStore(driver)
    task_id = store.create_task(uid, "Migrate telemetry to ClickHouse")
    try:
        result = retrieve_context(session_b, "What am I working on?", 1500, user_id=uid)

        assert "current task: migrate telemetry to clickhouse" in result["context"].lower()
        task_sources = [s for s in result["sources"] if s["type"] == "task"]
        assert task_sources, "no task source in provenance"
        assert task_sources[0]["details"]["task_id"] == task_id
        assert task_sources[0]["tier"] == "L3"
    finally:
        with driver.session() as s:
            s.run("MATCH (t:Task {id: $tid}) DETACH DELETE t", tid=task_id)


def test_no_task_line_when_user_has_no_open_tasks(seeded):
    uid, session_b = seeded
    result = retrieve_context(session_b, "What am I working on?", 1500, user_id=uid)
    assert "current task:" not in result["context"].lower()
