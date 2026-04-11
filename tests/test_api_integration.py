"""Integration tests for the Step 6 API endpoints against live services."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.main import app
from app.services.neo4j_store import Neo4jStore
from app.services.postgres_store import PostgresStore
from tests.conftest import (
    SYNTHETIC_L3_DECISIONS,
    SYNTHETIC_L3_ENTITIES_ACME,
    SYNTHETIC_L3_PREFERENCES,
    SYNTHETIC_L3_RELATED_PAIRS,
    SYNTHETIC_SESSION_1,
    unit_embedding_384,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dummy-api-key-123"}


@pytest.fixture(autouse=True)
def fake_query_embedder(monkeypatch):
    class _Vec:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class _Embedder:
        def encode(self, text, normalize_embeddings=True):
            text_lower = text.lower()
            if "john" in text_lower or "python" in text_lower or "dark mode" in text_lower:
                return _Vec(unit_embedding_384(primary_axis=0))
            if "postgresql" in text_lower or "database" in text_lower:
                return _Vec(unit_embedding_384(primary_axis=11))
            return _Vec(unit_embedding_384(primary_axis=200))

    monkeypatch.setattr("app.api.services.get_query_embedder", lambda: _Embedder())


@pytest.fixture
def retrieval_seed():
    import redis

    from app.db.neo4j import create_driver_from_settings, ensure_constraints

    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    pg_engine = create_engine_from_settings()
    ensure_l2_schema(pg_engine)
    neo4j_driver = create_driver_from_settings()
    ensure_constraints(neo4j_driver)

    session_id = f"retrieve-int-{uuid.uuid4()}"
    redis_key = f"session:{session_id}"
    now = datetime.now(timezone.utc)

    try:
        for message in SYNTHETIC_SESSION_1:
            payload = dict(message)
            payload["timestamp"] = now.isoformat()
            redis_client.rpush(redis_key, __import__("json").dumps(payload))
        redis_client.expire(redis_key, settings.redis_session_ttl_seconds)

        with session_scope(pg_engine) as session:
            store = PostgresStore(session)
            strong_id = store.insert_episode(
                session_id=session_id,
                summary="John from Acme prefers dark mode and chose Python.",
                embedding=unit_embedding_384(primary_axis=0),
                start_time=now,
                end_time=now,
                metadata={"synthetic": True},
            )
            store.insert_episode(
                session_id=session_id,
                summary="Unrelated hiking and weather discussion.",
                embedding=unit_embedding_384(primary_axis=200),
                start_time=now,
                end_time=now,
                metadata={"synthetic": True},
            )

        with neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        graph_store = Neo4jStore(neo4j_driver)
        graph_store.upsert_session(session_id)
        graph_store.upsert_episode(
            session_id,
            episode_id=strong_id,
            summary="John from Acme prefers dark mode and chose Python.",
        )
        graph_store.merge_entities(SYNTHETIC_L3_ENTITIES_ACME, episode_id=strong_id)
        graph_store.create_relationships(list(SYNTHETIC_L3_RELATED_PAIRS))
        graph_store.record_decisions_and_preferences(
            strong_id,
            decisions=list(SYNTHETIC_L3_DECISIONS),
            preferences=list(SYNTHETIC_L3_PREFERENCES),
        )

        yield session_id
    finally:
        redis_client.delete(redis_key)
        redis_client.close()
        with pg_engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM episodes WHERE session_id = %s", (session_id,))
        with neo4j_driver.session() as session:
            session.run("MATCH (s:Session {id: $sid}) DETACH DELETE s", sid=session_id)
        neo4j_driver.close()
        pg_engine.dispose()


def test_health_endpoint_with_live_backends(client, auth_headers):
    response = client.get("/health", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["redis"]["ok"] is True
    assert body["postgres"]["ok"] is True
    assert body["neo4j"]["ok"] is True


def test_ingest_writes_to_real_redis_and_returns_task_id(client, auth_headers):
    import redis

    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    session_id = f"api-int-{uuid.uuid4()}"
    redis_key = f"session:{session_id}"
    try:
        response = client.post(
            "/memory/ingest",
            headers=auth_headers,
            json={
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": "hello from api integration"},
                    {"role": "assistant", "content": "acknowledged"},
                ],
                "metadata": {"source": "integration"},
            },
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"
        assert body["session_id"] == session_id
        assert body["task_id"]

        recent = redis_client.lrange(redis_key, 0, 1)
        assert len(recent) == 2
        ttl = redis_client.ttl(redis_key)
        assert ttl > 0
        assert ttl <= settings.redis_session_ttl_seconds
    finally:
        redis_client.delete(redis_key)
        redis_client.close()


def test_retrieve_returns_hybrid_context_from_live_backends(client, auth_headers, retrieval_seed):
    response = client.post(
        "/memory/retrieve",
        headers=auth_headers,
        json={
            "session_id": retrieval_seed,
            "query": "What stack and preferences did John mention?",
            "max_tokens": 500,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "Recent Conversation:" in body["context"]
    assert "Relevant Past Episodes:" in body["context"]
    assert "Graph Facts:" in body["context"]
    assert "John from Acme prefers dark mode and chose Python." in body["context"]
    assert "Unrelated hiking and weather discussion." not in body["context"]
    assert "Decision: Use Python for the backend service." in body["context"]
    assert "Preference: Dark mode UI." in body["context"]
    source_types = {source["type"] for source in body["sources"]}
    assert "recent_message" in source_types
    assert "episode" in source_types
    assert "decision" in source_types or "preference" in source_types
    tiers = {source["tier"] for source in body["sources"]}
    assert "L1" in tiers
    assert "L2" in tiers
    assert "L3" in tiers
