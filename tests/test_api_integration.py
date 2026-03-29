"""Integration tests for the Step 6 API endpoints against live services."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dummy-api-key-123"}


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
