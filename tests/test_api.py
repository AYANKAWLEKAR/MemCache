"""Fast unit tests for the Step 6 API surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_still_responds(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "memory-cache", "status": "running"}


def test_health_rejects_missing_api_key(client):
    response = client.get("/health")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_health_rejects_invalid_api_key(client):
    response = client.get("/health", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_health_returns_ok_when_all_backends_pass(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.api_services.check_redis_health", lambda: (True, "ok"))
    monkeypatch.setattr("app.api.routes.api_services.check_postgres_health", lambda: (True, "ok"))
    monkeypatch.setattr("app.api.routes.api_services.check_neo4j_health", lambda: (True, "ok"))

    response = client.get("/health", headers={"X-API-Key": "dummy-api-key-123"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "redis": {"ok": True, "detail": "ok"},
        "postgres": {"ok": True, "detail": "ok"},
        "neo4j": {"ok": True, "detail": "ok"},
    }


def test_health_returns_degraded_when_backend_fails(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.api_services.check_redis_health", lambda: (True, "ok"))
    monkeypatch.setattr(
        "app.api.routes.api_services.check_postgres_health",
        lambda: (False, "error: postgres unavailable"),
    )
    monkeypatch.setattr("app.api.routes.api_services.check_neo4j_health", lambda: (True, "ok"))

    response = client.get("/health", headers={"X-API-Key": "dummy-api-key-123"})

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["postgres"] == {
        "ok": False,
        "detail": "error: postgres unavailable",
    }


def test_ingest_accepts_valid_request_and_enqueues_task(client, monkeypatch):
    store = MagicMock()
    monkeypatch.setattr("app.api.routes.api_services.get_redis_store", lambda: store)
    monkeypatch.setattr(
        "app.api.routes.api_services.enqueue_conversation_task",
        lambda session_id, messages, metadata: SimpleNamespace(id="task-123"),
    )

    payload = {
        "session_id": "sess-api-1",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "metadata": {"source": "unit-test"},
    }
    response = client.post(
        "/memory/ingest",
        json=payload,
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "task_id": "task-123",
        "session_id": "sess-api-1",
    }
    store.append_messages.assert_called_once_with(
        "sess-api-1",
        payload["messages"],
    )


def test_ingest_rejects_missing_api_key(client):
    response = client.post(
        "/memory/ingest",
        json={"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


def test_ingest_rejects_invalid_payload(client):
    response = client.post(
        "/memory/ingest",
        json={"session_id": "", "messages": []},
        headers={"X-API-Key": "dummy-api-key-123"},
    )
    assert response.status_code == 422


def test_ingest_returns_503_when_redis_write_fails(client, monkeypatch):
    store = MagicMock()
    store.append_messages.side_effect = RuntimeError("redis down")
    monkeypatch.setattr("app.api.routes.api_services.get_redis_store", lambda: store)

    response = client.post(
        "/memory/ingest",
        json={
            "session_id": "sess-api-2",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to persist messages to Redis"


def test_ingest_returns_503_when_enqueue_fails_after_redis_write(client, monkeypatch):
    store = MagicMock()
    monkeypatch.setattr("app.api.routes.api_services.get_redis_store", lambda: store)

    def raise_enqueue(*_args, **_kwargs):
        raise RuntimeError("broker down")

    monkeypatch.setattr(
        "app.api.routes.api_services.enqueue_conversation_task",
        raise_enqueue,
    )

    payload = {
        "session_id": "sess-api-3",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"source": "unit-test"},
    }
    response = client.post(
        "/memory/ingest",
        json=payload,
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to enqueue background processing"
    store.append_messages.assert_called_once_with("sess-api-3", payload["messages"])


def test_retrieve_rejects_missing_api_key(client):
    response = client.post("/memory/retrieve", json={"session_id": "s1", "query": "hello"})
    assert response.status_code == 401


def test_retrieve_rejects_invalid_payload(client):
    response = client.post(
        "/memory/retrieve",
        json={"session_id": "", "query": ""},
        headers={"X-API-Key": "dummy-api-key-123"},
    )
    assert response.status_code == 422


def test_retrieve_returns_service_result(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.retrieve_context",
        lambda session_id, query, max_tokens: {
            "context": "Recent Conversation:\nuser: hello",
            "sources": [
                {
                    "type": "recent_message",
                    "tier": "L1",
                    "details": {"session_id": session_id, "index": 0},
                }
            ],
            "status": "ok",
            "warnings": [],
        },
    )

    response = client.post(
        "/memory/retrieve",
        json={"session_id": "sess-r1", "query": "hello"},
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "Recent Conversation" in response.json()["context"]
    assert response.json()["sources"][0]["tier"] == "L1"


def test_retrieve_returns_degraded_payload(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.retrieve_context",
        lambda session_id, query, max_tokens: {
            "context": "Recent Conversation:\nuser: hello",
            "sources": [
                {
                    "type": "recent_message",
                    "tier": "L1",
                    "details": {"session_id": session_id, "index": 0},
                }
            ],
            "status": "degraded",
            "warnings": ["Neo4j retrieval unavailable; returning partial context"],
        },
    )

    response = client.post(
        "/memory/retrieve",
        json={"session_id": "sess-r2", "query": "hello"},
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["warnings"]


def test_retrieve_returns_503_when_redis_context_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.retrieve_context",
        MagicMock(side_effect=RuntimeError("Redis retrieval failed")),
    )
    monkeypatch.setattr(
        "app.api.routes.RetrievalError",
        RuntimeError,
    )

    response = client.post(
        "/memory/retrieve",
        json={"session_id": "sess-r3", "query": "hello"},
        headers={"X-API-Key": "dummy-api-key-123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Redis retrieval failed"
