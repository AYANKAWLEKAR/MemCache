"""Unit tests for retrieval orchestration and token-aware context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.services.neo4j_store import GraphEntityRow
from app.services.postgres_store import EpisodeSearchResult
from app.services.retrieval import RetrievalError, retrieve_context


@dataclass
class _FakeSessionScope:
    session: object

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_embedder(monkeypatch):
    class _Vec:
        def tolist(self):
            return [0.5] * 384

    class _Embedder:
        def encode(self, _text, normalize_embeddings=True):
            return _Vec()

    monkeypatch.setattr("app.services.retrieval.api_services.get_query_embedder", lambda: _Embedder())


def test_retrieve_context_returns_redis_only_when_l2_l3_empty(monkeypatch):
    _patch_embedder(monkeypatch)
    redis_store = MagicMock()
    redis_store.get_recent_messages.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_postgres_engine", lambda: object())
    monkeypatch.setattr("app.services.retrieval.session_scope", lambda _engine: _FakeSessionScope(object()))

    pg_store = MagicMock()
    pg_store.search_episodes.return_value = []
    monkeypatch.setattr("app.services.retrieval.PostgresStore", lambda _session: pg_store)

    graph_store = MagicMock()
    graph_store.query_session_entities.return_value = []
    graph_store.query_decisions_preferences.return_value = {"decisions": [], "preferences": []}
    monkeypatch.setattr("app.services.retrieval.Neo4jStore", lambda _driver: graph_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_neo4j_driver", lambda: object())

    result = retrieve_context("sess-1", "hello")

    assert result["status"] == "ok"
    assert "Recent Conversation:" in result["context"]
    assert "user: hello" in result["context"]
    assert all(source["type"] == "recent_message" for source in result["sources"])
    assert all(source["tier"] == "L1" for source in result["sources"])


def test_retrieve_context_filters_episodes_by_similarity_threshold(monkeypatch):
    _patch_embedder(monkeypatch)
    redis_store = MagicMock()
    redis_store.get_recent_messages.return_value = [{"role": "user", "content": "What stack did John pick?"}]
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_postgres_engine", lambda: object())
    monkeypatch.setattr("app.services.retrieval.session_scope", lambda _engine: _FakeSessionScope(object()))

    now = datetime.now(timezone.utc)
    strong = EpisodeSearchResult(1, "sess-1", "John chose Python.", now, now, None, 0.1)
    weak = EpisodeSearchResult(2, "sess-1", "Weather discussion.", now, now, None, 0.5)
    pg_store = MagicMock()
    pg_store.search_episodes.return_value = [strong, weak]
    monkeypatch.setattr("app.services.retrieval.PostgresStore", lambda _session: pg_store)

    graph_store = MagicMock()
    graph_store.query_session_entities.return_value = []
    graph_store.query_decisions_preferences.return_value = {"decisions": [], "preferences": []}
    monkeypatch.setattr("app.services.retrieval.Neo4jStore", lambda _driver: graph_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_neo4j_driver", lambda: object())

    result = retrieve_context("sess-1", "What stack did John pick?")

    assert "Relevant Past Episodes:" in result["context"]
    assert "Episode 1: John chose Python." in result["context"]
    assert "Weather discussion." not in result["context"]
    episode_sources = [source for source in result["sources"] if source["type"] == "episode"]
    assert episode_sources and all(source["tier"] == "L2" for source in episode_sources)


def test_retrieve_context_includes_graph_facts(monkeypatch):
    _patch_embedder(monkeypatch)
    redis_store = MagicMock()
    redis_store.get_recent_messages.return_value = [{"role": "user", "content": "Tell me about John at Acme."}]
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_postgres_engine", lambda: object())
    monkeypatch.setattr("app.services.retrieval.session_scope", lambda _engine: _FakeSessionScope(object()))

    pg_store = MagicMock()
    pg_store.search_episodes.return_value = []
    monkeypatch.setattr("app.services.retrieval.PostgresStore", lambda _session: pg_store)

    graph_store = MagicMock()
    graph_store.query_session_entities.return_value = [
        GraphEntityRow(name="john", display_name="John"),
        GraphEntityRow(name="acme corp", display_name="Acme Corp"),
    ]
    graph_store.query_decisions_preferences.return_value = {
        "decisions": ["Use Python for the backend service."],
        "preferences": ["Dark mode UI."],
    }
    graph_store.query_related_entities.side_effect = lambda name, max_hops=2: ["acme corp", "python"] if name == "john" else []
    monkeypatch.setattr("app.services.retrieval.Neo4jStore", lambda _driver: graph_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_neo4j_driver", lambda: object())

    result = retrieve_context("sess-1", "Tell me about John at Acme.")

    assert "Graph Facts:" in result["context"]
    assert "Session entity: John" in result["context"]
    assert "Related to John: acme corp, python" in result["context"]
    assert "Decision: Use Python for the backend service." in result["context"]
    assert "Preference: Dark mode UI." in result["context"]
    graph_sources = [source for source in result["sources"] if source["type"] != "recent_message"]
    assert graph_sources and all(source["tier"] in {"L2", "L3"} for source in graph_sources)


def test_retrieve_context_adds_warnings_for_degraded_backends(monkeypatch):
    _patch_embedder(monkeypatch)
    redis_store = MagicMock()
    redis_store.get_recent_messages.return_value = [{"role": "user", "content": "hello"}]
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_postgres_engine", lambda: object())
    monkeypatch.setattr("app.services.retrieval.session_scope", lambda _engine: _FakeSessionScope(object()))

    def raise_pg(_session):
        raise RuntimeError("postgres down")

    monkeypatch.setattr("app.services.retrieval.PostgresStore", raise_pg)

    def raise_graph(_driver):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr("app.services.retrieval.Neo4jStore", raise_graph)
    monkeypatch.setattr("app.services.retrieval.api_services.get_neo4j_driver", lambda: object())

    result = retrieve_context("sess-1", "hello")

    assert result["status"] == "degraded"
    assert any("PostgreSQL retrieval unavailable" in warning for warning in result["warnings"])
    assert any("Neo4j retrieval unavailable" in warning for warning in result["warnings"])


def test_retrieve_context_truncates_with_warning(monkeypatch):
    _patch_embedder(monkeypatch)
    redis_store = MagicMock()
    redis_store.get_recent_messages.return_value = [
        {"role": "user", "content": "This is a fairly long message about John and Acme Corp."},
        {"role": "assistant", "content": "Acknowledged and recorded."},
    ]
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_postgres_engine", lambda: object())
    monkeypatch.setattr("app.services.retrieval.session_scope", lambda _engine: _FakeSessionScope(object()))

    pg_store = MagicMock()
    pg_store.search_episodes.return_value = []
    monkeypatch.setattr("app.services.retrieval.PostgresStore", lambda _session: pg_store)

    graph_store = MagicMock()
    graph_store.query_session_entities.return_value = []
    graph_store.query_decisions_preferences.return_value = {"decisions": [], "preferences": []}
    monkeypatch.setattr("app.services.retrieval.Neo4jStore", lambda _driver: graph_store)
    monkeypatch.setattr("app.services.retrieval.api_services.get_neo4j_driver", lambda: object())

    result = retrieve_context("sess-1", "hello", max_tokens=12)

    assert result["context"]
    assert any("Context truncated" in warning or "Context reduced" in warning for warning in result["warnings"])


def test_retrieve_context_raises_when_redis_fails(monkeypatch):
    redis_store = MagicMock()
    redis_store.get_recent_messages.side_effect = RuntimeError("redis down")
    monkeypatch.setattr("app.services.retrieval.api_services.get_redis_store", lambda: redis_store)

    with pytest.raises(RetrievalError):
        retrieve_context("sess-1", "hello")
