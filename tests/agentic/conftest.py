"""Fixtures for the agentic end-to-end harness.

These tests run the real thing: real FastAPI app, real Redis/Postgres/Neo4j, real
spaCy NER, real Ollama summarization. The only substitution is Celery eager mode,
which runs `process_conversation` inline so a test can assert on its effects
without racing a worker process. Set MEMCACHE_AGENTIC_WORKER=1 to instead dispatch
to a real worker you are running yourself.

Cleanup is scoped to the session ids a test created. It never issues a global
`MATCH (n) DETACH DELETE n` — that pattern in the older suite is why a stray test
entity ended up attached to a real episode.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.config import settings
from tests.agentic.ollama_agent import DEFAULT_MODEL, OllamaAgent, ollama_available

pytestmark = pytest.mark.agentic

USE_REAL_WORKER = os.environ.get("MEMCACHE_AGENTIC_WORKER") == "1"
AGENT_MODEL = os.environ.get("MEMCACHE_AGENTIC_MODEL", DEFAULT_MODEL)


@pytest.fixture(scope="session", autouse=True)
def _celery_eager():
    """Run enqueued tasks inline unless testing against a live worker."""
    from app.workers.celery_app import celery_app

    if USE_REAL_WORKER:
        yield
        return

    previous = (
        celery_app.conf.task_always_eager,
        celery_app.conf.task_eager_propagates,
    )
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    (
        celery_app.conf.task_always_eager,
        celery_app.conf.task_eager_propagates,
    ) = previous


@pytest.fixture(scope="session")
def _require_ollama():
    if not ollama_available(settings.ollama_base_url, AGENT_MODEL):
        pytest.skip(
            f"Ollama model {AGENT_MODEL!r} unavailable at {settings.ollama_base_url}"
        )


@pytest.fixture(scope="session")
def agent(_require_ollama) -> OllamaAgent:
    return OllamaAgent(base_url=settings.ollama_base_url, model=AGENT_MODEL)


@pytest.fixture(scope="session")
def api():
    """Real ASGI app over real backends."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def auth() -> dict[str, str]:
    key = next(iter(settings.get_valid_api_keys()))
    return {"X-API-Key": key}


@pytest.fixture(scope="session")
def neo4j_driver():
    from app.db.neo4j import create_driver_from_settings, ensure_constraints

    driver = create_driver_from_settings()
    ensure_constraints(driver)
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def pg_engine():
    from app.db.postgres import create_engine_from_settings, ensure_l2_schema

    engine = create_engine_from_settings()
    ensure_l2_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_id(neo4j_driver, pg_engine):
    """A unique session id, with storage for it removed afterwards."""
    import redis

    sid = f"agentic-{uuid.uuid4().hex[:12]}"
    yield sid

    client = redis.from_url(settings.redis_url, decode_responses=True)
    try:
        client.delete(f"session:{sid}")
    finally:
        client.close()

    with pg_engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM episodes WHERE session_id = %s", (sid,))

    # Scoped teardown: drop this session's episodes and any Decision/Preference
    # nodes hanging off them. Entities are intentionally left — they are shared
    # across sessions by design, which is what makes cross-session persona
    # accumulation possible in the first place.
    with neo4j_driver.session() as s:
        s.run(
            """
            MATCH (se:Session {id: $sid})
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            DETACH DELETE se, ep, dp
            """,
            sid=sid,
        )


@pytest.fixture
def ingest(api, auth):
    """Post messages to the real /memory/ingest endpoint."""

    def _ingest(sid: str, messages: list[dict[str, str]], metadata=None):
        response = api.post(
            "/memory/ingest",
            headers=auth,
            json={"session_id": sid, "messages": messages, "metadata": metadata},
        )
        assert response.status_code == 202, response.text
        return response.json()

    return _ingest


@pytest.fixture
def retrieve(api, auth):
    """Query the real /memory/retrieve endpoint."""

    def _retrieve(sid: str, query: str, max_tokens: int | None = None):
        body: dict[str, object] = {"session_id": sid, "query": query}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        response = api.post("/memory/retrieve", headers=auth, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    return _retrieve
