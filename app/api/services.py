"""Shared wiring helpers for HTTP handlers."""

from __future__ import annotations

from functools import lru_cache

import redis
from neo4j import Driver
from sqlalchemy import Engine, text

from app.config import settings
from app.db.neo4j import create_driver_from_settings
from app.db.postgres import create_engine_from_settings, ensure_l2_schema
from app.services.redis_store import RedisStore
from app.workers.tasks import process_conversation


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """Create and cache the Redis client used by the API."""
    return redis.from_url(settings.redis_url, decode_responses=True)


def get_redis_store() -> RedisStore:
    """Construct the L1 store on top of the shared Redis client."""
    return RedisStore(get_redis_client())


@lru_cache(maxsize=1)
def get_postgres_engine() -> Engine:
    """Create and cache the PostgreSQL engine used by the API."""
    return create_engine_from_settings()


@lru_cache(maxsize=1)
def get_neo4j_driver() -> Driver:
    """Create and cache the Neo4j driver used by the API."""
    return create_driver_from_settings()


@lru_cache(maxsize=1)
def get_query_embedder():
    """Create and cache the query embedder used by retrieval."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def enqueue_conversation_task(
    session_id: str,
    messages: list[dict[str, str]],
    metadata: dict[str, object] | None,
):
    """Queue the Celery background job."""
    return process_conversation.delay(session_id, messages, metadata)


def check_redis_health() -> tuple[bool, str]:
    """Return Redis connectivity status for `/health`."""
    try:
        ok = bool(get_redis_client().ping())
    except Exception as exc:
        return False, f"error: {exc}"
    return (True, "ok") if ok else (False, "ping returned false")


def check_postgres_health() -> tuple[bool, str]:
    """Return PostgreSQL connectivity and schema readiness for `/health`."""
    try:
        engine = get_postgres_engine()
        ensure_l2_schema(engine)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.scalar() != 1:
                return False, "SELECT 1 returned unexpected result"
    except Exception as exc:
        return False, f"error: {exc}"
    return True, "ok"


def check_neo4j_health() -> tuple[bool, str]:
    """Return Neo4j connectivity status for `/health`."""
    try:
        with get_neo4j_driver().session() as session:
            record = session.run("RETURN 1 AS n").single()
            if record is None or record["n"] != 1:
                return False, "RETURN 1 returned unexpected result"
    except Exception as exc:
        return False, f"error: {exc}"
    return True, "ok"


def close_service_clients() -> None:
    """Dispose cached clients on app shutdown."""
    redis_client = get_redis_client.cache_info().currsize and get_redis_client()
    if redis_client:
        redis_client.close()
        get_redis_client.cache_clear()

    engine = get_postgres_engine.cache_info().currsize and get_postgres_engine()
    if engine:
        engine.dispose()
        get_postgres_engine.cache_clear()

    driver = get_neo4j_driver.cache_info().currsize and get_neo4j_driver()
    if driver:
        driver.close()
        get_neo4j_driver.cache_clear()

    cache_info = getattr(get_query_embedder, "cache_info", None)
    cache_clear = getattr(get_query_embedder, "cache_clear", None)
    if callable(cache_info) and callable(cache_clear) and cache_info().currsize:
        cache_clear()
