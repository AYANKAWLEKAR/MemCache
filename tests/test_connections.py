"""Integration tests for Redis, PostgreSQL, and Neo4j connectivity.

Requires docker-compose stack to be running:
    docker-compose up -d
"""

import pytest

pytestmark = pytest.mark.integration

from app.config import settings


@pytest.fixture
def redis_client():
    """Redis client for tests."""
    import redis
    client = redis.from_url(settings.redis_url)
    yield client
    client.close()


@pytest.fixture
def postgres_engine():
    """PostgreSQL engine for tests."""
    from sqlalchemy import create_engine, text
    engine = create_engine(settings.postgres_url)
    yield engine
    engine.dispose()


@pytest.fixture
def neo4j_driver():
    """Neo4j driver for tests."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    yield driver
    driver.close()


def test_redis_connection(redis_client):
    """Redis PING returns True when connected."""
    assert redis_client.ping() is True


def test_postgres_connection(postgres_engine):
    """PostgreSQL accepts connections and pgvector extension exists."""
    from sqlalchemy import text
    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        # Verify pgvector is available
        result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        assert result.fetchone() is not None


def test_neo4j_connection(neo4j_driver):
    """Neo4j driver can execute a simple query."""
    with neo4j_driver.session() as session:
        result = session.run("RETURN 1 AS n")
        record = result.single()
        assert record is not None
        assert record["n"] == 1
