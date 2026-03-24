"""Integration tests for L1 Redis store against real Redis (docker-compose)."""

import uuid

import pytest

pytestmark = pytest.mark.integration

from app.config import settings
from app.services.redis_store import RedisStore


@pytest.fixture
def redis_store_integration():
    import redis

    client = redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisStore(client)
    created_keys: list[str] = []
    yield store, client, created_keys
    for k in created_keys:
        client.delete(k)
    client.close()


def test_append_and_get_against_real_redis(redis_store_integration):
    store, client, keys = redis_store_integration
    sid = str(uuid.uuid4())
    keys.append(f"session:{sid}")

    store.append_messages(
        sid,
        [
            {"role": "user", "content": "hello from integration"},
            {"role": "assistant", "content": "acknowledged"},
        ],
    )

    recent = store.get_recent_messages(sid, limit=10)
    assert len(recent) == 2
    assert recent[0]["role"] == "user"
    assert recent[0]["content"] == "hello from integration"
    assert recent[1]["role"] == "assistant"
    ttl = client.ttl(f"session:{sid}")
    assert ttl > 0
    assert ttl <= settings.redis_session_ttl_seconds


def test_list_cap_against_real_redis(redis_store_integration):
    store, client, keys = redis_store_integration
    sid = str(uuid.uuid4())
    keys.append(f"session:{sid}")

    for i in range(210):
        store.append_messages(sid, [{"role": "user", "content": str(i)}])

    assert client.llen(f"session:{sid}") == 200
