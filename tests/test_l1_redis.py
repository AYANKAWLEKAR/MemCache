"""Unit tests for L1 Redis store using fakeredis."""

import json

import pytest

from app.services.redis_store import RedisStore
from tests.conftest import SYNTHETIC_SESSION_1, SYNTHETIC_SESSION_2


@pytest.fixture
def store(fake_redis):
    return RedisStore(
        fake_redis,
        ttl_seconds=86400,
        max_messages_per_session=200,
    )


def test_append_and_get_recent_chronological_order(store, fake_redis):
    store.append_messages("s1", [{"role": "user", "content": "a"}])
    store.append_messages("s1", [{"role": "assistant", "content": "b"}])
    store.append_messages("s1", [{"role": "user", "content": "c"}])

    recent = store.get_recent_messages("s1", limit=10)
    assert [m["role"] for m in recent] == ["user", "assistant", "user"]
    assert [m["content"] for m in recent] == ["a", "b", "c"]
    for m in recent:
        assert "timestamp" in m


def test_get_recent_respects_limit(store):
    for i in range(5):
        store.append_messages("s1", [{"role": "user", "content": str(i)}])
    recent = store.get_recent_messages("s1", limit=3)
    assert len(recent) == 3
    assert [m["content"] for m in recent] == ["2", "3", "4"]


def test_get_recent_limit_zero_empty(store):
    store.append_messages("s1", [{"role": "user", "content": "x"}])
    assert store.get_recent_messages("s1", limit=0) == []


def test_empty_session_returns_empty(store):
    assert store.get_recent_messages("missing", limit=10) == []


def test_append_empty_noop(store, fake_redis):
    store.append_messages("s1", [])
    assert fake_redis.keys() == []


def test_session_key_isolation(store):
    store.append_messages("a", [{"role": "user", "content": "1"}])
    store.append_messages("b", [{"role": "user", "content": "2"}])
    assert store.get_recent_messages("a", 10)[0]["content"] == "1"
    assert store.get_recent_messages("b", 10)[0]["content"] == "2"


def test_ltrim_caps_at_200(store, fake_redis):
    for i in range(250):
        store.append_messages("cap", [{"role": "user", "content": str(i)}])
    key = "session:cap"
    assert fake_redis.llen(key) == 200
    # Newest wins: last appended index 249 should be present
    raw = fake_redis.lrange(key, 0, 0)
    top = json.loads(raw[0])
    assert top["content"] == "249"


def test_expire_set(store, fake_redis):
    store.append_messages("ttl", [{"role": "user", "content": "x"}])
    ttl = fake_redis.ttl("session:ttl")
    assert ttl > 0
    assert ttl <= 86400


def test_synthetic_sessions(store):
    store.append_messages("syn1", SYNTHETIC_SESSION_1)
    store.append_messages("syn2", SYNTHETIC_SESSION_2)
    r1 = store.get_recent_messages("syn1", limit=10)
    r2 = store.get_recent_messages("syn2", limit=10)
    assert len(r1) == 2
    assert len(r2) == 1
    assert "Acme" in r1[0]["content"]
    assert "PostgreSQL" in r2[0]["content"]
