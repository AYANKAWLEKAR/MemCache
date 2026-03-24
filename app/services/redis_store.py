"""L1 Redis store: per-session lists of recent messages with TTL and size cap."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import redis

from app.config import settings


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


class RedisStore:
    """Append and read conversation turns in Redis (LIST per session)."""

    def __init__(
        self,
        client: redis.Redis,
        *,
        ttl_seconds: int | None = None,
        max_messages_per_session: int | None = None,
    ) -> None:
        self._r = client
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else settings.redis_session_ttl_seconds
        )
        self._max = (
            max_messages_per_session
            if max_messages_per_session is not None
            else settings.redis_max_messages_per_session
        )

    def append_messages(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        """LPUSH JSON messages, trim to max length, refresh TTL."""
        if not messages:
            return
        key = _session_key(session_id)
        ts = datetime.now(timezone.utc).isoformat()
        pipe = self._r.pipeline()
        for m in messages:
            payload = {
                "role": m["role"],
                "content": m["content"],
                "timestamp": ts,
            }
            pipe.lpush(key, json.dumps(payload))
        pipe.ltrim(key, 0, self._max - 1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return up to `limit` most recent messages, oldest first."""
        if limit <= 0:
            return []
        key = _session_key(session_id)
        raw = self._r.lrange(key, 0, limit - 1)
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            out.append(json.loads(item))
        out.reverse()
        return out
