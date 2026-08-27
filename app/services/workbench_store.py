"""L4 workbench: durable tool-call log — record, filter, claim, failure recall.

Every tool invocation is persisted with hard byte caps: outputs truncate
aggressively while errors keep a much larger cap, because a stack trace is the
highest-value payload in the tier. `call_hash` canonicalizes the call (sorted
keys, stable separators) so "have I already tried this?" is one indexed lookup.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from app.config import settings

_VALID_STATUSES = frozenset({"ok", "error"})

#: Shared by every SELECT / RETURNING so row parsing lives in one place.
_COLUMNS = (
    "id, session_id, user_id, task_id, episode_id, tool_name, args, status, "
    "output, error, output_bytes, truncated, call_hash, duration_ms, created_at"
)

_INSERT_SQL = text(
    """
    INSERT INTO tool_calls (
        session_id, user_id, task_id, tool_name, args, status, output, error,
        output_bytes, truncated, call_hash, duration_ms, created_at
    ) VALUES (
        :session_id, :user_id, :task_id, :tool_name, CAST(:args AS JSONB),
        :status, :output, :error, :output_bytes, :truncated, :call_hash,
        :duration_ms, :created_at
    )
    RETURNING id
    """
)


@dataclass(frozen=True)
class RecordedToolCall:
    """Receipt for one recorded call: enough to dedup and to warn about cuts."""

    id: int
    call_hash: str
    truncated: bool


@dataclass(frozen=True)
class ToolCallRow:
    """One full `tool_calls` row as read back from Postgres."""

    id: int
    session_id: str
    user_id: str | None
    task_id: str | None
    episode_id: int | None
    tool_name: str
    args: dict | None
    status: str
    output: str | None
    error: str | None
    output_bytes: int
    truncated: bool
    call_hash: str
    duration_ms: int | None
    created_at: datetime


def ensure_l4_schema(engine: Engine) -> None:
    """Create the `tool_calls` table and its indexes if missing. Idempotent.

    `episode_id` references `episodes(id)`, so callers must run
    :func:`app.db.postgres.ensure_l2_schema` first — the FK cannot be created
    before that table exists. `ON DELETE SET NULL` keeps the record that work
    happened even when its episode is deleted.
    """
    ddl_table = """
        CREATE TABLE IF NOT EXISTS tool_calls (
            id            BIGSERIAL PRIMARY KEY,
            session_id    VARCHAR(255) NOT NULL,
            user_id       VARCHAR(255),
            task_id       VARCHAR(64),
            episode_id    INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
            tool_name     VARCHAR(255) NOT NULL,
            args          JSONB,
            status        VARCHAR(32)  NOT NULL,
            output        TEXT,
            error         TEXT,
            output_bytes  INTEGER      NOT NULL,
            truncated     BOOLEAN      NOT NULL,
            call_hash     CHAR(64)     NOT NULL,
            duration_ms   INTEGER,
            created_at    TIMESTAMPTZ  NOT NULL
        )
    """
    ddl_indexes = (
        "CREATE INDEX IF NOT EXISTS idx_tool_calls_session_created "
        "ON tool_calls (session_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_tool_calls_user_created "
        "ON tool_calls (user_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_tool_calls_call_hash ON tool_calls (call_hash)",
        "CREATE INDEX IF NOT EXISTS idx_tool_calls_episode_id ON tool_calls (episode_id)",
        "CREATE INDEX IF NOT EXISTS idx_tool_calls_task_id ON tool_calls (task_id)",
    )
    with engine.begin() as conn:
        conn.execute(text(ddl_table))
        for ddl in ddl_indexes:
            conn.execute(text(ddl))


def canonical_call_hash(tool_name: str, args: dict | None) -> str:
    """sha256 hex over `tool_name` + canonical JSON of `args` (None means {}).

    Sorted keys and fixed separators make the hash key-order invariant:
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` are the same call — that
    equality is what makes "have I already tried this" answerable.
    """
    canonical = json.dumps(
        args if args is not None else {}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256((tool_name + canonical).encode("utf-8")).hexdigest()


def _utf8_len(value: str | None) -> int:
    return len(value.encode("utf-8")) if value is not None else 0


def _truncate_utf8(value: str | None, max_bytes: int) -> tuple[str | None, bool]:
    """Cap `value` at `max_bytes` of UTF-8 without splitting a multibyte char.

    The input is valid UTF-8, so the only damage a byte-prefix can carry is one
    partial sequence at its very end; ``errors="ignore"`` drops exactly that.
    """
    if value is None:
        return None, False
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value, False
    return raw[:max_bytes].decode("utf-8", errors="ignore"), True


def _to_row(row: Any) -> ToolCallRow:
    m = row._mapping
    return ToolCallRow(
        id=int(m["id"]),
        session_id=m["session_id"],
        user_id=m["user_id"],
        task_id=m["task_id"],
        episode_id=m["episode_id"],
        tool_name=m["tool_name"],
        args=m["args"],
        status=m["status"],
        output=m["output"],
        error=m["error"],
        output_bytes=int(m["output_bytes"]),
        truncated=bool(m["truncated"]),
        call_hash=m["call_hash"],
        duration_ms=m["duration_ms"],
        created_at=m["created_at"],
    )


def record_tool_call(
    engine: Engine,
    *,
    session_id: str,
    tool_name: str,
    status: str,
    args: dict | None = None,
    output: str | None = None,
    error: str | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
    duration_ms: int | None = None,
) -> RecordedToolCall:
    """Persist one tool call, truncating payloads to the configured byte caps.

    `output_bytes` records the true combined pre-truncation UTF-8 size of
    output + error, so a stored fragment is always identifiable as one.
    Raises ValueError for a status outside {"ok", "error"}.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(_VALID_STATUSES)}, got {status!r}"
        )
    call_hash = canonical_call_hash(tool_name, args)
    true_bytes = _utf8_len(output) + _utf8_len(error)
    output, output_cut = _truncate_utf8(output, settings.workbench_output_max_bytes)
    error, error_cut = _truncate_utf8(error, settings.workbench_error_max_bytes)
    truncated = output_cut or error_cut
    params = {
        "session_id": session_id,
        "user_id": user_id,
        "task_id": task_id,
        "tool_name": tool_name,
        "args": json.dumps(args) if args is not None else None,
        "status": status,
        "output": output,
        "error": error,
        "output_bytes": true_bytes,
        "truncated": truncated,
        "call_hash": call_hash,
        "duration_ms": duration_ms,
        "created_at": datetime.now(UTC),
    }
    with engine.begin() as conn:
        new_id = conn.execute(_INSERT_SQL, params).scalar_one()
    return RecordedToolCall(id=int(new_id), call_hash=call_hash, truncated=truncated)


def get_tool_call(engine: Engine, tool_call_id: int) -> ToolCallRow | None:
    """One row by id; None when absent. Used to hydrate an activated ToolCall node."""
    sql = text(f"SELECT {_COLUMNS} FROM tool_calls WHERE id = :id")
    with engine.begin() as conn:
        row = conn.execute(sql, {"id": tool_call_id}).fetchone()
    return _to_row(row) if row is not None else None


def recent_tool_calls(
    engine: Engine,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    tool_name: str | None = None,
    call_hash: str | None = None,
    limit: int = 20,
) -> list[ToolCallRow]:
    """Newest-first tool calls; every given filter is ANDed.

    Only hard-coded column names enter the SQL text — all values are bound
    parameters, never interpolated.
    """
    if limit <= 0:
        return []
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit}
    for column, value in (
        ("session_id", session_id),
        ("user_id", user_id),
        ("task_id", task_id),
        ("status", status),
        ("tool_name", tool_name),
        ("call_hash", call_hash),
    ):
        if value is not None:
            conditions.append(f"{column} = :{column}")
            params[column] = value
    where = f"WHERE {' AND '.join(conditions)} " if conditions else ""
    sql = text(
        f"SELECT {_COLUMNS} FROM tool_calls "
        f"{where}ORDER BY created_at DESC, id DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        return [_to_row(r) for r in conn.execute(sql, params)]


def claim_tool_calls(
    engine: Engine,
    *,
    session_id: str,
    episode_id: int,
    task_id: str | None = None,
) -> list[ToolCallRow]:
    """Link every still-unlinked call in the session to `episode_id`.

    One UPDATE: rows with ``episode_id IS NULL`` gain the episode, and — when
    `task_id` is given — rows whose own ``task_id IS NULL`` gain it too
    (COALESCE keeps any pre-existing task). Returns the claimed rows, newest
    first; a second claim on the same session returns nothing.
    """
    sql = text(
        f"""
        UPDATE tool_calls
        SET episode_id = :episode_id,
            task_id = COALESCE(task_id, :task_id)
        WHERE session_id = :session_id AND episode_id IS NULL
        RETURNING {_COLUMNS}
        """
    )
    params = {"episode_id": episode_id, "task_id": task_id, "session_id": session_id}
    with engine.begin() as conn:
        rows = [_to_row(r) for r in conn.execute(sql, params)]
    rows.sort(key=lambda r: (r.created_at, r.id), reverse=True)
    return rows


def failed_calls(
    engine: Engine,
    *,
    user_id: str,
    task_ids: list[str] | None = None,
    task_id: str | None = None,
    limit: int = 5,
) -> list[ToolCallRow]:
    """Most recent failed calls, lineage-ranked: the leaf task's own failures
    first, then any ancestor's, then the user's other failures by recency.

    `task_ids` is the active lineage `[leaf, parent, grandparent, ...]`;
    `task_id` (legacy) is folded into a one-element lineage. Scope UNIONS —
    it never switches. Switching was a live bug: a failure stamped to an
    older task vanished the moment any unrelated newer task became active.
    The rank CASE needs no COALESCE: `task_id IN (...)` on a NULL yields NULL,
    and `CASE WHEN NULL` falls through to ELSE 0 — untasked failures rank
    last, which is what we want.
    """
    if limit <= 0:
        return []
    lineage = list(task_ids or ([task_id] if task_id else []))
    leaf = lineage[0] if lineage else None
    # An empty IN-list is invalid SQL; use a value no task id can equal.
    in_list = lineage or ["__none__"]
    sql = text(
        f"""
        SELECT {_COLUMNS} FROM tool_calls
        WHERE status = 'error'
          AND (user_id = :user_id OR task_id IN :lineage)
        ORDER BY CASE WHEN task_id = :leaf THEN 2
                      WHEN task_id IN :lineage THEN 1
                      ELSE 0 END DESC,
                 created_at DESC, id DESC
        LIMIT :limit
        """
    ).bindparams(bindparam("lineage", expanding=True))
    params = {"user_id": user_id, "lineage": in_list, "leaf": leaf, "limit": limit}
    with engine.connect() as conn:
        return [_to_row(r) for r in conn.execute(sql, params)]
