"""L4 workbench tool-call log: record, filter, claim, failure recall. Live Postgres."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.config import settings
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.postgres_store import PostgresStore
from app.services.workbench_store import (
    canonical_call_hash,
    claim_tool_calls,
    ensure_l4_schema,
    failed_calls,
    recent_tool_calls,
    record_tool_call,
)
from tests.conftest import unit_embedding_384

pytestmark = pytest.mark.integration


@pytest.fixture
def engine():
    eng = create_engine_from_settings()
    ensure_l2_schema(eng)
    ensure_l4_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def ids(engine):
    """UUID-scoped session/user/task ids; teardown deletes exactly this test's rows."""
    tag = uuid.uuid4().hex[:12]
    scope = {
        "session": f"wb-s-{tag}",
        "session_other": f"wb-s2-{tag}",
        "user": f"wb-u-{tag}",
        "user_other": f"wb-u2-{tag}",
        "task": f"wb-t-{tag}",
        "task_other": f"wb-t2-{tag}",
    }
    yield scope
    with engine.begin() as conn:
        for sid in (scope["session"], scope["session_other"]):
            conn.execute(
                text("DELETE FROM tool_calls WHERE session_id = :sid"), {"sid": sid}
            )
        for uid in (scope["user"], scope["user_other"]):
            conn.execute(
                text("DELETE FROM tool_calls WHERE user_id = :uid"), {"uid": uid}
            )
        for sid in (scope["session"], scope["session_other"]):
            conn.execute(
                text("DELETE FROM episodes WHERE session_id = :sid"), {"sid": sid}
            )


def _insert_episode(engine, session_id: str, user_id: str | None = None) -> int:
    """Real L2 row so the tool_calls FK has something to point at."""
    now = datetime.now(timezone.utc)
    with session_scope(engine) as s:
        return PostgresStore(s).insert_episode(
            session_id=session_id,
            summary="workbench test episode",
            embedding=unit_embedding_384(primary_axis=1),
            start_time=now,
            end_time=now,
            metadata=None,
            user_id=user_id,
        )


# ---------------------------------------------------------------- schema


def test_schema_is_idempotent(engine):
    """L2 first (the FK needs episodes), then L4 twice — no error either time."""
    ensure_l2_schema(engine)
    ensure_l4_schema(engine)
    ensure_l4_schema(engine)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM tool_calls WHERE FALSE")).scalar_one()
    assert n == 0  # table exists and is queryable


# ---------------------------------------------------------------- hashing


def test_hash_is_key_order_invariant():
    assert canonical_call_hash("read_file", {"a": 1, "b": 2}) == canonical_call_hash(
        "read_file", {"b": 2, "a": 1}
    )


def test_hash_treats_none_args_as_empty_dict():
    assert canonical_call_hash("read_file", None) == canonical_call_hash("read_file", {})


def test_hash_distinguishes_tool_and_args():
    base = canonical_call_hash("read_file", {"path": "x"})
    assert canonical_call_hash("write_file", {"path": "x"}) != base
    assert canonical_call_hash("read_file", {"path": "y"}) != base
    assert len(base) == 64
    int(base, 16)  # valid hex


# ---------------------------------------------------------------- record


def test_record_round_trip_of_every_field(engine, ids):
    before = datetime.now(timezone.utc)
    rec = record_tool_call(
        engine,
        session_id=ids["session"],
        tool_name="grep",
        status="ok",
        args={"pattern": "TODO", "path": "src"},
        output="3 matches",
        error=None,
        user_id=ids["user"],
        task_id=ids["task"],
        duration_ms=42,
    )
    assert rec.truncated is False
    assert rec.call_hash == canonical_call_hash("grep", {"path": "src", "pattern": "TODO"})

    rows = recent_tool_calls(engine, session_id=ids["session"])
    assert len(rows) == 1
    row = rows[0]
    assert row.id == rec.id
    assert row.session_id == ids["session"]
    assert row.user_id == ids["user"]
    assert row.task_id == ids["task"]
    assert row.episode_id is None
    assert row.tool_name == "grep"
    assert row.args == {"pattern": "TODO", "path": "src"}
    assert row.status == "ok"
    assert row.output == "3 matches"
    assert row.error is None
    assert row.output_bytes == len("3 matches".encode("utf-8"))
    assert row.truncated is False
    assert row.call_hash == rec.call_hash
    assert row.duration_ms == 42
    assert row.created_at.tzinfo is not None
    assert before <= row.created_at <= datetime.now(timezone.utc)


def test_record_rejects_unknown_status(engine, ids):
    with pytest.raises(ValueError):
        record_tool_call(
            engine, session_id=ids["session"], tool_name="grep", status="pending"
        )


# ---------------------------------------------------------------- truncation


def test_output_truncates_at_cap_while_large_error_survives(engine, ids):
    out_cap = settings.workbench_output_max_bytes
    big_output = "x" * (out_cap + 500)
    big_error = "e" * (out_cap + 500)  # over the output cap, well under the error cap
    rec = record_tool_call(
        engine,
        session_id=ids["session"],
        tool_name="build",
        status="error",
        output=big_output,
        error=big_error,
    )
    assert rec.truncated is True
    row = recent_tool_calls(engine, session_id=ids["session"])[0]
    assert len(row.output.encode("utf-8")) == out_cap  # cut at exactly the cap
    assert row.error == big_error  # intact despite exceeding the *output* cap
    assert row.truncated is True
    assert row.output_bytes == len(big_output) + len(big_error)  # ASCII: bytes == chars


def test_error_truncates_at_its_own_bigger_cap(engine, ids):
    err_cap = settings.workbench_error_max_bytes
    big_error = "e" * (err_cap + 100)
    record_tool_call(
        engine,
        session_id=ids["session"],
        tool_name="build",
        status="error",
        error=big_error,
    )
    row = recent_tool_calls(engine, session_id=ids["session"])[0]
    assert len(row.error.encode("utf-8")) == err_cap
    assert row.truncated is True
    assert row.output_bytes == err_cap + 100


def test_multibyte_truncation_never_splits_a_char(engine, ids):
    out_cap = settings.workbench_output_max_bytes
    # "€" is 3 UTF-8 bytes; with an 8192 cap the boundary falls mid-char,
    # which is exactly what this test needs to bite.
    assert out_cap % 3 != 0
    big = "€" * (out_cap // 3 + 10)
    record_tool_call(
        engine, session_id=ids["session"], tool_name="cat", status="ok", output=big
    )
    row = recent_tool_calls(engine, session_id=ids["session"])[0]
    assert row.truncated is True
    assert set(row.output) == {"€"}  # no replacement chars, no broken sequences
    stored = len(row.output.encode("utf-8"))
    assert stored <= out_cap
    assert stored > out_cap - 3  # cut at the last whole char, not earlier


# ---------------------------------------------------------------- recent filters


@pytest.fixture
def seeded_calls(engine, ids):
    """Three calls in the main session/user plus one in another; oldest to newest."""
    r1 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="grep",
        status="ok",
        args={"q": 1},
    )
    r2 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="build",
        status="error",
        error="boom",
    )
    r3 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="grep",
        status="ok",
        args={"q": 2},
    )
    r_other = record_tool_call(
        engine,
        session_id=ids["session_other"],
        user_id=ids["user_other"],
        tool_name="grep",
        status="ok",
    )
    return {"r1": r1, "r2": r2, "r3": r3, "r_other": r_other}


def test_recent_filters_by_session_and_orders_newest_first(engine, ids, seeded_calls):
    rows = recent_tool_calls(engine, session_id=ids["session"])
    assert [r.id for r in rows] == [
        seeded_calls["r3"].id,
        seeded_calls["r2"].id,
        seeded_calls["r1"].id,
    ]


def test_recent_filters_by_user(engine, ids, seeded_calls):
    rows = recent_tool_calls(engine, user_id=ids["user_other"])
    assert [r.id for r in rows] == [seeded_calls["r_other"].id]


def test_recent_filters_by_status(engine, ids, seeded_calls):
    rows = recent_tool_calls(engine, session_id=ids["session"], status="error")
    assert [r.id for r in rows] == [seeded_calls["r2"].id]


def test_recent_filters_are_anded(engine, ids, seeded_calls):
    rows = recent_tool_calls(engine, session_id=ids["session"], tool_name="grep")
    assert [r.id for r in rows] == [seeded_calls["r3"].id, seeded_calls["r1"].id]


def test_recent_filters_by_call_hash(engine, ids, seeded_calls):
    wanted = canonical_call_hash("grep", {"q": 1})
    rows = recent_tool_calls(engine, session_id=ids["session"], call_hash=wanted)
    assert [r.id for r in rows] == [seeded_calls["r1"].id]


def test_recent_respects_limit(engine, ids, seeded_calls):
    rows = recent_tool_calls(engine, session_id=ids["session"], limit=2)
    assert [r.id for r in rows] == [seeded_calls["r3"].id, seeded_calls["r2"].id]


# ---------------------------------------------------------------- claim


def test_claim_links_only_unlinked_and_second_claim_returns_nothing(engine, ids):
    ep1 = _insert_episode(engine, ids["session"], ids["user"])
    ep2 = _insert_episode(engine, ids["session"], ids["user"])
    a = record_tool_call(
        engine, session_id=ids["session"], user_id=ids["user"], tool_name="grep", status="ok"
    )
    b = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="build",
        status="error",
        error="boom",
    )
    bystander = record_tool_call(
        engine,
        session_id=ids["session_other"],
        user_id=ids["user_other"],
        tool_name="grep",
        status="ok",
    )

    claimed = claim_tool_calls(engine, session_id=ids["session"], episode_id=ep1)
    assert {r.id for r in claimed} == {a.id, b.id}
    assert all(r.episode_id == ep1 for r in claimed)
    assert all(r.task_id is None for r in claimed)  # no task given, none invented

    # Everything is linked now; a second claim finds nothing.
    assert claim_tool_calls(engine, session_id=ids["session"], episode_id=ep2) == []

    # A call recorded after the first claim attaches to the later episode only.
    c = record_tool_call(
        engine, session_id=ids["session"], user_id=ids["user"], tool_name="test", status="ok"
    )
    claimed2 = claim_tool_calls(engine, session_id=ids["session"], episode_id=ep2)
    assert {r.id for r in claimed2} == {c.id}

    rows = {r.id: r for r in recent_tool_calls(engine, session_id=ids["session"])}
    assert rows[a.id].episode_id == ep1  # earlier rows keep their first episode
    assert rows[b.id].episode_id == ep1
    assert rows[c.id].episode_id == ep2

    # Another session's call was never touched.
    other = recent_tool_calls(engine, session_id=ids["session_other"])
    assert [r.id for r in other] == [bystander.id]
    assert other[0].episode_id is None


def test_claim_backfills_task_id_only_where_null(engine, ids):
    ep = _insert_episode(engine, ids["session"], ids["user"])
    pinned = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="grep",
        status="ok",
        task_id=ids["task"],
    )
    floating = record_tool_call(
        engine, session_id=ids["session"], user_id=ids["user"], tool_name="build", status="ok"
    )

    claimed = {
        r.id: r
        for r in claim_tool_calls(
            engine, session_id=ids["session"], episode_id=ep, task_id=ids["task_other"]
        )
    }
    assert set(claimed) == {pinned.id, floating.id}
    assert claimed[pinned.id].task_id == ids["task"]  # pre-existing task survives
    assert claimed[floating.id].task_id == ids["task_other"]  # NULL backfilled


# ---------------------------------------------------------------- failed_calls


def test_failed_calls_scopes_by_task_when_given_else_user(engine, ids):
    record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="fine",
        status="ok",
        task_id=ids["task"],
    )
    e1 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="task_fail",
        status="error",
        error="boom-1",
        task_id=ids["task"],
    )
    e2 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="other_task_fail",
        status="error",
        error="boom-2",
        task_id=ids["task_other"],
    )
    e3 = record_tool_call(
        engine,
        session_id=ids["session"],
        user_id=ids["user"],
        tool_name="untasked_fail",
        status="error",
        error="boom-3",
    )

    by_user = failed_calls(engine, user_id=ids["user"])
    assert [r.id for r in by_user] == [e3.id, e2.id, e1.id]  # errors only, newest first

    by_task = failed_calls(engine, user_id=ids["user"], task_id=ids["task"])
    assert [r.id for r in by_task] == [e1.id]

    assert [r.id for r in failed_calls(engine, user_id=ids["user"], limit=2)] == [
        e3.id,
        e2.id,
    ]
