"""L2 episode search across a user's sessions. Live Postgres, synthetic vectors."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.postgres_store import PostgresStore
from tests.conftest import unit_embedding_384

pytestmark = pytest.mark.integration


@pytest.fixture
def engine():
    eng = create_engine_from_settings()
    ensure_l2_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(engine):
    """One user with two sessions, plus a second user who must stay invisible."""
    user = f"u-{uuid.uuid4().hex[:10]}"
    other_user = f"o-{uuid.uuid4().hex[:10]}"
    session_a = f"{user}-a"
    session_b = f"{user}-b"
    session_other = f"{other_user}-x"
    now = datetime.now(timezone.utc)

    with session_scope(engine) as s:
        store = PostgresStore(s)
        store.insert_episode(
            session_id=session_a,
            summary="Chose Rust for motion control.",
            embedding=unit_embedding_384(primary_axis=0),
            start_time=now - timedelta(days=10),
            end_time=now - timedelta(days=10),
            metadata=None,
            user_id=user,
        )
        # Same session as the query, but ingested before user_id existed.
        store.insert_episode(
            session_id=session_b,
            summary="Unattributed legacy episode about Rust.",
            embedding=unit_embedding_384(primary_axis=0),
            start_time=now,
            end_time=now,
            metadata=None,
            user_id=None,
        )
        store.insert_episode(
            session_id=session_other,
            summary="Someone else also chose Rust.",
            embedding=unit_embedding_384(primary_axis=0),
            start_time=now,
            end_time=now,
            metadata=None,
            user_id=other_user,
        )

    yield {"user": user, "other_user": other_user, "a": session_a, "b": session_b}

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DELETE FROM episodes WHERE session_id IN (%s, %s, %s)",
            (session_a, session_b, session_other),
        )


def _search(engine, session_id, user_id=None, limit=10):
    with session_scope(engine) as s:
        return PostgresStore(s).search_episodes(
            unit_embedding_384(primary_axis=0), session_id, user_id=user_id, limit=limit
        )


def test_without_user_id_search_stays_session_locked(engine, seeded):
    """Backward compatibility: omitting user_id must behave exactly as before."""
    hits = _search(engine, seeded["b"])
    assert {h.session_id for h in hits} == {seeded["b"]}


def test_with_user_id_search_spans_sessions(engine, seeded):
    """The point: session B retrieves session A's episode."""
    hits = _search(engine, seeded["b"], user_id=seeded["user"])
    assert seeded["a"] in {h.session_id for h in hits}


def test_current_session_stays_visible_when_its_rows_have_no_owner(engine, seeded):
    """Enabling the feature mid-history must not blind the live conversation."""
    hits = _search(engine, seeded["b"], user_id=seeded["user"])
    assert seeded["b"] in {h.session_id for h in hits}


def test_another_users_episodes_are_never_returned(engine, seeded):
    hits = _search(engine, seeded["b"], user_id=seeded["user"])
    summaries = {h.summary for h in hits}
    assert "Someone else also chose Rust." not in summaries


def test_search_returns_the_owner_on_each_hit(engine, seeded):
    """Callers need provenance, and it proves the column round-trips."""
    hits = _search(engine, seeded["b"], user_id=seeded["user"])
    owners = {h.user_id for h in hits}
    assert seeded["user"] in owners
    assert None in owners  # the unattributed legacy row


def test_limit_is_respected(engine, seeded):
    assert len(_search(engine, seeded["b"], user_id=seeded["user"], limit=1)) == 1
