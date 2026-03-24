"""L2 PostgreSQL + pgvector: schema alignment, insert, cosine search, IVFFlat creation.

Requires docker-compose Postgres (pgvector). Deselect with: pytest -m "not integration"
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.config import settings
from app.db.postgres import (
    create_engine_from_settings,
    ensure_ivfflat_index,
    ensure_l2_schema,
    session_scope,
)
from app.services.postgres_store import PostgresStore
from tests.conftest import (
    SYNTHETIC_EPISODE_SUMMARIES,
    unit_embedding_384,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_engine():
    engine = create_engine_from_settings()
    yield engine
    engine.dispose()


@pytest.fixture
def clean_episodes_table(pg_engine):
    """Fresh `episodes` table state for each test (drop IVFFlat if present, truncate)."""
    ensure_l2_schema(pg_engine)
    with pg_engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_episodes_embedding_ivfflat"))
        conn.execute(text("TRUNCATE TABLE episodes RESTART IDENTITY"))
    yield
    with pg_engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_episodes_embedding_ivfflat"))
        conn.execute(text("TRUNCATE TABLE episodes RESTART IDENTITY"))


def test_episodes_table_and_session_index_exist(pg_engine):
    """Init script created `episodes` and B-tree on session_id."""
    ensure_l2_schema(pg_engine)
    with pg_engine.connect() as conn:
        tbl = conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'episodes'"
            )
        ).fetchone()
        assert tbl is not None
        idx = conn.execute(
            text(
                "SELECT 1 FROM pg_indexes "
                "WHERE tablename = 'episodes' AND indexname = 'idx_episodes_session_id'"
            )
        ).fetchone()
        assert idx is not None


def test_insert_episode_roundtrip(clean_episodes_table, pg_engine):
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc)
    emb = unit_embedding_384(primary_axis=0)
    meta = {"source": "synthetic", "turns": 2}

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        eid = store.insert_episode(
            session_id="sess_test_insert",
            summary="Synthetic summary for L2.",
            embedding=emb,
            start_time=t0,
            end_time=t1,
            metadata=meta,
        )
        assert eid >= 1

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        row = store.get_episode_by_id(eid)
        assert row is not None
        assert row.session_id == "sess_test_insert"
        assert row.summary == "Synthetic summary for L2."
        assert row.episode_metadata == meta
        assert len(row.embedding) == settings.embedding_dimension


def test_search_episodes_orders_by_cosine_distance(clean_episodes_table, pg_engine):
    """Nearest neighbor first: query aligned with axis 0."""
    sid = "sess_vec_order"
    t0 = datetime(2025, 3, 1, tzinfo=timezone.utc)
    t1 = datetime(2025, 3, 1, 0, 1, tzinfo=timezone.utc)

    # Far: energy on axis 100; Mid: blend axis 0 + 1; Near: almost axis 0
    emb_far = unit_embedding_384(primary_axis=100)
    emb_mid = unit_embedding_384(primary_axis=0, secondary_axis=1, secondary_weight=0.5)
    emb_near = unit_embedding_384(primary_axis=0, secondary_axis=1, secondary_weight=0.05)

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        store.insert_episode(sid, "far", emb_far, t0, t1, None)
        store.insert_episode(sid, "mid", emb_mid, t0, t1, None)
        store.insert_episode(sid, "near", emb_near, t0, t1, None)

    query = unit_embedding_384(primary_axis=0)
    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        hits = store.search_episodes(query, sid, limit=5)

    assert [h.summary for h in hits] == ["near", "mid", "far"]
    assert hits[0].distance <= hits[1].distance <= hits[2].distance


def test_search_episodes_respects_session_boundary(clean_episodes_table, pg_engine):
    """Vector search is restricted to the requested session_id (PRD)."""
    t0 = datetime(2025, 3, 2, tzinfo=timezone.utc)
    t1 = datetime(2025, 3, 2, 0, 1, tzinfo=timezone.utc)
    emb = unit_embedding_384(primary_axis=2)

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        store.insert_episode("sess_a", "only a", emb, t0, t1, None)
        store.insert_episode("sess_b", "only b", emb, t0, t1, None)

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        a_hits = store.search_episodes(emb, "sess_a", limit=5)
        b_hits = store.search_episodes(emb, "sess_b", limit=5)

    assert len(a_hits) == 1 and a_hits[0].summary == "only a"
    assert len(b_hits) == 1 and b_hits[0].summary == "only b"


def test_search_episodes_with_synthetic_summaries(clean_episodes_table, pg_engine):
    """Uses shared synthetic text fixtures (paired with distinct embeddings)."""
    sid = "session_alpha"
    summaries = SYNTHETIC_EPISODE_SUMMARIES[sid]
    t0 = datetime(2025, 3, 3, tzinfo=timezone.utc)
    t1 = datetime(2025, 3, 3, 0, 10, tzinfo=timezone.utc)
    axes = (10, 11, 200)

    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        for summary, axis in zip(summaries, axes, strict=True):
            store.insert_episode(
                sid,
                summary,
                unit_embedding_384(primary_axis=axis),
                t0,
                t1,
                {"synthetic": True},
            )

    query = unit_embedding_384(primary_axis=11)
    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        hits = store.search_episodes(query, sid, limit=2)

    assert hits[0].summary == summaries[1]
    assert hits[0].episode_metadata == {"synthetic": True}


def test_ensure_ivfflat_index_after_bulk_insert(clean_episodes_table, pg_engine):
    """After enough rows, IVFFlat can be built; search still returns ordered results."""
    t0 = datetime(2025, 3, 4, tzinfo=timezone.utc)
    t1 = datetime(2025, 3, 4, 0, 1, tzinfo=timezone.utc)
    n = 120
    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        for i in range(n):
            store.insert_episode(
                "bulk_sess",
                f"episode-{i}",
                unit_embedding_384(primary_axis=i % 384),
                t0,
                t1,
                None,
            )

    assert ensure_ivfflat_index(pg_engine, target_lists=100) is True

    with pg_engine.connect() as conn:
        idx = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'episodes' AND indexname = 'idx_episodes_embedding_ivfflat'"
            )
        ).scalar_one()
        assert "ivfflat" in idx.lower()
        assert "vector_cosine_ops" in idx

    q = unit_embedding_384(primary_axis=7)
    with session_scope(pg_engine) as session:
        store = PostgresStore(session)
        hits = store.search_episodes(q, "bulk_sess", limit=3)

    assert len(hits) == 3
    assert hits[0].distance <= hits[1].distance <= hits[2].distance
    assert all(h.summary.startswith("episode-") for h in hits)
