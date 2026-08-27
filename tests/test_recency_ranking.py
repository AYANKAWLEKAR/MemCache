"""Recency decay and reranking. Pure functions — fixed timestamps, no ML, no DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.retrieval import recency_decay, rerank_by_recency

NOW = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)


def _ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


def test_decay_is_one_at_zero_age():
    """A just-written episode is not penalised at all."""
    assert recency_decay(_ago(0), now=NOW, half_life_days=30) == pytest.approx(1.0)


def test_decay_is_one_half_at_exactly_one_half_life():
    assert recency_decay(_ago(30), now=NOW, half_life_days=30) == pytest.approx(0.5)


def test_decay_is_one_quarter_at_two_half_lives():
    assert recency_decay(_ago(60), now=NOW, half_life_days=30) == pytest.approx(0.25)


def test_decay_decreases_monotonically_with_age():
    values = [recency_decay(_ago(d), now=NOW, half_life_days=30) for d in (0, 1, 7, 30, 90, 365)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 < v <= 1.0 for v in values)


def test_future_timestamp_is_not_boosted_above_one():
    """Clock skew must not let an episode score above its raw similarity."""
    future = NOW + timedelta(days=5)
    assert recency_decay(future, now=NOW, half_life_days=30) == pytest.approx(1.0)


def test_naive_timestamp_is_treated_as_utc():
    """Postgres can hand back naive datetimes; they must not crash the comparison."""
    naive = NOW.replace(tzinfo=None) - timedelta(days=30)
    assert recency_decay(naive, now=NOW, half_life_days=30) == pytest.approx(0.5)


class _Hit:
    """Minimal stand-in carrying the fields rerank needs."""

    def __init__(self, ident: int, similarity: float, age_days: float):
        self.id = ident
        self.distance = 1.0 - similarity
        self.end_time = _ago(age_days)


def test_recent_moderate_match_outranks_ancient_strong_match():
    """The whole point: working memory favours recency over a stale near-duplicate."""
    ancient_strong = _Hit(1, similarity=0.90, age_days=365)
    recent_moderate = _Hit(2, similarity=0.55, age_days=0)

    ranked = rerank_by_recency(
        [ancient_strong, recent_moderate], now=NOW, half_life_days=30, limit=2
    )

    assert [h.id for h, _ in ranked] == [2, 1]


def test_ancient_match_still_appears_when_strong_enough():
    """Decay reorders; it never removes."""
    ancient = _Hit(1, similarity=0.90, age_days=365)
    ranked = rerank_by_recency([ancient], now=NOW, half_life_days=30, limit=5)
    assert [h.id for h, _ in ranked] == [1]


def test_rerank_truncates_to_limit():
    hits = [_Hit(i, similarity=0.5, age_days=i) for i in range(10)]
    ranked = rerank_by_recency(hits, now=NOW, half_life_days=30, limit=3)
    assert len(ranked) == 3
    # Youngest three win when raw similarity is identical.
    assert [h.id for h, _ in ranked] == [0, 1, 2]


def test_rerank_returns_the_decayed_score_alongside_each_hit():
    hit = _Hit(1, similarity=0.8, age_days=30)
    [(returned, score)] = rerank_by_recency([hit], now=NOW, half_life_days=30, limit=1)
    assert returned is hit
    assert score == pytest.approx(0.4)  # 0.8 similarity * 0.5 decay


def test_rerank_of_empty_candidate_set_is_empty():
    assert rerank_by_recency([], now=NOW, half_life_days=30, limit=5) == []
