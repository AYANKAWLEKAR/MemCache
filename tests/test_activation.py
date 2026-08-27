"""Edge weights and activation spreading. Pure functions — hand-built graphs,
no database, no model. Every contract in the spec's §1–§2 lives here."""

from __future__ import annotations

import math

import pytest

from app.services.activation import (
    Edge,
    Neighborhood,
    edge_weight,
    spread_activation,
)

# ---------------------------------------------------------------- weights


def test_weight_scales_with_prior():
    assert edge_weight("ADVANCES", count=1) > edge_weight("RELATED_TO", count=1)


def test_weight_is_prior_times_log_ratio():
    # prior 0.5, count 1, cap 20 -> 0.5 * log(2)/log(21)
    expected = 0.5 * math.log(2) / math.log(21)
    assert edge_weight("RELATED_TO", count=1, cap=20) == pytest.approx(expected)


def test_weight_increases_with_count_but_sublinearly():
    w1 = edge_weight("RELATED_TO", count=1)
    w2 = edge_weight("RELATED_TO", count=2)
    w10 = edge_weight("RELATED_TO", count=10)
    assert w1 < w2 < w10
    # 2 -> 10 is a 5x count jump but must be far less than a 5x weight jump.
    assert (w10 / w2) < 3.0


def test_weight_saturates_at_cap():
    at_cap = edge_weight("RELATED_TO", count=20, cap=20)
    beyond = edge_weight("RELATED_TO", count=500, cap=20)
    assert at_cap == pytest.approx(0.5)  # exactly the prior at the cap
    assert beyond == pytest.approx(0.5)  # clamped, never exceeds prior


def test_weight_treats_missing_or_zero_count_as_one():
    """Legacy edges have no count; existence was one observation."""
    assert edge_weight("MENTIONS", count=None) == edge_weight("MENTIONS", count=1)
    assert edge_weight("MENTIONS", count=0) == edge_weight("MENTIONS", count=1)


def test_structural_edges_carry_full_prior_regardless_of_count():
    """INVOKED / HAS_EPISODE / etc. are facts, not observations: a tool call has
    exactly one INVOKED edge forever. Log-scaling a count that is always 1 would
    strangle every structural hop to ~23% strength (measured: a ToolCall two hops
    from a seed died at 0.027 under a 0.05 floor). Only co-occurrence-style edges
    earn weight through repetition."""
    for rel in ("INVOKED", "HAS_EPISODE", "PARTICIPATED_IN", "PURSUES", "ADVANCES", "HAS_ALIAS"):
        assert edge_weight(rel, count=1) == pytest.approx(edge_weight(rel, count=20)), rel
    # Counted edges still scale.
    assert edge_weight("RELATED_TO", count=1) < edge_weight("RELATED_TO", count=20)
    assert edge_weight("MENTIONS", count=1) < edge_weight("MENTIONS", count=20)


def test_unknown_edge_type_gets_conservative_prior():
    """A new edge type must not silently spread at full strength: its prior must
    sit at or below the weakest *known* prior."""
    from app.services.activation import EDGE_PRIORS

    assert edge_weight("SOMETHING_NEW", count=1) <= min(EDGE_PRIORS.values())


# ------------------------------------------------------------ spreading


def _graph(*edges: tuple[str, str, str, int]) -> Neighborhood:
    """(src, rel, dst, count) tuples -> Neighborhood."""
    return Neighborhood(edges=[Edge(s, r, d, c) for s, r, d, c in edges])


def test_empty_seeds_yield_nothing():
    g = _graph(("a", "MENTIONS", "b", 5))
    assert spread_activation(g, seeds={}, floor=0.1, decay=0.8).scores == {}


def test_seed_keeps_its_own_activation():
    g = _graph()
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.1, decay=0.8)
    assert out.scores["a"] == pytest.approx(1.0)


def test_strong_path_reaches_hop_three():
    """ADVANCES chain at high count carries activation three hops out."""
    g = _graph(
        ("a", "ADVANCES", "b", 20),
        ("b", "ADVANCES", "c", 20),
        ("c", "ADVANCES", "d", 20),
    )
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.1, decay=0.8)
    assert "d" in out.scores, f"strong path died early: {out}"
    assert out.scores["a"] > out.scores["b"] > out.scores["c"] > out.scores["d"]


def test_weak_path_dies_at_hop_one():
    """A single co-occurrence (weakest prior, count 1) should not carry far."""
    g = _graph(
        ("a", "RELATED_TO", "b", 1),
        ("b", "RELATED_TO", "c", 1),
    )
    # One weak hop lands at ~0.091 (0.5 prior * log2/log21 * 0.8); a second
    # weak hop lands at ~0.008. A floor between the two admits b and kills c.
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert "b" in out.scores
    assert "c" not in out.scores, f"weak path over-reached: {out}"


def test_floor_is_respected_exactly():
    g = _graph(("a", "MENTIONS", "b", 1))
    w = edge_weight("MENTIONS", count=1)
    just_above = w * 0.8 - 1e-6
    just_below = w * 0.8 + 1e-6
    assert "b" in spread_activation(g, seeds={"a": 1.0}, floor=just_above, decay=0.8).scores
    assert "b" not in spread_activation(g, seeds={"a": 1.0}, floor=just_below, decay=0.8).scores


def test_cycle_terminates_and_seed_wins():
    """a<->b<->c cycle: must halt, and nothing exceeds its best incoming path."""
    g = _graph(
        ("a", "MENTIONS", "b", 5),
        ("b", "MENTIONS", "c", 5),
        ("c", "MENTIONS", "a", 5),
    )
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.9)
    assert out.scores["a"] == pytest.approx(1.0)  # seed is never lowered by the cycle
    # b and c are symmetric one-hop neighbours of a in a 3-cycle, so they tie;
    # neither may reach the seed's own activation.
    assert out.scores["b"] < 1.0 and out.scores["c"] < 1.0
    assert out.scores["c"] == pytest.approx(out.scores["b"])


def test_max_wins_on_convergence():
    """Two paths into d: the stronger one sets d's activation."""
    g = _graph(
        ("a", "RELATED_TO", "d", 1),  # weak
        ("a", "ADVANCES", "d", 20),  # strong
    )
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert out.scores["d"] == pytest.approx(edge_weight("ADVANCES", count=20) * 0.8)


def test_edges_are_traversed_undirected():
    """Relations are evidence in both directions for spreading purposes."""
    g = _graph(("b", "MENTIONS", "a", 5))  # stored b->a, seed at a
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert "b" in out.scores


def test_no_hop_limit_only_weight_limits_depth():
    """Ten hops of saturated ADVANCES with gentle decay: activation reaches
    the end. Depth is a consequence of weight, never a rule."""
    chain = [(f"n{i}", "ADVANCES", f"n{i + 1}", 20) for i in range(10)]
    g = _graph(*chain)
    out = spread_activation(g, seeds={"n0": 1.0}, floor=0.01, decay=0.95)
    assert "n10" in out.scores


def test_result_is_sorted_by_activation_descending():
    g = _graph(("a", "ADVANCES", "b", 20), ("a", "RELATED_TO", "c", 1))
    out = spread_activation(g, seeds={"a": 1.0}, floor=0.01, decay=0.8)
    values = list(out.scores.values())
    assert values == sorted(values, reverse=True)


# ------------------------------------------------- recorded provenance


def test_spreading_records_the_parent_that_set_each_activation():
    """The winning parent is recorded during the max-wins update, not guessed later."""
    g = _graph(("a", "MENTIONS", "b", 5))
    result = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert result.scores["b"] == pytest.approx(edge_weight("MENTIONS", 5) * 0.8)
    assert result.parents["b"] == ("a", "MENTIONS", 5)
    assert "a" not in result.parents, "a seed has no parent"


def test_recorded_parent_is_exact_where_reconstruction_would_lie():
    """Minimal case found by randomized search (174 divergences in 4000 graphs).

    c is reached directly from the seed via DECIDED(1) at 0.64. The two-hop
    route a->b->c proposes *exactly* 0.64 as well, so it never wins the update.
    Reconstructing backwards sees the tie and picks b — reporting a two-hop path
    that never happened. Recording the parent at update time cannot lie.
    """
    g = _graph(
        ("c", "ADVANCES", "b", 20),
        ("a", "ADVANCES", "b", 20),
        ("c", "DECIDED", "a", 1),
    )
    result = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert result.scores["c"] == pytest.approx(0.64)
    assert result.parents["c"] == ("a", "DECIDED", 1), (
        "c was set directly by the seed; any other parent is a fabricated path"
    )


def test_parent_is_updated_when_a_stronger_route_takes_over():
    """If a later, stronger path raises a node, the recorded parent follows it."""
    g = _graph(
        ("a", "RELATED_TO", "d", 1),  # weak, reaches d first
        ("a", "ADVANCES", "d", 20),  # strong, must win and own the parent
    )
    result = spread_activation(g, seeds={"a": 1.0}, floor=0.05, decay=0.8)
    assert result.parents["d"] == ("a", "ADVANCES", 20)


def test_result_still_behaves_as_the_scores_mapping():
    """Ordering and membership contracts are unchanged for callers."""
    g = _graph(("a", "ADVANCES", "b", 20), ("a", "RELATED_TO", "c", 1))
    result = spread_activation(g, seeds={"a": 1.0}, floor=0.01, decay=0.8)
    values = list(result.scores.values())
    assert values == sorted(values, reverse=True)
    assert set(result.parents) <= set(result.scores)


def test_subgoal_of_is_a_structural_edge_at_full_prior():
    """SUBGOAL_OF joins the tree-structure class (PURSUES/ADVANCES): uncounted,
    0.9, so activation crosses from a Task to its parent at the same strength
    it crosses from an Episode to its Task."""
    from app.services.activation import COUNTED_EDGES, EDGE_PRIORS, edge_weight

    assert EDGE_PRIORS["SUBGOAL_OF"] == pytest.approx(0.9)
    assert "SUBGOAL_OF" not in COUNTED_EDGES
    assert edge_weight("SUBGOAL_OF", 1) == pytest.approx(0.9)
    assert edge_weight("SUBGOAL_OF", 50) == pytest.approx(0.9)


def test_hierarchy_config_defaults():
    from app.config import settings

    assert settings.task_max_depth == 8
    assert settings.task_placement_candidates == 3
    assert settings.task_placement_min_score == 0.0
    assert settings.proactive_task_node_seed == pytest.approx(0.2)
    assert settings.proactive_task_depth_decay == pytest.approx(0.7)
