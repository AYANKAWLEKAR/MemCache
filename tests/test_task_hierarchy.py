"""Placement adjudication: parsing, prompt, and structural shortlist.

Pure functions — no Ollama, no DB. The parser is the safety boundary between a
3B model's output and the tree: everything malformed or contradictory must
degrade to "no edge", never to an exception and never to a wrong edge.
"""

from __future__ import annotations

import pytest

from app.services.task_hierarchy import (
    PlacementVerdict,
    build_placement_prompt,
    parse_placement,
    placement_score,
    shortlist_candidates,
)
from app.services.task_store import PlacementCandidate

A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
VALID = {A, B}


# ------------------------------------------------------------- parsing


def test_parses_child_of():
    v = parse_placement(f'{{"relation": "child_of", "task_id": "{A}"}}', VALID)
    assert v == PlacementVerdict(relation="child_of", task_id=A)


def test_parses_parent_of_in_fences_and_prose():
    text = f'Sure:\n```json\n{{"relation": "parent_of", "task_id": "{B}"}}\n```\nDone.'
    v = parse_placement(text, VALID)
    assert v == PlacementVerdict(relation="parent_of", task_id=B)


def test_none_relation_is_no_verdict():
    assert parse_placement('{"relation": "none", "task_id": null}', VALID) is None


def test_none_with_an_id_is_a_contradiction_and_degrades():
    assert parse_placement(f'{{"relation": "none", "task_id": "{A}"}}', VALID) is None


def test_relation_without_id_degrades():
    assert parse_placement('{"relation": "child_of", "task_id": null}', VALID) is None


def test_hallucinated_id_degrades():
    assert parse_placement('{"relation": "child_of", "task_id": "nope"}', VALID) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think it is a subgoal.",
        '{"relation": "child_of"',
        f'{{"task_id": "{A}"}}',
        f'{{"relation": "sibling_of", "task_id": "{A}"}}',
        f'{{"relation": 3, "task_id": "{A}"}}',
        '{"relation": "child_of", "task_id": 42}',
        "[]",
    ],
)
def test_malformed_degrades_to_none(text):
    assert parse_placement(text, VALID) is None


# -------------------------------------------------------------- prompt


def test_prompt_lists_candidates_and_the_vocabulary():
    p = build_placement_prompt("Fix duplicate column", [(A, "Migrate schema"), (B, "Ship v2")])
    assert "Fix duplicate column" in p
    assert f"id: {A} | title: Migrate schema" in p
    assert f"id: {B} | title: Ship v2" in p
    for word in ("child_of", "parent_of", "none", "When unsure, answer none"):
        assert word in p


# ----------------------------------------------------------- shortlist


def _c(id_, title, *, ents=(), sess=(), root=True, updated="2026-08-19T00:00:00+00:00"):
    return PlacementCandidate(
        id=id_, title=title, is_root=root, updated_at=updated,
        entities=frozenset(ents), sessions=frozenset(sess),
    )


def _sim_table(table):
    def sim(a, b):
        return table.get((a, b), table.get((b, a), 0.0))
    return sim


def test_score_formula():
    subj = _c("s", "S", ents={"x", "y"}, sess={"s1"})
    cand = _c("c", "C", ents={"y", "z"}, sess={"s1", "s9"})
    sim = _sim_table({("S", "C"): 0.5})
    # 0.5*0.5 + 0.3*(1/3) + 0.2*1
    assert placement_score(subj, cand, sim) == pytest.approx(0.25 + 0.1 + 0.2)


def test_score_without_any_overlap_is_zero():
    subj = _c("s", "S", ents={"x"}, sess={"s1"})
    cand = _c("c", "C", ents={"z"}, sess={"s2"})
    assert placement_score(subj, cand, lambda a, b: 0.0) == 0.0


def test_shortlist_ranks_by_score_then_recency_and_caps():
    subj = _c("s", "S", ents={"x"}, sess={"s1"})
    hi = _c("hi", "HI", ents={"x"}, updated="2026-01-01T00:00:00+00:00")
    new = _c("new", "NEW", updated="2026-08-01T00:00:00+00:00")
    old = _c("old", "OLD", updated="2025-01-01T00:00:00+00:00")
    out = shortlist_candidates(subj, [old, new, hi], similarity=lambda a, b: 0.0, limit=2, min_score=0.0)
    assert [c.id for c in out] == ["hi", "new"]  # score, then newer first


def test_shortlist_min_score_cuts():
    subj = _c("s", "S", ents={"x"})
    weak = _c("w", "W")
    out = shortlist_candidates(subj, [weak], similarity=lambda a, b: 0.0, limit=3, min_score=0.05)
    assert out == []


def test_shortlist_excludes_subject_defensively_and_handles_empty():
    subj = _c("s", "S")
    assert shortlist_candidates(subj, [], similarity=lambda a, b: 1.0, limit=3, min_score=0.0) == []
    assert shortlist_candidates(subj, [subj], similarity=lambda a, b: 1.0, limit=3, min_score=0.0) == []
