"""Unit tests for evals.scenarios — fairness rules and interleaving, no stores."""

import pytest

from evals.scenarios import (
    ALL_SCENARIOS,
    PASSING_MENTION,
    distractor_sessions,
    interleave,
)
from evals.scoring import normalize


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
class TestFairnessRules:
    def test_probe_question_contains_no_required_fact_or_anchor(self, scenario):
        anchors = [a for s in scenario.sessions for t in s.turns for a in t.anchors]
        for probe in scenario.probes:
            q = normalize(probe.question)
            for fact in probe.required_facts:
                assert normalize(fact) not in q, (probe.question, fact)
            for anchor in anchors:
                assert normalize(anchor) not in q, (probe.question, anchor)

    def test_required_facts_are_present_in_seeded_material(self, scenario):
        # Every required fact must be recoverable from what gets ingested:
        # the deterministic fallbacks, the tool failure payloads, or the
        # planted goal titles. Otherwise the probe is unanswerable by design.
        material = normalize(
            " ".join(
                [t.deterministic_text() for s in scenario.sessions for t in s.turns]
                + [tf["error"] for s in scenario.sessions for tf in s.tool_failures]
                + [title for title, _ in scenario.planted_goals]
            )
        )
        for fact in scenario.all_required_facts():
            assert normalize(fact) in material, fact

    def test_probe_from_session_indices_are_valid(self, scenario):
        for probe in scenario.probes:
            if probe.probe_from_session is not None:
                assert 0 <= probe.probe_from_session < len(scenario.sessions)

    def test_planted_goals_list_parents_before_children(self, scenario):
        for i, (_, parent) in enumerate(scenario.planted_goals):
            if parent is not None:
                assert 0 <= parent < i


class TestDistractors:
    def test_count_and_shape(self):
        sessions = distractor_sessions(50)
        assert len(sessions) == 50
        for s in sessions:
            assert 8 <= len(s.fixed_messages) <= 12
            assert not s.turns and not s.tool_failures

    def test_deterministic(self):
        a, b = distractor_sessions(30), distractor_sessions(30)
        assert [s.fixed_messages for s in a] == [s.fixed_messages for s in b]

    def test_varied_content_across_template_repeats(self):
        sessions = distractor_sessions(24)  # every template appears twice
        texts = {" ".join(m["content"] for m in s.fixed_messages) for s in sessions}
        assert len(texts) == 24  # fill rotation keeps repeats distinct

    def test_no_distractor_contains_any_required_fact(self):
        facts = [f for sc in ALL_SCENARIOS for f in sc.all_required_facts()]
        for s in distractor_sessions(50):
            text = normalize(" ".join(m["content"] for m in s.fixed_messages))
            for fact in facts:
                assert normalize(fact) not in text, (s.label, fact)


class TestInterleave:
    @pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
    @pytest.mark.parametrize("size", [10, 25, 50])
    def test_total_count_and_fact_order(self, scenario, size):
        placed = interleave(scenario, size)
        assert len(placed) == size
        fact_indices = [p.fact_index for p in placed if p.fact_index is not None]
        assert sorted(fact_indices) == list(range(len(scenario.sessions)))

    def test_non_continued_facts_land_in_earliest_third(self):
        for scenario in ALL_SCENARIOS:
            continued = {p.probe_from_session for p in scenario.probes} - {None}
            placed = interleave(scenario, 30)
            for pos, p in enumerate(placed):
                if p.fact_index is not None and p.fact_index not in continued:
                    assert pos < 10, (scenario.name, pos)

    def test_continued_session_is_last(self):
        placed = interleave(PASSING_MENTION, 25)
        assert placed[-1].fact_index == 1  # "today" session ends the history

    def test_history_smaller_than_fact_sessions_raises(self):
        with pytest.raises(ValueError):
            interleave(PASSING_MENTION, 1)
