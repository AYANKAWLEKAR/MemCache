"""Smoke test: one eval repetition end to end through the real pipeline.

Runs the failure-recall family (exercises L2 episodes and L4 tool calls) at
the minimum history size with one repetition, and asserts the results are
well-formed. Deliberately does NOT gate on coverage values: those are the
measurement, and model quality must not turn CI red. Needs the stack and both
Ollama models; skipped otherwise.
"""

from __future__ import annotations

import pytest

from evals.runner import preflight, run_matrix
from evals.scenarios import FAILURE_RECALL

pytestmark = pytest.mark.agentic


@pytest.fixture(scope="module")
def results():
    problems = preflight()
    if problems:
        pytest.skip("eval stack unavailable: " + "; ".join(problems))
    return run_matrix([FAILURE_RECALL], history_sizes=[10], repetitions=1)


def test_one_repetition_with_all_conditions(results):
    assert len(results) == 1
    rep = results[0]
    assert rep.error is None
    assert rep.seeded_sessions == 10
    assert [o.condition for o in rep.outcomes] == [
        "memcache", "full_transcript", "no_memory",
    ]


def test_outcomes_are_well_formed(results):
    for o in results[0].outcomes:
        assert o.error is None, (o.condition, o.error)
        assert 0.0 <= o.answer_coverage <= 1.0
        assert o.answer  # the agent said something under every condition


def test_contexts_have_the_expected_shape(results):
    by = {o.condition: o for o in results[0].outcomes}
    assert by["no_memory"].context_tokens == 0
    # 10 sessions of 8-12 turns dwarf the capped retrieval document.
    assert by["full_transcript"].context_tokens > by["memcache"].context_tokens > 0


def test_memcache_sources_span_tiers(results):
    # The seeded scenario writes L2 episodes and an L4 tool failure; the
    # retrieved provenance must reflect more than one tier or the eval is
    # not exercising the memory system it claims to measure.
    memcache = next(o for o in results[0].outcomes if o.condition == "memcache")
    assert len(memcache.tiers_used) >= 2, memcache.tiers_used
    assert memcache.retrieval_recall is not None
