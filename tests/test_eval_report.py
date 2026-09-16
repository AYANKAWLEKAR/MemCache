"""Unit tests for evals.report — pure rendering over hand-built results."""

from evals.report import render_markdown
from evals.runner import ProbeOutcome, RepetitionResult


def _outcome(condition: str, coverage: float, tokens: int, **kw) -> ProbeOutcome:
    return ProbeOutcome(
        condition=condition,
        question="q?",
        answer="a",
        answer_coverage=coverage,
        missing_facts=(),
        context_tokens=tokens,
        latency_seconds=1.0,
        **kw,
    )


def _rep(rep: int, outcomes, scenario="failure_recall", size=10, error=None):
    return RepetitionResult(
        scenario=scenario,
        history_size=size,
        repetition=rep,
        seeded_sessions=size,
        outcomes=tuple(outcomes),
        error=error,
    )


def test_renders_per_group_table_and_summary():
    results = [
        _rep(0, [
            _outcome("memcache", 1.0, 800, retrieval_recall=1.0, tiers_used=("L2", "L4")),
            _outcome("full_transcript", 1.0, 3200),
            _outcome("no_memory", 0.0, 0),
        ]),
        _rep(1, [
            _outcome("memcache", 0.5, 900, retrieval_recall=1.0),
            _outcome("full_transcript", 1.0, 3300),
            _outcome("no_memory", 0.0, 0),
        ]),
    ]
    text = render_markdown(results, "Test report")
    assert "# Test report" in text
    assert "## failure_recall, 10 sessions of history" in text
    assert "| memcache | 75% (50%..100%) | 100% |" in text
    assert "| no_memory | 0% (0%..0%) | n/a | 0 |" in text
    assert "## Summary across scenarios" in text
    # memcache row precedes the baselines in every table
    assert text.index("| memcache |") < text.index("| full_transcript |")


def test_failed_repetition_and_outcome_reach_the_footer():
    results = [
        _rep(0, [], error="OllamaUnavailable: dead"),
        _rep(1, [
            _outcome("memcache", 1.0, 800),
            ProbeOutcome(
                condition="full_transcript",
                question="q?",
                answer="",
                answer_coverage=0.0,
                missing_facts=("f",),
                context_tokens=0,
                latency_seconds=0.0,
                error="ReadTimeout: slow",
            ),
        ]),
    ]
    text = render_markdown(results, "Test report")
    assert "## Failed measurements" in text
    assert "failure_recall h10 r0: OllamaUnavailable: dead" in text
    assert "[full_transcript]: ReadTimeout: slow" in text
    # the failed outcome is excluded from aggregation, and the runs column shows it
    assert "| full_transcript | n/a | n/a | n/a | n/a | 0/1 |" in text


def test_empty_results_render_without_tables():
    text = render_markdown([], "Empty")
    assert "# Empty" in text
    assert "## Summary" not in text
