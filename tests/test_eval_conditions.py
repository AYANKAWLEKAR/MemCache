"""Unit tests for evals.conditions — pure over a hand-built run record."""

import pytest

from evals.conditions import (
    FullTranscriptCondition,
    MemcacheCondition,
    NoMemoryCondition,
    RunRecord,
)
from evals.scenarios import EvalProbe, EvalSession, PlacedSession


def _run_record() -> RunRecord:
    fact = EvalSession(
        label="the build breaks",
        fixed_messages=({"role": "user", "content": "the build failed"},),
        tool_failures=(
            {"tool_name": "dbt", "status": "error", "error": "CompilationError: boom"},
        ),
    )
    distractor = EvalSession(
        label="trip planning",
        fixed_messages=(
            {"role": "user", "content": "plan the trip"},
            {"role": "assistant", "content": "noted"},
        ),
    )
    return RunRecord(
        user_id="demo-ui-eval-x",
        placed=(
            PlacedSession(session=fact, fact_index=0),
            PlacedSession(session=distractor, fact_index=None),
        ),
        realized_messages=(
            ({"role": "user", "content": "the dbt build just failed"},),
            (
                {"role": "user", "content": "plan the trip"},
                {"role": "assistant", "content": "noted"},
            ),
        ),
        session_ids=("demo-ui-eval-x-s0", "demo-ui-eval-x-s1"),
    )


class TestNoMemory:
    def test_returns_no_context(self):
        built = NoMemoryCondition().build_context(_run_record(), EvalProbe("q?", ("f",)))
        assert built.context is None
        assert built.condition == "no_memory"


class TestFullTranscript:
    def test_chronological_sessions_with_markers(self):
        built = FullTranscriptCondition().build_context(
            _run_record(), EvalProbe("q?", ("f",))
        )
        text = built.context
        assert text.index("## Session 1: the build breaks") < text.index(
            "## Session 2: trip planning"
        )
        assert "User: the dbt build just failed" in text
        assert "Assistant: noted" in text

    def test_uses_realized_messages_not_fallbacks(self):
        # The transcript must be the exact ingested wording, so all
        # conditions share identical seeded content within a repetition.
        text = FullTranscriptCondition().build_context(
            _run_record(), EvalProbe("q?", ("f",))
        ).context
        assert "the dbt build just failed" in text
        assert "the build failed\n" not in text  # the unrealized fixture text

    def test_tool_failures_are_included(self):
        text = FullTranscriptCondition().build_context(
            _run_record(), EvalProbe("q?", ("f",))
        ).context
        assert "[tool dbt error: CompilationError: boom]" in text


class TestMemcache:
    def test_fresh_probe_session_and_cap(self):
        calls = []

        def fake_retrieve(session_id, user_id, query, max_tokens):
            calls.append((session_id, user_id, query, max_tokens))
            return {"context": "ctx", "sources": [{"tier": "L2"}]}

        built = MemcacheCondition(retrieve_fn=fake_retrieve).build_context(
            _run_record(), EvalProbe("what broke?", ("f",)), probe_index=2
        )
        assert calls == [("demo-ui-eval-x-probe-2", "demo-ui-eval-x", "what broke?", 1200)]
        assert built.context == "ctx"
        assert built.sources == ({"tier": "L2"},)
        assert built.warnings == ()

    def test_degraded_retrieval_is_flagged_not_hidden(self):
        def fake_retrieve(session_id, user_id, query, max_tokens):
            return {"context": "", "sources": [], "status": "degraded",
                    "warnings": ["postgres unavailable"]}

        built = MemcacheCondition(retrieve_fn=fake_retrieve).build_context(
            _run_record(), EvalProbe("q?", ("f",))
        )
        assert "retrieval status: degraded" in built.warnings
        assert "postgres unavailable" in built.warnings

    def test_continued_probe_uses_that_fact_sessions_id(self):
        def fake_retrieve(session_id, user_id, query, max_tokens):
            return {"context": "", "sources": []}

        cond = MemcacheCondition(retrieve_fn=fake_retrieve)
        probe = EvalProbe("anything else?", ("f",), probe_from_session=0)
        run = _run_record()
        calls = []
        cond.retrieve_fn = lambda *a: (calls.append(a), {"context": "", "sources": []})[1]
        cond.build_context(run, probe)
        assert calls[0][0] == "demo-ui-eval-x-s0"

    def test_unknown_fact_index_raises(self):
        run = _run_record()
        with pytest.raises(KeyError):
            run.fact_session_id(5)
