"""Unit tests for evals.conditions — pure over a hand-built run record."""

import pytest

from evals.conditions import (
    FullTranscriptCondition,
    MemcacheCondition,
    NaiveRagCondition,
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


def _keyword_embedder(keyword: str):
    """Fake embedder: [1,0] when the text contains the keyword, else [0,1]."""

    def embed(texts):
        return [[1.0, 0.0] if keyword in t.lower() else [0.0, 1.0] for t in texts]

    return embed


class TestNaiveRag:
    def test_ranks_by_similarity_and_includes_tool_failures(self):
        cond = NaiveRagCondition(embed_fn=_keyword_embedder("dbt"))
        built = cond.build_context(_run_record(), EvalProbe("dbt again?", ("f",)))
        assert built.condition == "naive_rag"
        # Both dbt documents (the tool failure line and the failing turn)
        # score 1.0 against the query and are selected.
        assert "[tool dbt error: CompilationError: boom]" in built.context
        assert "the dbt build just failed" in built.context

    def test_respects_token_cap(self):
        cond = NaiveRagCondition(embed_fn=_keyword_embedder("dbt"), max_tokens=12)
        built = cond.build_context(_run_record(), EvalProbe("dbt again?", ("f",)))
        from evals.scoring import count_tokens

        assert 0 < count_tokens(built.context) <= 12

    def test_selected_documents_are_presented_chronologically(self):
        # Query matches the LAST document best; presentation must still be
        # corpus order, not similarity order.
        cond = NaiveRagCondition(embed_fn=_keyword_embedder("trip"))
        built = cond.build_context(_run_record(), EvalProbe("the trip?", ("f",)))
        lines = built.context.splitlines()
        assert lines == sorted(
            lines,
            key=lambda ln: 1 if ln.startswith("Session 2") else 0,
        )

    def test_no_sources_and_no_warnings(self):
        built = NaiveRagCondition(embed_fn=_keyword_embedder("x")).build_context(
            _run_record(), EvalProbe("q?", ("f",))
        )
        assert built.sources == ()
        assert built.warnings == ()
