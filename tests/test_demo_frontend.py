"""Demo-frontend logic: registry sanity, source-table building, think-stripping.

Pure — no Streamlit, no stores, no model. The one integration round-trip
(seed → probe → retrieve → reset) lives at the bottom under the integration
marker.
"""

from __future__ import annotations

import pytest

from frontend.demo_runtime import build_source_rows, count_kinds, strip_think
from frontend.demos import DEMOS


# ------------------------------------------------------------- registry


def test_registry_has_four_well_formed_demos():
    assert len(DEMOS) == 4
    keys = [d.key for d in DEMOS]
    assert len(set(keys)) == 4
    for d in DEMOS:
        assert d.user_id == f"demo-ui-{d.key}"
        assert d.sessions, d.key
        assert d.retrieval_query and d.agent_question and d.blurb
        for i, s in enumerate(d.sessions):
            assert d.session_id(i).startswith(d.user_id)
            assert s.messages, f"{d.key} session {i} empty"
            for m in s.messages:
                assert m["role"] in {"user", "assistant"} and m["content"].strip()
            for tf in s.tool_failures:
                assert tf["tool_name"] and tf["error"]
    assert sum(1 for d in DEMOS if d.plant_hierarchy) == 1


def test_every_demo_seeds_a_failure_or_a_fact_the_question_needs():
    """Each demo's script must contain the anchor its side-by-side hinges on."""
    text = {d.key: " ".join(m["content"].lower() for s in d.sessions for m in s.messages)
            for d in DEMOS}
    assert "alembic" in text["failure-recall"]
    assert "telemetry v2" in text["goal-hierarchy"]
    assert "dana whitfield" in text["identity-preferences"]
    assert "clickhouse" in text["passing-mention"]


# ---------------------------------------------------------- strip_think


def test_strip_think_removes_closed_and_unclosed_blocks():
    assert strip_think("<think>hmm</think>Answer.") == "Answer."
    assert strip_think("A<think>x</think>B<think>y</think>C") == "ABC"
    assert strip_think("<think>never closed... Answer buried") == ""
    assert strip_think("no think here") == "no think here"
    assert strip_think("") == ""


# ---------------------------------------------------- build_source_rows


FIXTURE_SOURCES = [
    {"type": "recent_message", "tier": "L1", "details": {"session_id": "s", "index": 0}},
    {"type": "profile_identity", "tier": "L3", "details": {"user_id": "u", "display_name": "Dana"}},
    {"type": "task", "tier": "L3", "details": {"task_id": "T-LEAF", "title": "Fix column",
     "status": "open", "lineage": ["T-LEAF", "T-MID", "T-ROOT"], "depth": 2}},
    {"type": "episode", "tier": "L2", "details": {"episode_id": 41, "session_id": "old",
     "similarity": 0.61, "decayed_score": 0.55}},
    {"type": "tool_failure", "tier": "L4", "details": {"tool_call_id": 9, "tool_name": "alembic",
     "task_id": "T-ROOT", "error_head": "DuplicateColumn: boom"}},
    {"type": "proactive_episode", "tier": "L3", "details": {"episode_id": 42, "session_id": "old2",
     "activation": 0.144, "via": "Task:T-ROOT -ADVANCES(1)-> 42",
     "path": [["Task:T-ROOT", "ADVANCES", 1, "Episode:42"]], "is_seed": False}},
    {"type": "proactive_entity", "tier": "L3", "details": {"name": "clickhouse",
     "activation": 0.4, "via": "alembic -RELATED_TO(6)-> clickhouse", "path": [], "is_seed": False}},
    {"type": "proactive_task", "tier": "L3", "details": {"task_id": "T-OTHER",
     "activation": 0.09, "via": "x", "path": [], "is_seed": False}},
    {"type": "proactive_tool_failure", "tier": "L4", "details": {"tool_call_id": 10,
     "tool_name": "alembic", "status": "error", "activation": 0.06, "via": "p",
     "path": [], "is_seed": False}},
    {"type": "decision", "tier": "L3", "details": {"text": "use Rust"}},
]


def test_build_source_rows_extracts_ids_scores_and_paths():
    rows = build_source_rows(FIXTURE_SOURCES)
    assert len(rows) == len(FIXTURE_SOURCES)
    by_id = {r["ID"]: r for r in rows}

    assert by_id["episode 41"]["Tier"] == "L2"
    assert by_id["episode 41"]["Score"] == 0.55          # decayed beats raw
    assert by_id["episode 42"]["Score"] == 0.144         # activation
    assert by_id["episode 42"]["Path"] == "Task:T-ROOT -ADVANCES(1)-> 42"

    goal = by_id["goal T-LEAF"]
    assert "Fix column" in goal["Detail"] and "open" in goal["Detail"]
    assert "T-LEAF ▸ T-MID ▸ T-ROOT" in goal["Detail"]   # lineage surfaces

    assert by_id["entity clickhouse"]["Score"] == 0.4
    assert "DuplicateColumn" in by_id["tool_call 9"]["Detail"]
    assert by_id["tool_call 10"]["Tier"] == "L4"
    assert by_id["goal T-OTHER"]["ID"] == "goal T-OTHER"
    # Rows without a natural id still render.
    assert any(r["Type"] == "recent_message" for r in rows)
    assert any("use Rust" in r["Detail"] for r in rows)


def test_count_kinds_totals_by_id_kind():
    rows = build_source_rows(FIXTURE_SOURCES)
    counts = count_kinds(rows)
    assert counts == {"episodes": 2, "entities": 1, "goals": 2, "tool_calls": 2}


# ------------------------------------------------- integration round-trip


@pytest.mark.integration
def test_seed_retrieve_reset_round_trip_failure_recall():
    """The cheapest demo (one session) through the REAL pipeline: seed writes
    all tiers, retrieve surfaces the failure with provenance ids, reset wipes.
    Asserts via the runtime's own API — the tiers themselves are covered by
    the main suite; this is the frontend's contract."""
    from frontend.demo_runtime import bootstrap, is_seeded, reset, retrieve, seed
    from frontend.demos import FAILURE_RECALL as demo

    bootstrap()
    reset(demo)
    assert not is_seeded(demo)

    calls: list[str] = []
    seed(demo, progress_cb=lambda i, n, label: calls.append(label))
    assert calls, "progress callback never fired"
    assert is_seeded(demo)

    result = retrieve(demo)
    assert "duplicatecolumn" in result["context"].lower()
    from frontend.demo_runtime import build_source_rows, count_kinds
    counts = count_kinds(build_source_rows(result["sources"]))
    assert counts["tool_calls"] >= 1, counts
    assert counts["episodes"] >= 1, counts

    reset(demo)
    assert not is_seeded(demo)
