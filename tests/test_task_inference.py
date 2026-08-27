"""Task adjudication parsing and prompt-building. Pure functions — no Ollama, no DB.

The parser is the safety boundary between a 3B model's output and the graph:
everything malformed must degrade to "no task attachment", never to an exception
and never to a spurious task.
"""

from __future__ import annotations

from app.services.task_inference import (
    TaskAdjudication,
    build_adjudication_prompt,
    parse_adjudication,
)

VALID_IDS = {"11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"}
ID_A = "11111111-1111-1111-1111-111111111111"


# ---------------------------------------------------------------- parsing


def test_parses_clean_json():
    text = f'{{"goal": "migrate telemetry to ClickHouse", "matches_task_id": "{ID_A}", "task_complete": false}}'
    result = parse_adjudication(text, VALID_IDS)
    assert result == TaskAdjudication(
        goal="migrate telemetry to ClickHouse",
        matches_task_id=ID_A,
        task_complete=False,
    )


def test_parses_json_wrapped_in_markdown_fences():
    text = '```json\n{"goal": "ship the release", "matches_task_id": null, "task_complete": false}\n```'
    result = parse_adjudication(text, VALID_IDS)
    assert result is not None
    assert result.goal == "ship the release"
    assert result.matches_task_id is None


def test_parses_json_embedded_in_prose():
    text = (
        'Sure! Based on the summary, here is my answer:\n'
        '{"goal": "fix the auth bug", "matches_task_id": null, "task_complete": true}\n'
        "Let me know if you need anything else."
    )
    result = parse_adjudication(text, VALID_IDS)
    assert result is not None
    assert result.goal == "fix the auth bug"
    assert result.task_complete is True


def test_null_goal_is_preserved():
    text = '{"goal": null, "matches_task_id": null, "task_complete": false}'
    result = parse_adjudication(text, VALID_IDS)
    assert result is not None
    assert result.goal is None


def test_malformed_json_degrades_to_none():
    assert parse_adjudication('{"goal": "x", "matches', VALID_IDS) is None


def test_non_json_prose_degrades_to_none():
    assert parse_adjudication("I could not determine a goal here.", VALID_IDS) is None


def test_empty_response_degrades_to_none():
    assert parse_adjudication("", VALID_IDS) is None


def test_missing_field_degrades_to_none():
    assert parse_adjudication('{"goal": "x", "task_complete": false}', VALID_IDS) is None


def test_wrong_types_degrade_to_none():
    assert parse_adjudication(
        '{"goal": 42, "matches_task_id": null, "task_complete": false}', VALID_IDS
    ) is None
    assert parse_adjudication(
        '{"goal": "x", "matches_task_id": null, "task_complete": "yes"}', VALID_IDS
    ) is None


def test_hallucinated_task_id_degrades_to_none():
    """A confident wrong id must NOT fall back to creating a new task —
    that would mint duplicates from the model's worst outputs."""
    text = '{"goal": "x", "matches_task_id": "99999999-9999-9999-9999-999999999999", "task_complete": false}'
    assert parse_adjudication(text, VALID_IDS) is None


def test_whitespace_goal_is_treated_as_null():
    text = '{"goal": "   ", "matches_task_id": null, "task_complete": false}'
    result = parse_adjudication(text, VALID_IDS)
    assert result is not None
    assert result.goal is None


# ---------------------------------------------------------------- prompt


def test_prompt_contains_candidates_and_summary():
    prompt = build_adjudication_prompt(
        "User discussed migrating telemetry.",
        [(ID_A, "Migrate telemetry to ClickHouse")],
    )
    assert ID_A in prompt
    assert "Migrate telemetry to ClickHouse" in prompt
    assert "User discussed migrating telemetry." in prompt


def test_prompt_handles_empty_candidate_list():
    prompt = build_adjudication_prompt("A summary.", [])
    assert "none" in prompt.lower()
