"""Task/goal adjudication via Ollama, with a defensive parsing boundary.

A separate call from summarization, on purpose: asking a 3B model for prose
*and* strict JSON in one response risks a corrupted summary — damaging L2 to
save a round trip the async worker doesn't need to save.

The parser is the safety boundary. Anything malformed — fenced JSON, prose
wrapping, missing fields, wrong types, a hallucinated task id — degrades to
``None`` ("no task attachment"), never to an exception and never to a spurious
new task. An ingest must never fail because task inference failed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskAdjudication:
    """One parsed adjudication verdict."""

    goal: str | None
    matches_task_id: str | None
    task_complete: bool


#: Finds the first JSON object in a response, tolerating fences and prose.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_adjudication(text: str, valid_task_ids: set[str]) -> TaskAdjudication | None:
    """Parse a model response into a verdict, or ``None`` when unusable.

    A hallucinated ``matches_task_id`` invalidates the whole verdict rather than
    falling back to "new task" — minting duplicates from the model's worst
    outputs would erode the precision this tier exists to provide.
    """
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not {"goal", "matches_task_id", "task_complete"} <= set(data):
        return None

    goal = data["goal"]
    matches = data["matches_task_id"]
    complete = data["task_complete"]

    if goal is not None and not isinstance(goal, str):
        return None
    if matches is not None and not isinstance(matches, str):
        return None
    if not isinstance(complete, bool):
        return None
    if matches is not None and matches not in valid_task_ids:
        logger.warning("adjudication referenced unknown task id %r; discarding", matches)
        return None

    goal = goal.strip() if goal else None
    return TaskAdjudication(
        goal=goal or None,
        matches_task_id=matches,
        task_complete=complete,
    )


def build_adjudication_prompt(
    summary: str,
    open_tasks: list[tuple[str, str]],
) -> str:
    """Prompt for goal extraction + same-or-new adjudication + completion.

    ``open_tasks`` is (id, title) pairs, already capped by the caller — the cap
    is enforced in code because an unbounded list outgrows a small model's
    usable context and the judgement quality collapses.
    """
    if open_tasks:
        lines = "\n".join(f"- id: {tid} | title: {title}" for tid, title in open_tasks)
    else:
        lines = "(none)"
    return (
        "You maintain a task list for a user based on their conversations.\n"
        "Respond with ONLY a JSON object, no prose, no code fences.\n\n"
        f"The user's open tasks:\n{lines}\n\n"
        f"Conversation summary:\n{summary}\n\n"
        "Respond with exactly:\n"
        '{"goal": <short imperative phrase for the work objective, or null if none is stated>, '
        '"matches_task_id": <the id of the open task this conversation continues, or null>, '
        '"task_complete": <true only if the conversation clearly states the goal is finished, else false>}'
    )


def adjudicate_task(
    summary: str,
    open_tasks: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    timeout_seconds: float = 60.0,
) -> TaskAdjudication | None:
    """Call Ollama and parse the verdict. Returns ``None`` on any failure."""
    cfg = settings or default_settings
    if not summary.strip():
        return None

    prompt = build_adjudication_prompt(summary, open_tasks)
    url = cfg.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        # Deterministic-ish output: adjudication is a judgement call we want
        # repeatable, not creative.
        "options": {"temperature": 0.0},
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if (cfg.ollama_api_key or "").strip():
        headers["Authorization"] = f"Bearer {cfg.ollama_api_key.strip()}"

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("task adjudication call failed: %s", exc)
        return None

    return parse_adjudication(text, {tid for tid, _ in open_tasks})
