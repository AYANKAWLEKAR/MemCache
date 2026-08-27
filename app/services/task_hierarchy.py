"""Goal-hierarchy placement: shortlist by graph evidence, ask one question,
parse defensively.

A separate call from task adjudication, on purpose. Adjudication already asks a
3B model for goal extraction, same-or-new, and completion against up to twenty
candidates; bolting "and is it a subgoal of one of them" onto that call would
overload exactly the judgement this tier depends on. Here the model sees at
most three candidates and answers one question.

The parser is the safety boundary. A hallucinated id, a `none` that names an
id, a relation with no id, an unknown relation — all degrade to ``None`` ("no
edge"). A wrong SUBGOAL_OF edge injects one goal's failures into another goal's
context, so every ambiguous case resolves to nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx

from app.config import Settings
from app.config import settings as default_settings
from app.services.task_store import PlacementCandidate

logger = logging.getLogger(__name__)

Relation = Literal["child_of", "parent_of"]


@dataclass(frozen=True)
class PlacementVerdict:
    """A parsed, non-null placement: the subject is `relation` the named task."""

    relation: Relation
    task_id: str


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_RELATIONS = {"child_of", "parent_of", "none"}


def parse_placement(text: str, valid_task_ids: set[str]) -> PlacementVerdict | None:
    """Parse a model response into a verdict, or ``None`` when unusable or `none`."""
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not {"relation", "task_id"} <= set(data):
        return None
    relation, task_id = data["relation"], data["task_id"]
    if not isinstance(relation, str) or relation not in _RELATIONS:
        return None
    if task_id is not None and not isinstance(task_id, str):
        return None
    if relation == "none":
        if task_id is not None:
            logger.warning("placement said none but named %r; discarding", task_id)
        return None
    if task_id is None:
        return None
    if task_id not in valid_task_ids:
        logger.warning("placement referenced unknown task id %r; discarding", task_id)
        return None
    return PlacementVerdict(relation=relation, task_id=task_id)  # type: ignore[arg-type]


def build_placement_prompt(subject_title: str, candidates: list[tuple[str, str]]) -> str:
    """One question, fixed vocabulary, ≤3 candidates (capped by the caller)."""
    lines = "\n".join(f"- id: {tid} | title: {title}" for tid, title in candidates)
    return (
        "You organize a user's goals into a tree of goals and subgoals.\n"
        "Respond with ONLY a JSON object, no prose, no code fences.\n\n"
        f"New goal:\n{subject_title}\n\n"
        f"Existing goals:\n{lines}\n\n"
        "Decide how the new goal relates to ONE of the existing goals:\n"
        "child_of  = the new goal is a smaller step toward that existing goal.\n"
        "parent_of = that existing goal is a smaller step toward the new goal.\n"
        "none      = unrelated, siblings under some larger goal, or the same goal.\n"
        "When unsure, answer none.\n\n"
        "Respond with exactly:\n"
        '{"relation": <"child_of" | "parent_of" | "none">, '
        '"task_id": <the id of the existing goal the relation is with, or null>}'
    )


# ---------------------------------------------------------------- shortlist


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def placement_score(
    subject: PlacementCandidate,
    cand: PlacementCandidate,
    similarity: Callable[[str, str], float],
) -> float:
    """`0.5·title_similarity + 0.3·entity_jaccard + 0.2·[shares a session]`.

    Every term is symmetric — it says "these goals share territory," never
    which is larger. Direction is the model's job. Recency is deliberately not
    a term: it is not evidence of relatedness, and it would push every recent
    task past any min-score cut.
    """
    sim = max(0.0, min(1.0, float(similarity(subject.title, cand.title))))
    return (
        0.5 * sim
        + 0.3 * _jaccard(subject.entities, cand.entities)
        + 0.2 * (1.0 if subject.sessions & cand.sessions else 0.0)
    )


def shortlist_candidates(
    subject: PlacementCandidate,
    candidates: list[PlacementCandidate],
    *,
    similarity: Callable[[str, str], float],
    limit: int,
    min_score: float,
) -> list[PlacementCandidate]:
    """Top-`limit` candidates by score (recency breaks ties), above `min_score`."""
    scored = [
        (placement_score(subject, c, similarity), c.updated_at, c)
        for c in candidates
        if c.id != subject.id
    ]
    scored = [t for t in scored if t[0] >= min_score]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored[: max(0, int(limit))]]


# ------------------------------------------------------------------ Ollama


def adjudicate_placement(
    subject_title: str,
    candidates: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    timeout_seconds: float = 60.0,
) -> PlacementVerdict | None:
    """Ask the model; parse; ``None`` on any failure. Same shape as
    `task_inference.adjudicate_task`."""
    cfg = settings or default_settings
    if not subject_title.strip() or not candidates:
        return None
    prompt = build_placement_prompt(subject_title, candidates)
    url = cfg.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
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
        logger.warning("placement adjudication call failed: %s", exc)
        return None
    return parse_placement(text, {tid for tid, _ in candidates})
