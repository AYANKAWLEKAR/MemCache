"""Runtime for the demo frontend: every side effect, plus the pure helpers.

Deliberately Streamlit-free — `demo_app.py` owns rendering; this module owns
doing. The pure helpers (`strip_think`, `build_source_rows`, `count_kinds`)
are unit-tested without any store or model.
"""

from __future__ import annotations

import re
from typing import Any

#: <think>…</think> (qwen3 thinking traces), including an unclosed trailing block.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove qwen3 thinking blocks; an unclosed block swallows to the end."""
    return _THINK.sub("", text or "").strip()


def _score(details: dict) -> float | None:
    for key in ("activation", "decayed_score", "similarity"):
        if details.get(key) is not None:
            return details[key]
    return None


def build_source_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provenance sources → table rows: Tier | Type | ID | Detail | Score | Path.

    The ID column is the point of the table: `episode <L2 row id>`,
    `entity <normalized name>` (Entity nodes are keyed by name — that IS the
    entity id in this graph), `goal <task uuid>`, `tool_call <L4 row id>`.
    """
    rows: list[dict[str, Any]] = []
    for src in sources:
        d = src.get("details", {}) or {}
        stype = src.get("type", "?")
        id_, detail = "—", ""
        if "episode_id" in d:
            id_ = f"episode {d['episode_id']}"
            detail = f"session {d.get('session_id', '?')}"
        elif "task_id" in d and stype in {"task", "proactive_task"}:
            id_ = f"goal {d['task_id']}"
            detail = f"{d.get('title', '')} ({d.get('status', '?')})".strip()
            if d.get("lineage"):
                detail += f" — lineage: {' ▸ '.join(d['lineage'])}"
        elif "tool_call_id" in d:
            id_ = f"tool_call {d['tool_call_id']}"
            detail = f"{d.get('tool_name', '?')}: {d.get('error_head') or d.get('status', '')}"
        elif "name" in d:
            id_ = f"entity {d['name']}"
            detail = d.get("display_name") or ""
        elif "text" in d:
            detail = d["text"]
        elif "display_name" in d or "user_id" in d:
            id_ = f"profile {d.get('user_id', '?')}"
            detail = d.get("display_name") or d.get("key", "")
        else:
            detail = ", ".join(f"{k}={v}" for k, v in d.items() if k not in {"path", "via"})
        rows.append({
            "Tier": src.get("tier", "?"),
            "Type": stype,
            "ID": id_,
            "Detail": detail,
            "Score": _score(d),
            "Path": d.get("via", "") or "",
        })
    return rows


def count_kinds(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Caption totals: how many episode/entity/goal/tool-call ids were retrieved."""
    prefixes = {"episodes": "episode ", "entities": "entity ",
                "goals": "goal ", "tool_calls": "tool_call "}
    return {
        kind: sum(1 for r in rows if str(r["ID"]).startswith(prefix))
        for kind, prefix in prefixes.items()
    }
