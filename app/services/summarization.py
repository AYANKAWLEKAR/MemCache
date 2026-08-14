"""Summarize conversations via Ollama HTTP API (local or compatible gateway)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def format_conversation_for_prompt(messages: list[dict[str, Any]]) -> str:
    """Turn API-style messages into a single prompt block."""
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_conversation_ollama(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    timeout_seconds: float = 120.0,
) -> str | None:
    """Call Ollama `/api/generate` with a summarization prompt.

    Returns ``None`` on HTTP/network errors or empty model output (caller skips L2/L3).
    Sends ``Authorization: Bearer`` when ``ollama_api_key`` is non-empty (dummy default is fine).
    """
    cfg = settings or default_settings
    text = format_conversation_for_prompt(messages)
    if not text.strip():
        logger.warning("summarization skipped: empty conversation")
        return None

    prompt = (
        "Summarize the following conversation in 2-4 concise sentences. "
        "Focus on facts, entities, decisions, and preferences. "
        "Do not add bullet points unless essential.\n\n"
        f"{text}"
    )

    url = cfg.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if (cfg.ollama_api_key or "").strip():
        headers["Authorization"] = f"Bearer {cfg.ollama_api_key.strip()}"

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.warning("Ollama summarization failed: %s", e)
        return None

    summary = (data.get("response") or "").strip()
    if not summary:
        logger.warning("Ollama returned empty response")
        return None
    return summary
