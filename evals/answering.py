"""The answering call, owned by the eval so fairness is enforced in one place.

Every condition gets the IDENTICAL prompt template; only the contents of the
context block differ (the no_memory condition's block is "(none)"). The
context window is pinned explicitly: Ollama's default num_ctx varies by
install, and a silent prompt truncation would delete exactly the buried facts
this eval measures, turning an environment default into a result.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

ANSWER_MODEL = os.environ.get("DEMO_AGENT_MODEL", "qwen3:4b")
ANSWER_TIMEOUT = float(os.environ.get("DEMO_AGENT_TIMEOUT", "180"))
#: Large enough to hold the biggest full_transcript condition (h=50) intact.
ANSWER_NUM_CTX = int(os.environ.get("EVAL_NUM_CTX", "32768"))


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    seconds: float


def ask(question: str, context: str | None) -> AgentAnswer:
    """One generation at temperature 0 under the shared template."""
    from app.config import settings
    from frontend.demo_runtime import strip_think

    prompt = (
        "Context from your memory system:\n"
        f"{context if context else '(none)'}\n\n"
        f"{question} /no_think"
    )
    start = time.monotonic()
    resp = httpx.post(
        settings.ollama_base_url.rstrip("/") + "/api/generate",
        json={
            "model": ANSWER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": ANSWER_NUM_CTX},
        },
        timeout=ANSWER_TIMEOUT,
    )
    resp.raise_for_status()
    text = strip_think((resp.json().get("response") or "").strip())
    return AgentAnswer(text=text, seconds=time.monotonic() - start)
