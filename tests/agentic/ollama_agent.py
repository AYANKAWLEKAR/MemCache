"""A small Ollama-driven agent that produces realistic conversation traffic.

The agent exists to make the *input* to MemCache realistic — natural phrasing,
varied sentence structure, real assistant replies — while keeping the *assertions*
deterministic. That balance is the whole point of the design:

    the model chooses the words; the scenario chooses the facts.

Every generated turn is verified to contain its scenario-mandated anchor strings.
If the model drifts and omits one, the agent retries once and then falls back to a
deterministic template. So a flaky model degrades traffic realism, never test
correctness — a test can fail because MemCache lost a fact, never because
qwen2.5:3b had an off day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Small, fast, and locally available. Generation quality only affects phrasing.
DEFAULT_MODEL = "qwen2.5:3b"


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama daemon cannot be reached or lacks the model."""


def ollama_available(base_url: str, model: str, timeout: float = 5.0) -> bool:
    """True when the daemon answers and `model` is pulled."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return False
    # Ollama reports "qwen2.5:3b"; accept a bare family name too.
    return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)


def _generate(
    base_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"Ollama generate failed: {exc}") from exc


@dataclass
class Turn:
    """One planned exchange.

    `intent` steers phrasing; `anchors` are the substrings that MUST survive into
    the emitted text, because downstream assertions depend on them.
    """

    intent: str
    anchors: list[str] = field(default_factory=list)
    fallback: str = ""

    def satisfied_by(self, text: str) -> bool:
        low = text.lower()
        return all(a.lower() in low for a in self.anchors)

    def deterministic_text(self) -> str:
        """Template used when the model will not cooperate."""
        if self.fallback:
            return self.fallback
        return f"{self.intent} ({', '.join(self.anchors)})"


@dataclass
class OllamaAgent:
    """Generates user turns and assistant replies against a local Ollama model."""

    base_url: str = "http://localhost:11434"
    model: str = DEFAULT_MODEL
    timeout: float = 120.0
    temperature: float = 0.4

    #: Populated as the conversation proceeds; useful for debugging failures.
    transcript: list[dict[str, str]] = field(default_factory=list)

    def user_turn(self, turn: Turn) -> str:
        """Produce a natural user message that provably contains every anchor."""
        prompt = (
            "You are the HUMAN USER writing to your AI assistant.\n"
            "Write ONE short, natural first-person statement (1-2 sentences).\n"
            "Rules:\n"
            "- You are the user, never the assistant. Never offer help or ask "
            "'how can I assist you'.\n"
            "- Write a declarative statement, not a greeting or a question.\n"
            "- No quotation marks, labels, or preamble. Output only the message.\n\n"
            f"What you want to convey: {turn.intent}\n"
        )
        if turn.anchors:
            listed = ", ".join(f'"{a}"' for a in turn.anchors)
            prompt += (
                f"Your message MUST contain these exact strings verbatim: {listed}\n"
            )

        for attempt in range(2):
            text = _generate(
                self.base_url,
                self.model,
                prompt,
                timeout=self.timeout,
                temperature=self.temperature if attempt == 0 else 0.0,
            )
            text = _strip_wrapping(text)
            if text and turn.satisfied_by(text):
                self.transcript.append({"role": "user", "content": text})
                return text
            logger.debug("agent turn missed anchors (attempt %d): %r", attempt, text)

        text = turn.deterministic_text()
        logger.info("agent fell back to deterministic text for intent=%r", turn.intent)
        self.transcript.append({"role": "user", "content": text})
        return text

    def assistant_turn(self, user_message: str, memory_context: str = "") -> str:
        """Produce a plausible assistant reply, optionally grounded in memory."""
        context_block = memory_context.strip() or "(no memory context)"
        prompt = (
            "You are a helpful assistant. Reply in 1-2 short sentences.\n"
            "Acknowledge concrete details the user gave. Do not invent facts.\n"
            "Output only the reply.\n\n"
            f"Memory context:\n{context_block}\n\n"
            f"User: {user_message}\n"
            "Assistant:"
        )
        text = _strip_wrapping(
            _generate(
                self.base_url,
                self.model,
                prompt,
                timeout=self.timeout,
                temperature=self.temperature,
            )
        )
        if not text:
            text = "Understood."
        self.transcript.append({"role": "assistant", "content": text})
        return text

    def exchange(self, turn: Turn, memory_context: str = "") -> list[dict[str, str]]:
        """One user+assistant pair, in MemCache ingest format."""
        user = self.user_turn(turn)
        assistant = self.assistant_turn(user, memory_context)
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    def answer_from_context(self, question: str, memory_context: str) -> str:
        """Ask the model to answer using ONLY retrieved memory.

        Used to check that retrieved context is actually *usable*, not merely
        non-empty. Reported as a metric, never asserted on.
        """
        prompt = (
            "Answer the question using ONLY the memory context below.\n"
            "If the context does not contain the answer, reply exactly: UNKNOWN\n"
            "Answer in one short sentence.\n\n"
            f"Memory context:\n{memory_context.strip() or '(empty)'}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return _strip_wrapping(
            _generate(
                self.base_url,
                self.model,
                prompt,
                timeout=self.timeout,
                temperature=0.0,
            )
        )


def _strip_wrapping(text: str) -> str:
    """Remove the quoting/labelling small models add despite instructions."""
    text = text.strip()
    for prefix in ("User:", "Assistant:", "Message:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return " ".join(text.split())


def summarize_anchor_recall(turns: list[Turn], transcript: list[dict[str, str]]) -> float:
    """Fraction of planned anchors that made it into the user turns.

    A harness self-check: a low value means the traffic generator degraded to
    templates, which is worth knowing when reading results.
    """
    user_text = " ".join(m["content"].lower() for m in transcript if m["role"] == "user")
    anchors = [a for t in turns for a in t.anchors]
    if not anchors:
        return 1.0
    hit = sum(1 for a in anchors if a.lower() in user_text)
    return hit / len(anchors)
