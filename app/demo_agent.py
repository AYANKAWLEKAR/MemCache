"""Terminal demo agent that chats via Ollama and logs MemCache provenance."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import defaultdict
from typing import Any

import httpx

from app.config import settings

SOURCE_TIER_MAP = {
    "recent_message": "L1",
    "episode": "L2",
    "graph_entity": "L3",
    "graph_related": "L3",
    "decision": "L3",
    "preference": "L3",
}
TIER_ORDER = ("L1", "L2", "L3")


def classify_source_tier(source: dict[str, Any]) -> str:
    """Return the explicit source tier, with a compatibility fallback."""
    explicit = source.get("tier")
    if explicit in TIER_ORDER:
        return str(explicit)
    return SOURCE_TIER_MAP.get(str(source.get("type", "")), "L3")


def _details(source: dict[str, Any]) -> dict[str, Any]:
    details = source.get("details")
    if isinstance(details, dict):
        return details
    return source


def _format_float(value: Any) -> str | None:
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return None


def format_source_detail(source: dict[str, Any]) -> str:
    """Render one retrieval source into a human-readable log line."""
    source_type = str(source.get("type", "unknown"))
    details = _details(source)
    if source_type == "recent_message":
        return f"recent_message session={details.get('session_id')} index={details.get('index')}"
    if source_type == "episode":
        parts = [
            f"episode id={details.get('episode_id')}",
            f"session={details.get('session_id')}",
        ]
        similarity = _format_float(details.get("similarity"))
        distance = _format_float(details.get("distance"))
        if similarity is not None:
            parts.append(f"similarity={similarity}")
        if distance is not None:
            parts.append(f"distance={distance}")
        return " ".join(parts)
    if source_type == "graph_entity":
        return f"graph_entity name={details.get('name')}"
    if source_type == "graph_related":
        related = ", ".join(str(item) for item in details.get("related", []))
        return f"graph_related name={details.get('name')} related=[{related}]"
    if source_type in {"decision", "preference"}:
        return f"{source_type} text={details.get('text')}"
    return f"{source_type} {source}"


def render_provenance_log(
    *,
    status: str,
    warnings: list[str],
    sources: list[dict[str, Any]],
) -> str:
    """Render a structured evidence block grouped by memory tier."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        tier = classify_source_tier(source)
        grouped[tier].append(format_source_detail(source))

    lines = [
        "Memory provenance",
        f"  retrieval_status: {status}",
    ]
    if warnings:
        lines.append("  warnings:")
        for warning in warnings:
            lines.append(f"    - {warning}")
    else:
        lines.append("  warnings: none")

    for tier in TIER_ORDER:
        entries = grouped.get(tier, [])
        lines.append(f"  {tier}:")
        if entries:
            for entry in entries:
                lines.append(f"    - {entry}")
        else:
            lines.append("    - none")
    return "\n".join(lines)


def build_agent_prompt(*, user_message: str, memory_context: str) -> str:
    """Build a strict answer-generation prompt for the demo agent."""
    context_block = memory_context.strip() or "No memory context was retrieved."
    return (
        "You are a demo assistant connected to MemCache.\n"
        "Use the retrieved memory context when it is relevant.\n"
        "If memory is missing or insufficient, say so plainly instead of inventing facts.\n"
        "Do not claim information came from memory unless it appears in the provided context.\n"
        "Answer conversationally and concisely.\n\n"
        f"Retrieved memory context:\n{context_block}\n\n"
        f"User message: {user_message}\n\n"
        "Assistant answer:"
    )


class DemoAgent:
    """HTTP client wrapper for the terminal demo flow."""

    def __init__(
        self,
        *,
        memcache_base_url: str,
        memcache_api_key: str,
        ollama_base_url: str,
        ollama_model: str,
        ollama_api_key: str | None,
        session_id: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.memcache_base_url = memcache_base_url.rstrip("/")
        self.memcache_api_key = memcache_api_key
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_api_key = (ollama_api_key or "").strip() or None
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds

    @property
    def _memcache_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.memcache_api_key}

    @property
    def _ollama_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"
        return headers

    def preflight(self) -> dict[str, Any]:
        """Verify the MemCache API is healthy before entering the chat loop."""
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(
                f"{self.memcache_base_url}/health",
                headers=self._memcache_headers,
            )
            response.raise_for_status()
            body = response.json()
        if body.get("status") != "ok":
            raise RuntimeError(f"MemCache health check degraded: {body}")
        return body

    def retrieve_memory(self, user_message: str, max_tokens: int | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.memcache_base_url}/memory/retrieve",
                headers=self._memcache_headers,
                json={
                    "session_id": self.session_id,
                    "query": user_message,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()

    def generate_answer(self, user_message: str, memory_context: str) -> str:
        prompt = build_agent_prompt(user_message=user_message, memory_context=memory_context)
        with httpx.Client(timeout=self.timeout_seconds * 4) as client:
            response = client.post(
                f"{self.ollama_base_url}/api/generate",
                headers=self._ollama_headers,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            body = response.json()
        answer = str(body.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response for demo generation")
        return answer

    def ingest_turn(self, user_message: str, answer: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.memcache_base_url}/memory/ingest",
                headers=self._memcache_headers,
                json={
                    "session_id": self.session_id,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {"source": "demo-agent"},
                },
            )
            response.raise_for_status()
            return response.json()

    def chat_turn(self, user_message: str, max_tokens: int | None = None) -> dict[str, Any]:
        retrieval = self.retrieve_memory(user_message, max_tokens=max_tokens)
        answer = self.generate_answer(user_message, retrieval.get("context", ""))
        ingest = self.ingest_turn(user_message, answer)
        provenance = render_provenance_log(
            status=str(retrieval.get("status", "unknown")),
            warnings=list(retrieval.get("warnings", [])),
            sources=list(retrieval.get("sources", [])),
        )
        return {
            "answer": answer,
            "provenance": provenance,
            "retrieval": retrieval,
            "ingest": ingest,
        }


def _build_agent_from_settings(session_id: str) -> DemoAgent:
    ollama_base_url = (settings.demo_ollama_base_url or settings.ollama_base_url).strip()
    ollama_model = (settings.demo_ollama_model or settings.ollama_model).strip()
    ollama_api_key = settings.demo_ollama_api_key or settings.ollama_api_key
    return DemoAgent(
        memcache_base_url=settings.demo_memcache_base_url,
        memcache_api_key=settings.demo_memcache_api_key,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_api_key=ollama_api_key,
        session_id=session_id,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MemCache demo agent.")
    parser.add_argument(
        "--session-id",
        default=f"demo-{uuid.uuid4()}",
        help="Conversation session identifier used for MemCache retrieval and ingest.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional retrieval token budget for the MemCache context request.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    agent = _build_agent_from_settings(args.session_id)

    try:
        health = agent.preflight()
    except Exception as exc:
        print(f"MemCache preflight failed: {exc}", file=sys.stderr)
        return 1

    print("MemCache demo agent ready.")
    print(f"Session ID: {agent.session_id}")
    print(f"MemCache health: {health['status']}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except EOFError:
            print()
            return 0

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            return 0

        try:
            result = agent.chat_turn(user_message, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            continue

        print(f"\nAssistant: {result['answer']}\n")
        print(result["provenance"])
        print()


if __name__ == "__main__":
    raise SystemExit(main())
