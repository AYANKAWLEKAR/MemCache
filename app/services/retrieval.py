"""Placeholder retrieval service for the next API step."""

from __future__ import annotations

from typing import Any


def retrieve_context(
    session_id: str,
    query: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Reserved interface for the future `/memory/retrieve` endpoint."""
    raise NotImplementedError(
        "retrieve_context is planned for the next step and is not implemented yet"
    )
