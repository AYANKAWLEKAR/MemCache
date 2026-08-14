"""FastAPI dependencies for the Memory-Cache API."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate the configured API key header."""
    if x_api_key and x_api_key in settings.get_valid_api_keys():
        return x_api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
