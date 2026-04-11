"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One conversation turn accepted by the ingest endpoint."""

    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class MemoryIngestRequest(BaseModel):
    """Payload for storing raw turns and triggering async processing."""

    session_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class MemoryIngestResponse(BaseModel):
    """Accepted ingest response with the Celery task id."""

    status: Literal["accepted"]
    task_id: str
    session_id: str


class BackendHealth(BaseModel):
    """Connectivity status for one backend service."""

    ok: bool
    detail: str


class HealthResponse(BaseModel):
    """Health response with backend-level detail."""

    status: Literal["ok", "degraded"]
    redis: BackendHealth
    postgres: BackendHealth
    neo4j: BackendHealth


class MemoryRetrieveRequest(BaseModel):
    """Payload for hybrid context retrieval."""

    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)


class MemorySource(BaseModel):
    """One structured provenance item returned by hybrid retrieval."""

    type: str = Field(min_length=1)
    tier: Literal["L1", "L2", "L3"]
    details: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrieveResponse(BaseModel):
    """Hybrid retrieval response."""

    context: str
    sources: list[MemorySource]
    status: Literal["ok", "degraded"]
    warnings: list[str] = Field(default_factory=list)
