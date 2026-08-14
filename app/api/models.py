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
    #: Optional canonical identity. When absent, no profile work occurs.
    user_id: str | None = Field(default=None, min_length=1)
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
    #: Optional canonical identity. When present, profile facts are included.
    user_id: str | None = Field(default=None, min_length=1)


class MemorySource(BaseModel):
    """One structured provenance item returned by hybrid retrieval."""

    type: str = Field(min_length=1)
    tier: Literal["L1", "L2", "L3", "L4"]
    details: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrieveResponse(BaseModel):
    """Hybrid retrieval response."""

    context: str
    sources: list[MemorySource]
    status: Literal["ok", "degraded"]
    warnings: list[str] = Field(default_factory=list)


class ProfileAttributeValue(BaseModel):
    """One resolved attribute value with its provenance."""

    value: str
    source: Literal["explicit", "inferred"]
    confidence: float
    observed_at: str
    evidence: str | None = None


class ProfileResponse(BaseModel):
    """Resolved canonical profile."""

    user_id: str
    display_name: str | None = None
    attributes: dict[str, ProfileAttributeValue] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


class ProfileUpdateRequest(BaseModel):
    """Explicitly set profile attributes. Overrides any inferred value."""

    attributes: dict[str, str] = Field(default_factory=dict)
    display_name: str | None = None


class ProfileAliasRequest(BaseModel):
    """Manually register an alias for a profile."""

    entity_name: str = Field(min_length=1)


class WorkbenchToolCallRequest(BaseModel):
    """One tool invocation to record in the L4 workbench."""

    session_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: Literal["ok", "error"]
    args: dict[str, Any] | None = None
    output: str | None = None
    error: str | None = None
    user_id: str | None = Field(default=None, min_length=1)
    task_id: str | None = Field(default=None, min_length=1)
    duration_ms: int | None = Field(default=None, ge=0)


class WorkbenchToolCallResponse(BaseModel):
    """Receipt: enough to dedup later and to know a payload was cut."""

    id: int
    call_hash: str
    truncated: bool


class WorkbenchCall(BaseModel):
    """One stored tool call as returned by /workbench/recent."""

    id: int
    session_id: str
    user_id: str | None = None
    task_id: str | None = None
    episode_id: int | None = None
    tool_name: str
    args: dict[str, Any] | None = None
    status: str
    output: str | None = None
    error: str | None = None
    output_bytes: int
    truncated: bool
    call_hash: str
    duration_ms: int | None = None
    created_at: str


class WorkbenchRecentResponse(BaseModel):
    """Newest-first tool calls matching the given filters."""

    calls: list[WorkbenchCall] = Field(default_factory=list)
