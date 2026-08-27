"""HTTP routes for the first Memory-Cache API slice."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.api import services as api_services
from app.api.deps import require_api_key
from app.api.models import (
    BackendHealth,
    HealthResponse,
    MemoryIngestRequest,
    MemoryIngestResponse,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    ProfileAliasRequest,
    ProfileAttributeValue,
    ProfileResponse,
    ProfileUpdateRequest,
    WorkbenchCall,
    WorkbenchRecentResponse,
    WorkbenchToolCallRequest,
    WorkbenchToolCallResponse,
)
from app.services import workbench_store
from app.services.profile_store import (
    ATTRIBUTE_KEYS,
    ProfileAliasConflictError,
    ProfileStore,
    resolve_attributes,
)
from app.services.retrieval import RetrievalError, retrieve_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(_api_key: str = Depends(require_api_key)):
    """Check connectivity to Redis, PostgreSQL, and Neo4j."""
    redis_ok, redis_detail = api_services.check_redis_health()
    postgres_ok, postgres_detail = api_services.check_postgres_health()
    neo4j_ok, neo4j_detail = api_services.check_neo4j_health()

    ok = redis_ok and postgres_ok and neo4j_ok
    payload = HealthResponse(
        status="ok" if ok else "degraded",
        redis=BackendHealth(ok=redis_ok, detail=redis_detail),
        postgres=BackendHealth(ok=postgres_ok, detail=postgres_detail),
        neo4j=BackendHealth(ok=neo4j_ok, detail=neo4j_detail),
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.post(
    "/memory/ingest",
    response_model=MemoryIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_memory(
    payload: MemoryIngestRequest,
    _api_key: str = Depends(require_api_key),
) -> MemoryIngestResponse:
    """Persist raw turns to Redis and enqueue asynchronous processing."""
    messages = [message.model_dump() for message in payload.messages]

    try:
        api_services.get_redis_store().append_messages(payload.session_id, messages)
    except Exception as exc:
        logger.exception("Failed to append raw messages to Redis")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to persist messages to Redis",
        ) from exc

    try:
        task = api_services.enqueue_conversation_task(
            payload.session_id,
            messages,
            payload.metadata,
            payload.user_id,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue background processing task")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue background processing",
        ) from exc

    return MemoryIngestResponse(
        status="accepted",
        task_id=str(task.id),
        session_id=payload.session_id,
    )


@router.post("/memory/retrieve", response_model=MemoryRetrieveResponse)
def retrieve_memory(
    payload: MemoryRetrieveRequest,
    _api_key: str = Depends(require_api_key),
) -> MemoryRetrieveResponse:
    """Return hybrid memory context for a query."""
    try:
        result = retrieve_context(
            payload.session_id,
            payload.query,
            payload.max_tokens,
            payload.user_id,
        )
    except RetrievalError as exc:
        logger.exception("Failed to retrieve required Redis context")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return MemoryRetrieveResponse(**result)


def _profile_store() -> ProfileStore:
    return ProfileStore(api_services.get_neo4j_driver())


def _profile_response(store: ProfileStore, user_id: str) -> ProfileResponse:
    row = store.get_profile(user_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile for user_id {user_id!r}",
        )
    current = resolve_attributes(store.get_attributes(user_id))
    return ProfileResponse(
        user_id=row.user_id,
        display_name=row.display_name,
        attributes={
            key: ProfileAttributeValue(
                value=attr.value,
                source=attr.source,
                confidence=attr.confidence,
                observed_at=attr.observed_at,
                evidence=attr.evidence,
            )
            for key, attr in current.items()
        },
        aliases=store.get_aliases(user_id),
        decisions=store.get_profile_decisions(user_id),
        preferences=store.get_profile_preferences(user_id),
    )


@router.get("/profile/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str, _api_key: str = Depends(require_api_key)) -> ProfileResponse:
    """Return the resolved canonical profile."""
    return _profile_response(_profile_store(), user_id)


@router.patch("/profile/{user_id}", response_model=ProfileResponse)
def update_profile(
    user_id: str,
    payload: ProfileUpdateRequest,
    _api_key: str = Depends(require_api_key),
) -> ProfileResponse:
    """Set attributes explicitly. Explicit values always beat inferred ones."""
    unknown = sorted(set(payload.attributes) - ATTRIBUTE_KEYS)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown attribute keys: {unknown}; expected {sorted(ATTRIBUTE_KEYS)}",
        )

    store = _profile_store()
    store.upsert_profile(user_id, display_name=payload.display_name)
    for key, value in payload.attributes.items():
        store.set_attribute(user_id, key, value, source="explicit", confidence=1.0)
    return _profile_response(store, user_id)


@router.post("/profile/{user_id}/alias", response_model=ProfileResponse)
def add_profile_alias(
    user_id: str,
    payload: ProfileAliasRequest,
    _api_key: str = Depends(require_api_key),
) -> ProfileResponse:
    """Register an alias manually. Conflicts are reported, never resolved by guessing."""
    store = _profile_store()
    store.upsert_profile(user_id)
    try:
        store.link_alias(user_id, payload.entity_name, source="explicit", confidence=1.0)
    except ProfileAliasConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _profile_response(store, user_id)


@router.post(
    "/workbench/tool-call",
    response_model=WorkbenchToolCallResponse,
    status_code=status.HTTP_201_CREATED,
)
def record_workbench_tool_call(
    payload: WorkbenchToolCallRequest,
    _api_key: str = Depends(require_api_key),
) -> WorkbenchToolCallResponse:
    """Record one tool invocation in the L4 workbench."""
    engine = api_services.ensure_workbench_ready()
    recorded = workbench_store.record_tool_call(
        engine,
        session_id=payload.session_id,
        tool_name=payload.tool_name,
        status=payload.status,
        args=payload.args,
        output=payload.output,
        error=payload.error,
        user_id=payload.user_id,
        task_id=payload.task_id,
        duration_ms=payload.duration_ms,
    )
    return WorkbenchToolCallResponse(
        id=recorded.id, call_hash=recorded.call_hash, truncated=recorded.truncated
    )


@router.get("/workbench/recent", response_model=WorkbenchRecentResponse)
def recent_workbench_calls(
    session_id: str | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
    call_status: str | None = None,
    tool_name: str | None = None,
    call_hash: str | None = None,
    limit: int = 20,
    _api_key: str = Depends(require_api_key),
) -> WorkbenchRecentResponse:
    """Newest-first tool calls; dedup via call_hash, failure review via call_status=error."""
    engine = api_services.ensure_workbench_ready()
    rows = workbench_store.recent_tool_calls(
        engine,
        session_id=session_id,
        user_id=user_id,
        task_id=task_id,
        status=call_status,
        tool_name=tool_name,
        call_hash=call_hash,
        limit=max(1, min(limit, 200)),
    )
    return WorkbenchRecentResponse(
        calls=[
            WorkbenchCall(
                id=r.id,
                session_id=r.session_id,
                user_id=r.user_id,
                task_id=r.task_id,
                episode_id=r.episode_id,
                tool_name=r.tool_name,
                args=r.args,
                status=r.status,
                output=r.output,
                error=r.error,
                output_bytes=r.output_bytes,
                truncated=r.truncated,
                call_hash=r.call_hash,
                duration_ms=r.duration_ms,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
    )
