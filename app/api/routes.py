"""HTTP routes for the first Memory-Cache API slice."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.deps import require_api_key
from app.api.models import (
    BackendHealth,
    HealthResponse,
    MemoryIngestRequest,
    MemoryIngestResponse,
)
from app.api import services as api_services

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
