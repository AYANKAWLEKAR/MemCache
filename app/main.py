"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.api.services import close_service_clients
from app.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_service_clients()


app = FastAPI(
    title="Memory-Cache",
    description="Episodic memory infrastructure for long-running agents",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/")
def root():
    """Simple unauthenticated banner for local smoke checks."""
    return {"service": "memory-cache", "status": "running"}
