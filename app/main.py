"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import settings

app = FastAPI(
    title="Memory-Cache",
    description="Episodic memory infrastructure for long-running agents",
    version="0.1.0",
)


@app.get("/")
def root():
    """Health check at root."""
    return {"service": "memory-cache", "status": "running"}
