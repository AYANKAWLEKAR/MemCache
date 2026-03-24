"""SQLAlchemy engine, session helpers, and L2 `episodes` model (pgvector)."""

from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from sqlalchemy import DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from pgvector.sqlalchemy import Vector

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


class Episode(Base):
    """L2 episodic summary row: semantic chunk per session for retrieval (PRD)."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dimension))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    episode_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )


def create_engine_from_settings(url: str | None = None) -> Engine:
    return create_engine(url or settings.postgres_url, pool_pre_ping=True)


def ensure_l2_schema(engine: Engine) -> None:
    """Create pgvector extension, `episodes` table, and B-tree on `session_id` if missing.

    Idempotent. Keeps older Docker volumes (initialized before L2 DDL existed) usable without
    recreating volumes; matches `scripts/init-postgres.sql`.
    """
    dim = settings.embedding_dimension
    ddl_table = f"""
        CREATE TABLE IF NOT EXISTS episodes (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            summary TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            metadata JSONB
        )
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(ddl_table))
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes (session_id)"
            )
        )


def ensure_ivfflat_index(
    engine: Engine,
    *,
    target_lists: int = 100,
    index_name: str = "idx_episodes_embedding_ivfflat",
) -> bool:
    """Create the IVFFlat cosine index when the table has enough rows (rows >= lists).

    pgvector refuses to build IVFFlat when row count is below `lists`. For smaller tables,
    `lists` is lowered to the row count. If the index already exists, does nothing.

    Returns True if the index exists or was created, False if there are zero rows.
    """
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", index_name):
        raise ValueError("invalid index_name for SQL identifier")
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT 1 FROM pg_indexes WHERE indexname = :name LIMIT 1"
            ),
            {"name": index_name},
        ).fetchone()
        if exists:
            return True
        n = conn.execute(text("SELECT COUNT(*) FROM episodes")).scalar_one()
    if n == 0:
        return False
    lists = min(target_lists, int(n))
    ddl = text(
        f"CREATE INDEX IF NOT EXISTS {index_name} ON episodes "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
    )
    with engine.begin() as conn:
        conn.execute(ddl)
    return True


@contextmanager
def session_scope(engine: Engine) -> Generator[Session, None, None]:
    """Yield a Session with commit on success, rollback on error."""
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
