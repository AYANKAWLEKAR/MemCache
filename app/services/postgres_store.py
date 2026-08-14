"""L2 PostgreSQL + pgvector: episodic summaries and cosine similarity search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.postgres import Episode


@dataclass(frozen=True)
class EpisodeSearchResult:
    """One row from `search_episodes`, including cosine distance to the query."""

    id: int
    session_id: str
    summary: str
    start_time: datetime
    end_time: datetime
    episode_metadata: dict[str, Any] | None
    distance: float
    #: Canonical owner; None for rows written before ownership existed. Last so
    #: existing positional construction keeps working.
    user_id: str | None = None


class PostgresStore:
    """CRUD and vector search over `episodes` (session-scoped, `<=>` cosine distance)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def insert_episode(
        self,
        session_id: str,
        summary: str,
        embedding: Sequence[float],
        start_time: datetime,
        end_time: datetime,
        metadata: Mapping[str, Any] | None = None,
        user_id: str | None = None,
    ) -> int:
        """Insert one episode; returns PostgreSQL `id` (used by L3 Episode linkage in PRD)."""
        row = Episode(
            session_id=session_id,
            user_id=user_id,
            summary=summary,
            embedding=list(embedding),
            start_time=start_time,
            end_time=end_time,
            episode_metadata=dict(metadata) if metadata is not None else None,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return row.id

    def search_episodes(
        self,
        query_embedding: Sequence[float],
        session_id: str,
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[EpisodeSearchResult]:
        """Episodes nearest `query_embedding`, ordered by cosine distance.

        Without `user_id` the search is session-locked, exactly as before. With
        it, the scope widens to the whole user's history *plus* the current
        session — the `OR` keeps the live conversation visible even when its rows
        predate ownership, so enabling this mid-history does not blind it.
        """
        if limit <= 0:
            return []
        q = list(query_embedding)
        dist = Episode.embedding.cosine_distance(q)
        scope = (
            Episode.session_id == session_id
            if user_id is None
            else or_(Episode.user_id == user_id, Episode.session_id == session_id)
        )
        stmt = (
            select(Episode, dist.label("distance"))
            .where(scope)
            .order_by(dist)
            .limit(limit)
        )
        rows = self._session.execute(stmt).all()
        out: list[EpisodeSearchResult] = []
        for ep, distance in rows:
            out.append(
                EpisodeSearchResult(
                    id=ep.id,
                    session_id=ep.session_id,
                    user_id=ep.user_id,
                    summary=ep.summary,
                    start_time=ep.start_time,
                    end_time=ep.end_time,
                    episode_metadata=ep.episode_metadata,
                    distance=float(distance),
                )
            )
        return out

    def get_episode_by_id(self, episode_id: int) -> Episode | None:
        """Load a single episode by primary key (handy for tests and admin)."""
        return self._session.get(Episode, episode_id)

    def find_episode_id_by_celery_task(
        self,
        session_id: str,
        celery_task_id: str,
    ) -> int | None:
        """Idempotency for Celery retries: match ``metadata.celery_task_id`` if present."""
        if not celery_task_id:
            return None
        stmt = (
            select(Episode.id)
            .where(Episode.session_id == session_id)
            .where(Episode.episode_metadata["celery_task_id"].astext == celery_task_id)
            .limit(1)
        )
        row = self._session.execute(stmt).scalar_one_or_none()
        return int(row) if row is not None else None
