"""L3 Task nodes: inferred goals a user is pursuing, linked to the episodes
that advance them.

Tasks live in Neo4j because they are relational — they connect episodes,
decisions, and tool calls — and need no vector search. `updated_at` advances
whenever an episode links, which is what makes "most recently active" the
candidate ordering for adjudication.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import Driver


@dataclass(frozen=True)
class TaskRow:
    """One Task node as seen by callers."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    """Graph operations for inferred user tasks."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def create_task(self, user_id: str, title: str) -> str:
        """Create an open Task pursued by this profile. Returns its UUID."""
        task_id = str(uuid.uuid4())
        q = """
        MERGE (p:UserProfile {user_id: $user_id})
        CREATE (t:Task {
            id: $task_id,
            title: $title,
            status: 'open',
            created_at: $now,
            updated_at: $now
        })
        MERGE (p)-[:PURSUES]->(t)
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, task_id=task_id, title=title.strip(), now=_now())
        return task_id

    def get_task(self, task_id: str) -> TaskRow | None:
        q = """
        MATCH (t:Task {id: $task_id})
        RETURN t.id AS id, t.title AS title, t.status AS status,
               t.created_at AS created_at, t.updated_at AS updated_at
        """
        with self._driver.session() as session:
            record = session.run(q, task_id=task_id).single()
        if record is None:
            return None
        return TaskRow(
            id=record["id"],
            title=record["title"],
            status=record["status"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    def list_open_tasks(self, user_id: str, *, limit: int) -> list[TaskRow]:
        """Open tasks for a user, most recently active first.

        The limit is the adjudication candidate cap — enforced here, in code,
        because an unbounded task list in the prompt collapses a small model's
        judgement.
        """
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:PURSUES]->(t:Task {status: 'open'})
        RETURN t.id AS id, t.title AS title, t.status AS status,
               t.created_at AS created_at, t.updated_at AS updated_at
        ORDER BY t.updated_at DESC
        LIMIT $limit
        """
        with self._driver.session() as session:
            return [
                TaskRow(
                    id=r["id"],
                    title=r["title"],
                    status=r["status"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in session.run(q, user_id=user_id, limit=max(1, limit))
            ]

    def link_episode(self, task_id: str, *, episode_id: int) -> None:
        """MERGE `(:Episode)-[:ADVANCES]->(:Task)` and bump task activity."""
        q = """
        MATCH (t:Task {id: $task_id})
        MERGE (e:Episode {id: $episode_id})
        MERGE (e)-[:ADVANCES]->(t)
        SET t.updated_at = $now
        """
        with self._driver.session() as session:
            session.run(q, task_id=task_id, episode_id=episode_id, now=_now())

    def close_task(self, task_id: str) -> None:
        """Mark a task done. Idempotent."""
        q = """
        MATCH (t:Task {id: $task_id})
        SET t.status = 'done', t.closed_at = $now, t.updated_at = $now
        """
        with self._driver.session() as session:
            session.run(q, task_id=task_id, now=_now())
