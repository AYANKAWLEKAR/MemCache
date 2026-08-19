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

from app.config import settings


class TaskHierarchyError(RuntimeError):
    """A SUBGOAL_OF write would break the tree: second parent, self-loop,
    cycle, or cross-user parentage. Raised loudly — never degraded — because a
    wrong parent edge injects one goal's failures into another goal's context.
    """


@dataclass(frozen=True)
class TaskRow:
    """One Task node as seen by callers."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    last_advanced_at: str | None = None


def _row(record) -> TaskRow:
    keys = record.keys() if hasattr(record, "keys") else ()
    return TaskRow(
        id=record["id"],
        title=record["title"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        last_advanced_at=record["last_advanced_at"] if "last_advanced_at" in keys else None,
    )


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

    # ------------------------------------------------------------ hierarchy

    def _depth_cap(self) -> int:
        # Cypher variable-length bounds must be literals; clamp to something sane.
        return max(1, min(int(settings.task_max_depth), 32))

    def get_parent(self, task_id: str) -> TaskRow | None:
        q = """
        MATCH (:Task {id: $task_id})-[:SUBGOAL_OF]->(p:Task)
        RETURN p.id AS id, p.title AS title, p.status AS status,
               p.created_at AS created_at, p.updated_at AS updated_at,
               p.last_advanced_at AS last_advanced_at
        """
        with self._driver.session() as session:
            rec = session.run(q, task_id=task_id).single()
        return _row(rec) if rec else None

    def get_ancestors(self, task_id: str) -> list[TaskRow]:
        """Nearest-first, root last; empty for a root. Bounded by `task_max_depth`
        as a fetch cap. Ignores status on purpose — a closed parent with an open
        child is legitimate and filtering would silently truncate the path."""
        d = self._depth_cap()
        q = f"""
        MATCH p = (:Task {{id: $task_id}})-[:SUBGOAL_OF*1..{d}]->(a:Task)
        WITH a, length(p) AS depth
        ORDER BY depth ASC
        RETURN a.id AS id, a.title AS title, a.status AS status,
               a.created_at AS created_at, a.updated_at AS updated_at,
               a.last_advanced_at AS last_advanced_at
        """
        with self._driver.session() as session:
            return [_row(r) for r in session.run(q, task_id=task_id)]

    def get_children(self, task_id: str) -> list[TaskRow]:
        q = """
        MATCH (c:Task)-[:SUBGOAL_OF]->(:Task {id: $task_id})
        RETURN c.id AS id, c.title AS title, c.status AS status,
               c.created_at AS created_at, c.updated_at AS updated_at,
               c.last_advanced_at AS last_advanced_at
        ORDER BY c.updated_at DESC
        """
        with self._driver.session() as session:
            return [_row(r) for r in session.run(q, task_id=task_id)]

    def get_descendant_ids(self, task_id: str) -> set[str]:
        d = self._depth_cap()
        q = f"""
        MATCH (c:Task)-[:SUBGOAL_OF*1..{d}]->(:Task {{id: $task_id}})
        RETURN DISTINCT c.id AS id
        """
        with self._driver.session() as session:
            return {r["id"] for r in session.run(q, task_id=task_id)}

    def get_lineage_ids(self, task_id: str) -> list[str]:
        """`[task_id, parent, grandparent, ...]` — what retrieval scopes by."""
        return [task_id] + [a.id for a in self.get_ancestors(task_id)]

    def set_parent(self, child_id: str, parent_id: str) -> None:
        """MERGE `(child)-[:SUBGOAL_OF]->(parent)`; no-op if that edge exists.

        Raises `TaskHierarchyError` for: self, a different existing parent, a
        parent inside the child's own subtree (cycle), or a parent pursued by a
        different profile. All checks run before any write.
        """
        if child_id == parent_id:
            raise TaskHierarchyError(f"task {child_id} cannot be its own parent")
        current = self.get_parent(child_id)
        if current is not None:
            if current.id == parent_id:
                return
            raise TaskHierarchyError(
                f"task {child_id} already has parent {current.id}; refusing {parent_id}"
            )
        if parent_id in self.get_descendant_ids(child_id):
            raise TaskHierarchyError(
                f"task {parent_id} is a descendant of {child_id}; parenting would cycle"
            )
        q_owner = """
        MATCH (c:Task {id: $child}), (p:Task {id: $parent})
        OPTIONAL MATCH (uc:UserProfile)-[:PURSUES]->(c)
        OPTIONAL MATCH (up:UserProfile)-[:PURSUES]->(p)
        RETURN uc.user_id AS child_owner, up.user_id AS parent_owner
        """
        with self._driver.session() as session:
            rec = session.run(q_owner, child=child_id, parent=parent_id).single()
            if rec is None:
                raise TaskHierarchyError(f"task {child_id} or {parent_id} does not exist")
            if rec["child_owner"] is None or rec["child_owner"] != rec["parent_owner"]:
                raise TaskHierarchyError(
                    f"tasks {child_id} ({rec['child_owner']!r}) and {parent_id} "
                    f"({rec['parent_owner']!r}) are not pursued by the same profile"
                )
            session.run(
                """
                MATCH (c:Task {id: $child}), (p:Task {id: $parent})
                MERGE (c)-[:SUBGOAL_OF]->(p)
                """,
                child=child_id,
                parent=parent_id,
            )
