"""L3 Neo4j: Session / Episode / Entity / Decision / Preference graph (PRD)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from neo4j import Driver


def normalize_entity_name(name: str) -> str:
    """Normalize for MERGE uniqueness: lowercase, collapse whitespace, strip."""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class GraphEntityRow:
    """Entity returned from session-scoped or traversal queries."""

    name: str
    display_name: str | None = None


class Neo4jStore:
    """Graph operations: upserts, co-occurrence edges, and retrieval-oriented queries."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def upsert_session(self, session_id: str) -> None:
        """MERGE Session by `id` (conversation/session key from API)."""
        q = """
        MERGE (s:Session {id: $session_id})
        """
        with self._driver.session() as session:
            session.run(q, session_id=session_id)

    def upsert_episode(
        self,
        session_id: str,
        episode_id: int,
        summary: str,
    ) -> None:
        """MERGE Session and Episode (L2 row id), link with HAS_EPISODE.

        `episode_id` is the PostgreSQL `episodes.id` returned by L2 `insert_episode`.
        """
        q = """
        MERGE (s:Session {id: $session_id})
        MERGE (e:Episode {id: $episode_id})
        SET e.summary = $summary, e.session_id = $session_id
        MERGE (s)-[:HAS_EPISODE]->(e)
        """
        with self._driver.session() as session:
            session.run(
                q,
                session_id=session_id,
                episode_id=episode_id,
                summary=summary,
            )

    def merge_entities(
        self,
        names: list[str],
        *,
        episode_id: int | None = None,
    ) -> list[str]:
        """MERGE Entity nodes on normalized `name`.

        When `episode_id` is set, creates MENTIONS from that Episode to each Entity.
        Returns normalized names in input order (deduped by normalization).
        """
        if not names:
            return []
        seen: set[str] = set()
        ordered: list[tuple[str, str]] = []
        for raw in names:
            norm = normalize_entity_name(raw)
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            ordered.append((raw, norm))

        q_merge_only = """
        UNWIND $rows AS row
        MERGE (ent:Entity {name: row.norm})
        SET ent.display_name = row.raw
        """
        q_with_mentions = """
        UNWIND $rows AS row
        MERGE (ent:Entity {name: row.norm})
        SET ent.display_name = row.raw
        WITH ent
        MATCH (ep:Episode {id: $episode_id})
        MERGE (ep)-[:MENTIONS]->(ent)
        """
        rows = [{"raw": raw, "norm": norm} for raw, norm in ordered]
        cypher = q_with_mentions if episode_id is not None else q_merge_only
        with self._driver.session() as session:
            if episode_id is not None:
                session.run(cypher, rows=rows, episode_id=episode_id)
            else:
                session.run(cypher, rows=rows)
        return [norm for _, norm in ordered]

    def create_relationships(self, pairs: list[tuple[str, str]]) -> None:
        """Create RELATED_TO between Entity pairs (normalized names). MERGEs missing entities."""
        if not pairs:
            return
        q = """
        UNWIND $pairs AS pair
        MERGE (a:Entity {name: pair.a})
        MERGE (b:Entity {name: pair.b})
        MERGE (a)-[:RELATED_TO]->(b)
        """
        payload = [
            {"a": normalize_entity_name(a), "b": normalize_entity_name(b)}
            for a, b in pairs
            if normalize_entity_name(a) and normalize_entity_name(b)
        ]
        if not payload:
            return
        with self._driver.session() as session:
            session.run(q, pairs=payload)

    def record_decisions_and_preferences(
        self,
        episode_id: int,
        decisions: list[str],
        preferences: list[str],
    ) -> None:
        """Attach Decision / Preference nodes to an Episode via DECIDED / PREFERS.

        Uses stable synthetic ids per episode+text so reruns stay idempotent.
        """
        def stable_suffix(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        q_dec = """
        MATCH (ep:Episode {id: $episode_id})
        UNWIND $rows AS row
        MERGE (d:Decision {id: row.did})
        SET d.text = row.text
        MERGE (ep)-[:DECIDED]->(d)
        """
        q_pref = """
        MATCH (ep:Episode {id: $episode_id})
        UNWIND $rows AS row
        MERGE (p:Preference {id: row.pid})
        SET p.text = row.text
        MERGE (ep)-[:PREFERS]->(p)
        """
        dec_rows: list[dict[str, Any]] = []
        for t in decisions:
            tt = t.strip()
            if not tt:
                continue
            did = f"{episode_id}:dec:{stable_suffix(tt)}"
            dec_rows.append({"did": did, "text": tt})
        pref_rows: list[dict[str, Any]] = []
        for t in preferences:
            tt = t.strip()
            if not tt:
                continue
            pid = f"{episode_id}:pref:{stable_suffix(tt)}"
            pref_rows.append({"pid": pid, "text": tt})

        with self._driver.session() as session:
            if dec_rows:
                session.run(q_dec, episode_id=episode_id, rows=dec_rows)
            if pref_rows:
                session.run(q_pref, episode_id=episode_id, rows=pref_rows)

    def query_session_entities(self, session_id: str) -> list[GraphEntityRow]:
        """Entities mentioned across episodes for this session."""
        q = """
        MATCH (s:Session {id: $session_id})-[:HAS_EPISODE]->(:Episode)-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e.name AS name, e.display_name AS display_name
        ORDER BY name
        """
        with self._driver.session() as session:
            result = session.run(q, session_id=session_id)
            rows = []
            for record in result:
                rows.append(
                    GraphEntityRow(
                        name=record["name"],
                        display_name=record["display_name"],
                    )
                )
            return rows

    def query_related_entities(self, entity_name: str, *, max_hops: int = 2) -> list[str]:
        """Variable-length RELATED_TO (undirected) up to `max_hops`."""
        norm = normalize_entity_name(entity_name)
        if not norm:
            return []
        # max_hops bound must be in query statically for Neo4j
        mh = max(1, min(max_hops, 5))
        q = f"""
        MATCH (e:Entity {{name: $name}})-[:RELATED_TO*1..{mh}]-(o:Entity)
        WHERE e <> o
        RETURN DISTINCT o.name AS name
        ORDER BY name
        """
        with self._driver.session() as session:
            result = session.run(q, name=norm)
            return [r["name"] for r in result]

    def query_decisions_preferences(self, session_id: str) -> dict[str, list[str]]:
        """All Decision and Preference texts reachable from this session's episodes."""
        q = """
        MATCH (s:Session {id: $session_id})-[:HAS_EPISODE]->(ep:Episode)
        OPTIONAL MATCH (ep)-[:DECIDED]->(d:Decision)
        OPTIONAL MATCH (ep)-[:PREFERS]->(p:Preference)
        RETURN collect(DISTINCT d.text) AS decisions, collect(DISTINCT p.text) AS preferences
        """
        with self._driver.session() as session:
            record = session.run(q, session_id=session_id).single()
            if record is None:
                return {"decisions": [], "preferences": []}
            dec = [x for x in record["decisions"] if x]
            pref = [x for x in record["preferences"] if x]
            return {"decisions": sorted(set(dec)), "preferences": sorted(set(pref))}
