"""Read-only Cypher probes used by assertions.

Kept separate from `Neo4jStore` on purpose: tests should verify the graph by
querying it independently, not by calling the same methods that wrote it. A bug
in `query_session_entities` would otherwise hide a bug in `merge_entities`.
"""

from __future__ import annotations

from typing import Any


def mentioned_entities(driver: Any, session_id: str) -> set[str]:
    """Normalized entity names reachable as Session -> Episode -> Entity."""
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(:Episode)-[:MENTIONS]->(e:Entity)
    RETURN DISTINCT e.name AS name
    """
    with driver.session() as s:
        return {r["name"] for r in s.run(q, sid=session_id)}


def episode_ids(driver: Any, session_id: str) -> set[int]:
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(e:Episode)
    RETURN e.id AS id
    """
    with driver.session() as s:
        return {r["id"] for r in s.run(q, sid=session_id)}


def related_pairs(driver: Any, session_id: str) -> set[frozenset[str]]:
    """Unordered RELATED_TO pairs among entities this session mentions."""
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(:Episode)-[:MENTIONS]->(a:Entity)
    MATCH (a)-[:RELATED_TO]-(b:Entity)
    RETURN DISTINCT a.name AS a, b.name AS b
    """
    with driver.session() as s:
        return {frozenset((r["a"], r["b"])) for r in s.run(q, sid=session_id)}


def decision_texts(driver: Any, session_id: str) -> list[str]:
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(:Episode)-[:DECIDED]->(d:Decision)
    RETURN DISTINCT d.text AS text
    """
    with driver.session() as s:
        return [r["text"] for r in s.run(q, sid=session_id)]


def preference_texts(driver: Any, session_id: str) -> list[str]:
    q = """
    MATCH (:Session {id: $sid})-[:HAS_EPISODE]->(:Episode)-[:PREFERS]->(p:Preference)
    RETURN DISTINCT p.text AS text
    """
    with driver.session() as s:
        return [r["text"] for r in s.run(q, sid=session_id)]


def entity_nodes_like(driver: Any, prefix: str) -> set[str]:
    """All Entity names starting with `prefix` — catches near-duplicate nodes."""
    q = """
    MATCH (e:Entity)
    WHERE e.name STARTS WITH $prefix
    RETURN e.name AS name
    """
    with driver.session() as s:
        return {r["name"] for r in s.run(q, prefix=prefix.lower())}


def episode_rows(engine: Any, session_id: str) -> list[dict[str, Any]]:
    """L2 rows for a session, including whether an embedding was stored."""
    sql = """
    SELECT id, summary, (embedding IS NOT NULL) AS has_embedding
    FROM episodes WHERE session_id = %s ORDER BY id
    """
    with engine.begin() as conn:
        rows = conn.exec_driver_sql(sql, (session_id,)).fetchall()
    return [
        {"id": r[0], "summary": r[1], "has_embedding": bool(r[2])} for r in rows
    ]
