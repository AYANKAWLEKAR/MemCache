"""L3 Neo4j: Session / Episode / Entity / Decision / Preference graph (PRD)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from neo4j import Driver

# Leading/trailing punctuation that NER routinely leaves attached to a span
# ("Acme Corp." / "(Acme Corp)" / "Acme Corp,"). Kept conservative on purpose:
# interior punctuation is meaningful ("AT&T", "Yahoo!Japan", "St. Louis").
_EDGE_PUNCT = r"""[\s.,;:!?'"“”‘’()\[\]{}]+"""


def normalize_entity_name(name: str) -> str:
    """Normalize for MERGE uniqueness: lowercase, collapse whitespace, strip edge punctuation.

    Trailing punctuation is stripped so a sentence-final mention ("Acme Corp.")
    unifies with the same entity mid-sentence ("Acme Corp"). Possessives are
    folded for the same reason ("Acme Corp's" -> "acme corp").
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(rf"^{_EDGE_PUNCT}", "", s)
    s = re.sub(rf"{_EDGE_PUNCT}$", "", s)
    # Fold possessives after edge-stripping so "Acme Corp.'s" also collapses.
    s = re.sub(r"'s$|’s$", "", s)
    s = s.strip()

    # spaCy sometimes swallows a repeated mention into a single span
    # ("Vertex Labs, Vertex Labs" -> one ORG), which would otherwise become a
    # junk entity distinct from the real one. Collapse only exact repetitions:
    # a comma-separated name whose parts genuinely differ ("Springfield,
    # Illinois") is a real name and must survive untouched.
    parts = [p.strip() for p in s.split(",")]
    if len(parts) > 1 and all(p and p == parts[0] for p in parts):
        return parts[0]

    return s


class EpisodeCollisionError(RuntimeError):
    """Raised when an Episode id is already owned by a different session.

    Signals that `Episode.id` (a PostgreSQL row id) is not globally unique across
    whatever is writing to this graph — usually a shared/stale Neo4j instance.
    """


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

        `episode_id` is the PostgreSQL `episodes.id` returned by L2 `insert_episode`,
        and `Episode.id` carries a global uniqueness constraint. That makes the node
        key only as unique as the Postgres sequence: a truncated/restored L2, a second
        deployment sharing one graph, or a test run writing literal ids will otherwise
        silently adopt an unrelated session's episode along with its MENTIONS edges.
        Detect that and fail loudly rather than corrupt the graph.
        """
        q_owner = """
        MATCH (e:Episode {id: $episode_id})
        RETURN e.session_id AS owner
        """
        q_write = """
        MERGE (s:Session {id: $session_id})
        MERGE (e:Episode {id: $episode_id})
        SET e.summary = $summary, e.session_id = $session_id
        MERGE (s)-[:HAS_EPISODE]->(e)
        """
        with self._driver.session() as session:
            existing = session.run(q_owner, episode_id=episode_id).single()
            owner = existing["owner"] if existing is not None else None
            if owner is not None and owner != session_id:
                raise EpisodeCollisionError(
                    f"Episode id {episode_id} already belongs to session {owner!r}; "
                    f"refusing to re-link it to {session_id!r}. This usually means the "
                    f"Neo4j graph is shared with a different PostgreSQL instance."
                )
            session.run(
                q_write,
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
        MERGE (ep)-[m:MENTIONS]->(ent)
        ON CREATE SET m.count = 1
        ON MATCH  SET m.count = coalesce(m.count, 1) + 1
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
        MERGE (a)-[r:RELATED_TO]->(b)
        ON CREATE SET r.count = pair.n
        ON MATCH  SET r.count = coalesce(r.count, 1) + pair.n
        """
        # Canonicalize direction so (a,b) and (b,a) share one edge and one
        # counter, and fold repeats within this call into a single increment.
        tally: dict[tuple[str, str], int] = {}
        for raw_a, raw_b in pairs:
            na, nb = normalize_entity_name(raw_a), normalize_entity_name(raw_b)
            if not na or not nb or na == nb:
                continue
            key = (na, nb) if na <= nb else (nb, na)
            tally[key] = tally.get(key, 0) + 1
        payload = [{"a": a, "b": b, "n": n} for (a, b), n in tally.items()]
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

    def link_tool_calls(
        self,
        episode_id: int,
        calls: list[tuple[int, str, str, str]],
    ) -> None:
        """MERGE `(:Episode)-[:INVOKED]->(:ToolCall)` for claimed L4 calls.

        `calls` is (id, tool_name, status, at-iso) tuples. The node carries
        identity and outcome only — args and output stay in Postgres under the
        same id, so the graph can be traversed without ever carrying payloads.
        """
        if not calls:
            return
        q = """
        MATCH (ep:Episode {id: $episode_id})
        UNWIND $rows AS row
        MERGE (tc:ToolCall {id: row.id})
        SET tc.tool_name = row.tool_name, tc.status = row.status, tc.at = row.at
        MERGE (ep)-[:INVOKED]->(tc)
        """
        rows = [
            {"id": cid, "tool_name": name, "status": st, "at": at}
            for cid, name, st, at in calls
        ]
        with self._driver.session() as session:
            session.run(q, episode_id=episode_id, rows=rows)

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

    def fetch_neighborhood(
        self,
        entity_names: list[str],
        *,
        task_ids: list[str] | None = None,
        radius: int = 4,
    ):
        """Pull the subgraph within `radius` hops of the named entities and/or
        the given Task ids.

        One round-trip. `radius` is a fetch-size safety cap, not a semantic hop
        limit — activation spreading decides depth from weight; the cap only
        bounds how much is loaded. Node ids are `Label:key` strings so a
        neighborhood can be reasoned about without the driver; a Task's is
        `Task:<uuid>` via the `toString(id)` fallback.
        """
        from app.services.activation import Edge, Neighborhood

        names = [normalize_entity_name(n) for n in entity_names]
        names = [n for n in names if n]
        tids = [t for t in (task_ids or []) if t]
        if not names and not tids:
            return Neighborhood()
        r = max(1, min(int(radius), 6))
        q = f"""
        MATCH (seed)
        WHERE (seed:Entity AND seed.name IN $names)
           OR (seed:Task AND seed.id IN $task_ids)
        MATCH p = (seed)-[*1..{r}]-(other)
        UNWIND relationships(p) AS rel
        WITH DISTINCT rel, startNode(rel) AS a, endNode(rel) AS b
        RETURN labels(a)[0] AS la,
               coalesce(a.name, a.user_id, toString(a.id)) AS ka,
               type(rel) AS rel_type,
               coalesce(rel.count, 1) AS cnt,
               labels(b)[0] AS lb,
               coalesce(b.name, b.user_id, toString(b.id)) AS kb
        """
        edges: list = []
        labels: dict[str, str] = {}
        with self._driver.session() as session:
            for rec in session.run(q, names=names, task_ids=tids):
                src = f"{rec['la']}:{rec['ka']}"
                dst = f"{rec['lb']}:{rec['kb']}"
                labels[src] = rec["la"]
                labels[dst] = rec["lb"]
                edges.append(Edge(src, rec["rel_type"], dst, int(rec["cnt"])))
        return Neighborhood(edges=edges, labels=labels)

