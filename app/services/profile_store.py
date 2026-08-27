"""L3 UserProfile: canonical identity, attributes, aliases, profile-scoped edges.

The profile is an overlay. It links to existing `Entity` nodes rather than
replacing them, so an identity decision made from unreliable extraction stays
cheap to reverse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from neo4j import Driver

from app.services.neo4j_store import normalize_entity_name


class ProfileAliasConflictError(RuntimeError):
    """Raised when one Entity is claimed as an alias by two different profiles.

    Follows the `EpisodeCollisionError` precedent: identity ambiguity fails
    loudly rather than silently merging two people.
    """


@dataclass(frozen=True)
class ProfileRow:
    """Identity fields of a UserProfile node."""

    user_id: str
    display_name: str | None = None


#: The attribute keys this profile schema supports.
ATTRIBUTE_KEYS = frozenset({"name", "gender", "title", "role", "location"})

#: Where a value came from. Explicit statements always win over inference.
ATTRIBUTE_SOURCES = frozenset({"explicit", "inferred"})


@dataclass(frozen=True)
class AttributeRow:
    """One observed value for a profile attribute, with provenance."""

    key: str
    value: str
    source: str
    confidence: float
    observed_at: str
    evidence: str | None = None


def _rank(row: AttributeRow) -> tuple[int, str, float]:
    return (1 if row.source == "explicit" else 0, row.observed_at, row.confidence)


def resolve_attributes(rows: list[AttributeRow]) -> dict[str, AttributeRow]:
    """Reduce observation history to the current value per key.

    Precedence: explicit beats inferred, then most recent, then most confident.
    Keeping this a pure function over rows means there is exactly one place that
    decides what a profile currently says.
    """
    current: dict[str, AttributeRow] = {}
    for row in rows:
        incumbent = current.get(row.key)
        if incumbent is None or _rank(row) > _rank(incumbent):
            current[row.key] = row
    return current


def _stable_suffix(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ProfileStore:
    """Graph operations for the canonical user identity."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def upsert_profile(self, user_id: str, display_name: str | None = None) -> None:
        """MERGE the profile. A None `display_name` leaves any existing value intact."""
        q = """
        MERGE (p:UserProfile {user_id: $user_id})
        ON CREATE SET p.created_at = $now
        SET p.updated_at = $now,
            p.display_name = CASE
                WHEN $display_name IS NULL THEN p.display_name
                ELSE $display_name
            END
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, display_name=display_name, now=_now())

    def get_profile(self, user_id: str) -> ProfileRow | None:
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        RETURN p.user_id AS user_id, p.display_name AS display_name
        """
        with self._driver.session() as session:
            record = session.run(q, user_id=user_id).single()
        if record is None:
            return None
        return ProfileRow(
            user_id=record["user_id"],
            display_name=record["display_name"],
        )

    def set_attribute(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        source: str,
        confidence: float,
        evidence: str | None = None,
    ) -> None:
        """Record an observed attribute value. Re-asserting a value is idempotent."""
        if key not in ATTRIBUTE_KEYS:
            raise ValueError(
                f"unknown attribute key {key!r}; expected one of {sorted(ATTRIBUTE_KEYS)}"
            )
        if source not in ATTRIBUTE_SOURCES:
            raise ValueError(
                f"unknown source {source!r}; expected one of {sorted(ATTRIBUTE_SOURCES)}"
            )
        cleaned = value.strip()
        if not cleaned:
            return

        # Id keys on (profile, key, value) so the same assertion collapses while a
        # genuinely new value becomes a new node and history is preserved.
        attr_id = f"{user_id}:{key}:{_stable_suffix(cleaned)}"
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (a:ProfileAttribute {id: $attr_id})
        ON CREATE SET a.observed_at = $now
        SET a.key = $key,
            a.value = $value,
            a.source = $source,
            a.confidence = $confidence,
            a.evidence = $evidence
        MERGE (p)-[:HAS_ATTRIBUTE]->(a)
        """
        with self._driver.session() as session:
            session.run(
                q,
                user_id=user_id,
                attr_id=attr_id,
                key=key,
                value=cleaned,
                source=source,
                confidence=float(confidence),
                evidence=evidence,
                now=_now(),
            )

    def get_attributes(self, user_id: str) -> list[AttributeRow]:
        """All observed attribute values for a profile, oldest first."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
        RETURN a.key AS key, a.value AS value, a.source AS source,
               a.confidence AS confidence, a.observed_at AS observed_at,
               a.evidence AS evidence
        ORDER BY a.observed_at
        """
        with self._driver.session() as session:
            return [
                AttributeRow(
                    key=r["key"],
                    value=r["value"],
                    source=r["source"],
                    confidence=float(r["confidence"]),
                    observed_at=r["observed_at"],
                    evidence=r["evidence"],
                )
                for r in session.run(q, user_id=user_id)
            ]

    def link_alias(
        self,
        user_id: str,
        entity_name: str,
        *,
        source: str,
        confidence: float,
    ) -> str:
        """Alias an Entity to this profile. Returns the normalized name.

        Raises `ProfileAliasConflictError` when the entity is already aliased to
        a different profile — that means identity is ambiguous, and guessing is
        how two people get silently merged.
        """
        if source not in ATTRIBUTE_SOURCES:
            raise ValueError(
                f"unknown source {source!r}; expected one of {sorted(ATTRIBUTE_SOURCES)}"
            )
        norm = normalize_entity_name(entity_name)
        if not norm:
            raise ValueError(f"entity name {entity_name!r} normalizes to empty")

        q_owner = """
        MATCH (other:UserProfile)-[:HAS_ALIAS]->(:Entity {name: $name})
        WHERE other.user_id <> $user_id
        RETURN other.user_id AS owner
        LIMIT 1
        """
        q_link = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.display_name = $display_name
        MERGE (p)-[r:HAS_ALIAS]->(e)
        ON CREATE SET r.source = $source, r.confidence = $confidence, r.linked_at = $now
        """
        with self._driver.session() as session:
            conflict = session.run(q_owner, name=norm, user_id=user_id).single()
            if conflict is not None:
                raise ProfileAliasConflictError(
                    f"Entity {norm!r} is already an alias of profile "
                    f"{conflict['owner']!r}; refusing to also alias it to {user_id!r}."
                )
            session.run(
                q_link,
                user_id=user_id,
                name=norm,
                display_name=entity_name.strip(),
                source=source,
                confidence=float(confidence),
                now=_now(),
            )
        return norm

    def get_aliases(self, user_id: str) -> list[str]:
        """Normalized entity names aliased to this profile."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:HAS_ALIAS]->(e:Entity)
        RETURN e.name AS name ORDER BY name
        """
        with self._driver.session() as session:
            return [r["name"] for r in session.run(q, user_id=user_id)]

    def unlink_alias(self, user_id: str, entity_name: str) -> None:
        """Remove one alias edge. The Entity node itself is left intact."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[r:HAS_ALIAS]->(:Entity {name: $name})
        DELETE r
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, name=normalize_entity_name(entity_name))

    def link_session(self, user_id: str, session_id: str) -> None:
        """Record that this profile participated in a session."""
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (s:Session {id: $session_id})
        MERGE (p)-[:PARTICIPATED_IN]->(s)
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, session_id=session_id)

    def promote_episode_facts(self, user_id: str, episode_id: int) -> None:
        """Attach an episode's decisions and preferences to the profile.

        The `(:Episode)-[:DECIDED]->` edges stay as provenance; these additional
        profile edges make "everything this person has ever decided" a one-hop
        query spanning every session.
        """
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MATCH (ep:Episode {id: $episode_id})
        OPTIONAL MATCH (ep)-[:DECIDED]->(d:Decision)
        OPTIONAL MATCH (ep)-[:PREFERS]->(pr:Preference)
        FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
            MERGE (p)-[:DECIDED]->(d))
        FOREACH (_ IN CASE WHEN pr IS NULL THEN [] ELSE [1] END |
            MERGE (p)-[:PREFERS]->(pr))
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, episode_id=episode_id)

    def get_profile_decisions(self, user_id: str) -> list[str]:
        """Decision texts attached directly to the profile, across all sessions."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:DECIDED]->(d:Decision)
        RETURN DISTINCT d.text AS text ORDER BY text
        """
        with self._driver.session() as session:
            return [r["text"] for r in session.run(q, user_id=user_id) if r["text"]]

    def get_profile_preferences(self, user_id: str) -> list[str]:
        """Preference texts attached directly to the profile, across all sessions."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:PREFERS]->(p:Preference)
        RETURN DISTINCT p.text AS text ORDER BY text
        """
        with self._driver.session() as session:
            return [r["text"] for r in session.run(q, user_id=user_id) if r["text"]]
