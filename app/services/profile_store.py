"""L3 UserProfile: canonical identity, attributes, aliases, profile-scoped edges.

The profile is an overlay. It links to existing `Entity` nodes rather than
replacing them, so an identity decision made from unreliable extraction stays
cheap to reverse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import Driver


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


def _stable_suffix(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
