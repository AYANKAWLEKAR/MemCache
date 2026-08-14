"""Neo4j driver factory and idempotent schema constraints for L3 graph store."""

from __future__ import annotations

from neo4j import Driver, GraphDatabase

from app.config import settings


def create_driver_from_settings(
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> Driver:
    """Bolt driver with pool pre-ping behavior suitable for app and tests."""
    return GraphDatabase.driver(
        uri or settings.neo4j_uri,
        auth=(user or settings.neo4j_user, password or settings.neo4j_password),
    )


def ensure_constraints(driver: Driver) -> None:
    """Create uniqueness constraints for Session, Episode, Entity, and UserProfile if missing.

    Idempotent (IF NOT EXISTS). Safe to call on every app startup.
    Decision/Preference nodes use composite `id` strings set at write time.
    """
    stmts = [
        "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT preference_id_unique IF NOT EXISTS FOR (p:Preference) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT user_profile_id_unique IF NOT EXISTS FOR (p:UserProfile) REQUIRE p.user_id IS UNIQUE",
        "CREATE CONSTRAINT profile_attribute_id_unique IF NOT EXISTS FOR (a:ProfileAttribute) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT tool_call_id_unique IF NOT EXISTS FOR (tc:ToolCall) REQUIRE tc.id IS UNIQUE",
    ]
    with driver.session() as session:
        for cypher in stmts:
            session.run(cypher)
