"""L3 UserProfile: identity node, attributes, aliases, profile-scoped edges."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_store import ProfileRow, ProfileStore

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def user_id(driver):
    """Unique profile id, removed with everything it owns afterwards."""
    uid = f"u-{uuid.uuid4().hex[:12]}"
    yield uid
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )


@pytest.fixture
def store(driver):
    return ProfileStore(driver)


def test_upsert_profile_is_idempotent(store, user_id):
    store.upsert_profile(user_id, display_name="Dana Whitfield")
    store.upsert_profile(user_id, display_name="Dana Whitfield")

    row = store.get_profile(user_id)
    assert row == ProfileRow(user_id=user_id, display_name="Dana Whitfield")


def test_get_profile_returns_none_when_absent(store):
    assert store.get_profile("does-not-exist") is None


def test_upsert_profile_preserves_display_name_when_omitted(store, user_id):
    store.upsert_profile(user_id, display_name="Dana Whitfield")
    store.upsert_profile(user_id)

    row = store.get_profile(user_id)
    assert row is not None
    assert row.display_name == "Dana Whitfield"
