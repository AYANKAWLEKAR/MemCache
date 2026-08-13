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


def test_explicit_attribute_beats_inferred_regardless_of_recency(store, user_id):
    from app.services.profile_store import resolve_attributes

    store.upsert_profile(user_id)
    store.set_attribute(user_id, "title", "Staff Engineer", source="explicit", confidence=1.0)
    store.set_attribute(user_id, "title", "Intern", source="inferred", confidence=0.9)

    current = resolve_attributes(store.get_attributes(user_id))
    assert current["title"].value == "Staff Engineer"
    assert current["title"].source == "explicit"


def test_more_recent_inferred_value_supersedes_older(store, user_id):
    from app.services.profile_store import resolve_attributes

    store.upsert_profile(user_id)
    store.set_attribute(user_id, "location", "Boston", source="inferred", confidence=0.7)
    store.set_attribute(user_id, "location", "Seattle", source="inferred", confidence=0.7)

    current = resolve_attributes(store.get_attributes(user_id))
    assert current["location"].value == "Seattle"
    # History is retained, not overwritten.
    assert {r.value for r in store.get_attributes(user_id) if r.key == "location"} == {
        "Boston",
        "Seattle",
    }


def test_set_attribute_rejects_unknown_key(store, user_id):
    store.upsert_profile(user_id)
    with pytest.raises(ValueError, match="unknown attribute key"):
        store.set_attribute(user_id, "favourite_colour", "blue", source="explicit", confidence=1.0)


def test_set_attribute_rejects_unknown_source(store, user_id):
    store.upsert_profile(user_id)
    with pytest.raises(ValueError, match="unknown source"):
        store.set_attribute(user_id, "title", "Staff Engineer", source="guessed", confidence=1.0)


def test_reasserting_same_value_is_idempotent(store, user_id):
    store.upsert_profile(user_id)
    store.set_attribute(user_id, "role", "tech lead", source="inferred", confidence=0.6)
    store.set_attribute(user_id, "role", "tech lead", source="inferred", confidence=0.6)

    rows = [r for r in store.get_attributes(user_id) if r.key == "role"]
    assert len(rows) == 1
