"""Profile resolution: messages in, graph writes out."""

from __future__ import annotations

import uuid

import pytest
import spacy

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_extraction import resolve_profile_from_messages
from app.services.profile_store import ProfileStore, resolve_attributes

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nlp():
    return spacy.load(settings.spacy_model)


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def user_id(driver):
    uid = f"ur-{uuid.uuid4().hex[:12]}"
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
        s.run(
            "MATCH (se:Session) WHERE se.id STARTS WITH $uid DETACH DELETE se",
            uid=uid,
        )


def test_resolution_creates_profile_alias_and_attributes(driver, user_id, nlp):
    store = ProfileStore(driver)
    messages = [
        {"role": "user", "content": "I'm Dana Whitfield and I'm based in Seattle."},
        {"role": "assistant", "content": "Nice to meet you, Dana Whitfield."},
    ]

    result = resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=messages,
        nlp=nlp,
    )

    assert "dana whitfield" in result["aliases"]
    current = resolve_attributes(store.get_attributes(user_id))
    assert current["name"].value == "Dana Whitfield"
    assert current["location"].value == "Seattle"
    assert current["name"].source == "inferred"


def test_short_form_aliases_via_subset_rule(driver, user_id, nlp):
    store = ProfileStore(driver)
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=[{"role": "user", "content": "I'm Dana Whitfield."}],
        nlp=nlp,
    )
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s2",
        episode_id=None,
        messages=[{"role": "user", "content": "Dana shipped the release."}],
        nlp=nlp,
    )

    assert set(store.get_aliases(user_id)) >= {"dana whitfield", "dana"}


def test_assistant_messages_never_create_aliases(driver, user_id, nlp):
    store = ProfileStore(driver)
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=[{"role": "assistant", "content": "I'm Claude, your assistant."}],
        nlp=nlp,
    )
    assert store.get_aliases(user_id) == []
