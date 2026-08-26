"""user_id threading: optional field, profile built on ingest."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.main import app
from app.services.profile_store import ProfileStore

pytestmark = pytest.mark.integration

AUTH = {"X-API-Key": "dummy-api-key-123"}


@pytest.fixture(scope="module", autouse=True)
def _eager():
    from app.workers.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


def test_ingest_without_user_id_still_works(client):
    """The field is optional; omitting it must not change behaviour."""
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": f"nouser-{uuid.uuid4().hex[:8]}",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 202


def test_ingest_with_user_id_builds_profile(client, driver):
    user_id = f"api-{uuid.uuid4().hex[:10]}"
    session_id = f"{user_id}-s1"
    try:
        response = client.post(
            "/memory/ingest",
            headers=AUTH,
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I'm Dana Whitfield and I'm based in Seattle.",
                    },
                    {"role": "assistant", "content": "Good to meet you."},
                ],
            },
        )
        assert response.status_code == 202

        store = ProfileStore(driver)
        assert store.get_profile(user_id) is not None
        assert "dana whitfield" in store.get_aliases(user_id)
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (p:UserProfile {user_id: $uid})
                OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
                OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
                DETACH DELETE p, a, e
                """,
                uid=user_id,
            )
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_id)


def _purge(driver, user_id, session_id=None):
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            DETACH DELETE p, a, e, t
            """,
            uid=user_id,
        )
        if session_id:
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_id)


def test_profile_endpoints_roundtrip(client, driver):
    user_id = f"ep-{uuid.uuid4().hex[:10]}"
    try:
        assert client.get(f"/profile/{user_id}", headers=AUTH).status_code == 404

        patch = client.patch(
            f"/profile/{user_id}",
            headers=AUTH,
            json={"attributes": {"title": "Principal Engineer", "location": "Seattle"}},
        )
        assert patch.status_code == 200

        alias = client.post(
            f"/profile/{user_id}/alias",
            headers=AUTH,
            json={"entity_name": "Dana Whitfield"},
        )
        assert alias.status_code == 200
        assert alias.json()["aliases"] == ["dana whitfield"]

        body = client.get(f"/profile/{user_id}", headers=AUTH).json()
        assert body["user_id"] == user_id
        assert body["attributes"]["title"]["value"] == "Principal Engineer"
        assert body["attributes"]["title"]["source"] == "explicit"
        assert body["aliases"] == ["dana whitfield"]
    finally:
        _purge(driver, user_id)


def test_explicit_patch_overrides_inference(client, driver):
    user_id = f"ov-{uuid.uuid4().hex[:10]}"
    session_id = f"{user_id}-s1"
    try:
        client.post(
            "/memory/ingest",
            headers=AUTH,
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [{"role": "user", "content": "I'm based in Boston."}],
            },
        )
        client.patch(
            f"/profile/{user_id}",
            headers=AUTH,
            json={"attributes": {"location": "Seattle"}},
        )
        body = client.get(f"/profile/{user_id}", headers=AUTH).json()
        assert body["attributes"]["location"]["value"] == "Seattle"
        assert body["attributes"]["location"]["source"] == "explicit"
    finally:
        _purge(driver, user_id, session_id)


def test_alias_conflict_returns_409(client, driver):
    first = f"c1-{uuid.uuid4().hex[:10]}"
    second = f"c2-{uuid.uuid4().hex[:10]}"
    try:
        client.patch(f"/profile/{first}", headers=AUTH, json={"attributes": {}})
        client.patch(f"/profile/{second}", headers=AUTH, json={"attributes": {}})
        client.post(
            f"/profile/{first}/alias", headers=AUTH, json={"entity_name": "Dana Whitfield"}
        )

        clash = client.post(
            f"/profile/{second}/alias", headers=AUTH, json={"entity_name": "Dana Whitfield"}
        )
        assert clash.status_code == 409
    finally:
        _purge(driver, first)
        _purge(driver, second)
