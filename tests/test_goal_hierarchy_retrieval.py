"""Retrieval pulls context up the goal hierarchy. Hand-built graph, live stack.

Three effects, each asserted independently against a three-level tree
(leaf <- mid <- root) where only the ROOT has a failing tool call:
  1. `Current task:` shows the path.
  2. Known Failures surfaces the root's failure ranked by lineage.
  3. Proactive context surfaces the root's ToolCall via the lineage Task seed,
     with a provenance path that starts at the root Task node.
Plus severed tests proving each mechanism is the one doing the work.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.neo4j_store import Neo4jStore
from app.services.postgres_store import PostgresStore
from app.services.retrieval import retrieve_context
from app.services.task_store import TaskStore
from app.services.workbench_store import claim_tool_calls, ensure_l4_schema, record_tool_call
from tests.conftest import unit_embedding_384

pytestmark = pytest.mark.integration


@pytest.fixture
def stack():
    engine = create_engine_from_settings()
    ensure_l2_schema(engine)
    ensure_l4_schema(engine)
    driver = create_driver_from_settings()
    ensure_constraints(driver)
    yield engine, driver
    driver.close()
    engine.dispose()


@pytest.fixture
def tree(stack, monkeypatch):
    """root <- mid <- leaf. Root's episode invoked a failing `alembic` call.
    Live session has one L1 message with no NER entities at all, so the ONLY
    routes to the failure are the lineage (Known Failures) and the Task seeds
    (proactive)."""
    engine, driver = stack
    uid = f"gh-{uuid.uuid4().hex[:8]}"
    root_s, mid_s, leaf_s, live_s = (f"{uid}-{x}" for x in ("root", "mid", "leaf", "live"))
    now = datetime.now(timezone.utc)
    ts = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $u})", u=uid)
    root = ts.create_task(uid, "Ship telemetry v2")
    mid = ts.create_task(uid, "Migrate telemetry schema")
    leaf = ts.create_task(uid, "Fix duplicate column on episodes")
    ts.set_parent(mid, root)
    ts.set_parent(leaf, mid)

    def episode(sid, summary, axis):
        with session_scope(engine) as s:
            return PostgresStore(s).insert_episode(
                session_id=sid, summary=summary, embedding=unit_embedding_384(primary_axis=axis),
                start_time=now, end_time=now, metadata=None, user_id=uid,
            )

    e_root = episode(root_s, "Kicked off shipping telemetry v2.", 301)
    e_mid = episode(mid_s, "Started migrating the telemetry schema.", 302)
    e_leaf = episode(leaf_s, "Looking at the duplicate column.", 303)
    call = record_tool_call(
        engine, session_id=root_s, user_id=uid, task_id=root,
        tool_name="alembic", status="error",
        error="DuplicateColumn: user_id already exists",
    )
    claim_tool_calls(engine, session_id=root_s, episode_id=e_root, task_id=root)

    # An UNRELATED open goal of the same user, with its own failing call. It is
    # reachable from the leaf only through the UserProfile hub
    # (Task -PURSUES- Profile -PURSUES- Task); the hierarchy must not pull it.
    other_s = f"{uid}-other"
    other = ts.create_task(uid, "Plan the Lisbon offsite")
    e_other = episode(other_s, "Looking at venues in Lisbon.", 304)
    other_call = record_tool_call(
        engine, session_id=other_s, user_id=uid, task_id=other,
        tool_name="calendar", status="error", error="VenueUnavailable: all booked",
    )
    claim_tool_calls(engine, session_id=other_s, episode_id=e_other, task_id=other)

    g = Neo4jStore(driver)
    for sid, eid, summ in (
        (root_s, e_root, "root"), (mid_s, e_mid, "mid"), (leaf_s, e_leaf, "leaf"), (other_s, e_other, "other")
    ):
        g.upsert_session(sid)
        g.upsert_episode(sid, eid, summ)
    ts.link_episode(other, episode_id=e_other)
    ts.link_episode(root, episode_id=e_root)
    ts.link_episode(mid, episode_id=e_mid)
    ts.link_episode(leaf, episode_id=e_leaf)   # leaf is now active
    g.link_tool_calls(e_root, [(call.id, "alembic", "error", now.isoformat())])
    g.link_tool_calls(e_other, [(other_call.id, "calendar", "error", now.isoformat())])

    from app.api import services as api_services

    api_services.get_redis_store().append_messages(
        live_s, [{"role": "user", "content": "ok, picking this back up."}]
    )
    monkeypatch.setattr(
        "app.api.services.get_query_embedder",
        lambda: type("E", (), {"encode": lambda self, t, normalize_embeddings=True: type(
            "V", (), {"tolist": lambda s: unit_embedding_384(primary_axis=7)})()})(),
    )
    yield {"uid": uid, "live": live_s, "root": root, "mid": mid, "leaf": leaf,
           "call_id": call.id, "other": other, "other_call_id": other_call.id}

    api_services.get_redis_client().delete(f"session:{live_s}")
    with engine.begin() as c:
        c.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (uid,))
        c.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    with driver.session() as s:
        s.run("MATCH (p:UserProfile {user_id:$u}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) DETACH DELETE p, t", u=uid)
        for sid in (root_s, mid_s, leaf_s, other_s, live_s):
            s.run("MATCH (se:Session {id:$s}) OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep) "
                  "OPTIONAL MATCH (ep)-[:INVOKED]->(tc) DETACH DELETE se, ep, tc", s=sid)


def _retrieve(tree):
    return retrieve_context(tree["live"], "what should I keep in mind?", 2000, tree["uid"])


def test_current_task_line_shows_the_path(tree):
    r = _retrieve(tree)
    ctx = r["context"]
    assert (
        "Current task: Fix duplicate column on episodes "
        "(under: Migrate telemetry schema ▸ Ship telemetry v2)"
    ) in ctx
    (src,) = [s for s in r["sources"] if s["type"] == "task"]
    assert src["details"]["lineage"] == [tree["leaf"], tree["mid"], tree["root"]]
    assert src["details"]["depth"] == 2


def test_closed_ancestor_renders_done(tree, stack):
    _, driver = stack
    TaskStore(driver).close_task(tree["root"])
    ctx = _retrieve(tree)["context"]
    assert "Ship telemetry v2 (done)" in ctx


def test_known_failures_surface_the_roots_failure(tree):
    r = _retrieve(tree)
    fails = [s for s in r["sources"] if s["type"] == "tool_failure"]
    assert fails and fails[0]["details"]["task_id"] == tree["root"]
    assert "DuplicateColumn" in r["context"]


def test_proactive_surfaces_root_tool_call_via_lineage_seed(tree):
    r = _retrieve(tree)
    pf = [s for s in r["sources"] if s["type"] == "proactive_tool_failure"]
    assert pf, (
        "root's failure did not surface proactively; "
        f"types={sorted({s['type'] for s in r['sources']})}"
    )
    d = pf[0]["details"]
    assert d["tool_call_id"] == tree["call_id"]
    assert d["path"], "no provenance path"
    first_src = d["path"][0][0]
    assert first_src == f"Task:{tree['root']}", f"path should start at the root Task seed, got {first_src}"
    # Grandparent's tool call lands at 0.060 per the pinned §4c arithmetic.
    assert d["activation"] == pytest.approx(0.060, abs=0.003)


def test_severed_lineage_seeds_lose_the_proactive_failure(tree, monkeypatch):
    monkeypatch.setattr(settings, "proactive_task_node_seed", 0.0)
    r = _retrieve(tree)
    assert not [s for s in r["sources"] if s["type"] == "proactive_tool_failure"], (
        "with Task seeds off the failure must not surface proactively — something else carried it"
    )
    # ...but Known Failures (deterministic lineage) still has it.
    assert [s for s in r["sources"] if s["type"] == "tool_failure"]


def test_unrelated_goals_failure_does_not_surface_via_the_profile_hub(tree):
    """Precision. The offsite task is the same user's, one UserProfile hub away
    from the leaf (Task -PURSUES- Profile -PURSUES- Task). With PURSUES at its
    old 0.9 that route lit the offsite's failure to 0.06 — indistinguishable
    from a grandparent's. At 0.6 it dies. Known Failures still lists it (user
    scope, union), ranked last."""
    r = _retrieve(tree)
    pf = [s for s in r["sources"] if s["type"] == "proactive_tool_failure"]
    assert pf, "lineage failure must still surface"
    assert all(s["details"]["tool_call_id"] != tree["other_call_id"] for s in pf), (
        "unrelated goal's failure leaked into proactive context via the profile hub"
    )
    related_lines = [l for l in r["context"].splitlines() if l.startswith("Related episode")]
    assert not any("Lisbon" in l for l in related_lines), related_lines
    kf = [s["details"]["tool_call_id"] for s in r["sources"] if s["type"] == "tool_failure"]
    assert kf[0] == tree["call_id"] and tree["other_call_id"] in kf, kf


def test_severed_subgoal_prior_still_reaches_via_direct_seed(tree, monkeypatch):
    """Per-depth seeds start the fetch AND give every ancestor a floor of its
    own; with SUBGOAL_OF's prior zeroed the root's tool call still surfaces
    from its own seed (0.098·0.8·0.72 = 0.056 — distinguishable from the
    propagated 0.060), proving the seeds — not the edge — guarantee reach. The
    edge is what carries paths that must cross the tree (the agentic sibling
    scenario)."""
    from app.services import activation

    monkeypatch.setitem(activation.EDGE_PRIORS, "SUBGOAL_OF", 0.0)
    r = _retrieve(tree)
    pf = [s for s in r["sources"] if s["type"] == "proactive_tool_failure"]
    assert pf
    assert pf[0]["details"]["activation"] == pytest.approx(0.056, abs=0.003)
