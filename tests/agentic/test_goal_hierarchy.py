"""Goal hierarchy under real traffic: three sessions, real Ollama for both the
agent's words AND MemCache's adjudication + placement, real stack.

Every test prints S3's retrieved context under [context], the user's task tree
under [tree], and the model's judgement calls under [metric], so the run can be
read qualitatively as well as pass/fail. Gates are the claims the deterministic
suite already proves the plumbing for; the tree shape is a metric.
"""

from __future__ import annotations

import pytest

from tests.agentic.hierarchy_scenarios import HIERARCHY_SCENARIOS, HierarchyScenario
from tests.agentic.ollama_agent import summarize_anchor_recall

pytestmark = [pytest.mark.agentic, pytest.mark.integration]


def _tree(driver, uid) -> list[tuple[str, str, str | None]]:
    """[(title, status, parent_title)] for every task the user pursues."""
    with driver.session() as s:
        return [
            (r["title"], r["status"], r["parent"])
            for r in s.run(
                "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task) "
                "OPTIONAL MATCH (t)-[:SUBGOAL_OF]->(p:Task) "
                "RETURN t.title AS title, t.status AS status, p.title AS parent, "
                "t.created_at AS created ORDER BY created",
                uid=uid,
            )
        ]


def _render_tree(rows) -> str:
    return "\n".join(f"  {t!r} ({st})  <- parent: {p!r}" for t, st, p in rows) or "  (no tasks)"


@pytest.fixture
def hierarchy_run(
    agent, api, auth, ingest, retrieve, profile_user_id, neo4j_driver, pg_engine, release_person_names
):
    """Play a scenario; return (context, sources, tree_rows). Cleans up all
    three sessions afterwards (the profile fixture removes the tasks)."""
    made: list[str] = []

    def _run(sc: HierarchyScenario):
        release_person_names("dana whitfield", "dana")
        agent.transcript.clear()
        uid = profile_user_id
        base = f"agentic-h-{uid[-6:]}"
        for i, plan in enumerate(sc.sessions, start=1):
            sid = f"{base}-s{i}"
            made.append(sid)
            if plan.tool_failure:
                r = api.post(
                    "/workbench/tool-call",
                    headers=auth,
                    json={"session_id": sid, "user_id": uid, "status": "error", **plan.tool_failure},
                )
                assert r.status_code == 201, r.text
            ingest(sid, agent.exchange(plan.turn), {"scenario": sc.name}, uid)
        result = retrieve(made[-1], sc.final_query, 2000, uid)
        rows = _tree(neo4j_driver, uid)
        recall = summarize_anchor_recall([p.turn for p in sc.sessions], agent.transcript)
        print(f"\n[{sc.name}] anchor_recall={recall:.2f}")
        print("[transcript]")
        for m in agent.transcript:
            if m["role"] == "user":
                print(f"  user: {m['content']}")
        print(f"[tree]\n{_render_tree(rows)}")
        print(f"[context]\n{result['context']}\n")
        return result["context"], result["sources"], rows

    yield _run

    import redis as _redis

    from app.config import settings as _settings

    c = _redis.from_url(_settings.redis_url, decode_responses=True)
    try:
        for sid in made:
            c.delete(f"session:{sid}")
    finally:
        c.close()
    with pg_engine.begin() as conn:
        for sid in made:
            conn.exec_driver_sql("DELETE FROM tool_calls WHERE session_id = %s", (sid,))
            conn.exec_driver_sql("DELETE FROM episodes WHERE session_id = %s", (sid,))
    with neo4j_driver.session() as s:
        for sid in made:
            s.run(
                "MATCH (se:Session {id: $sid}) OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode) "
                "OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp) OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall) "
                "DETACH DELETE se, ep, dp, tc",
                sid=sid,
            )


def _has_edge(rows, child_frag: str, parent_frag: str) -> bool:
    return any(
        child_frag in (t or "").lower() and parent_frag in (p or "").lower()
        for t, _, p in rows
    )


@pytest.mark.parametrize("scenario", HIERARCHY_SCENARIOS, ids=lambda s: s.name)
def test_hierarchy_scenario(scenario, hierarchy_run):
    context, sources, rows = hierarchy_run(scenario)
    low = context.lower()

    # ---- gates: behaviour the agent can rely on
    for frag in scenario.gate_contains:
        assert frag in low, f"[{scenario.name}] context missing {frag!r}"
    for frag in scenario.gate_absent:
        assert frag not in low, f"[{scenario.name}] context must not contain {frag!r} (false parent)"

    # ---- metrics: LLM judgement, reported never gated
    for child, parent in scenario.metric_tree:
        ok = _has_edge(rows, child, parent)
        print(f"[metric] {scenario.name}: edge {child!r} -SUBGOAL_OF-> {parent!r}: {'yes' if ok else 'NO'}")
    if scenario.metric_roots:
        parented = [(t, p) for t, _, p in rows if p is not None]
        print(f"[metric] {scenario.name}: parented tasks (expect none): {parented or 'none'}")
    for frag in scenario.metric_contains:
        print(f"[metric] {scenario.name}: context contains {frag!r}: {'yes' if frag in low else 'NO'}")
    task_src = [s for s in sources if s["type"] == "task"]
    depth = task_src[0]["details"].get("depth") if task_src else None
    print(f"[metric] {scenario.name}: active task depth={depth}")
    proactive_from_task = [
        s for s in sources
        if s["type"].startswith("proactive_") and s["details"].get("path")
        and str(s["details"]["path"][0][0]).startswith("Task:")
    ]
    print(f"[metric] {scenario.name}: proactive items carried by a Task seed: {len(proactive_from_task)}")
    for s in proactive_from_task:
        print(f"         {s['type']}: via {s['details'].get('via')}")
