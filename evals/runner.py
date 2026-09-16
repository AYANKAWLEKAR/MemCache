"""Orchestration for one eval run: seed once, score every condition.

The runner owns every side effect: the in-process API (eager Celery, the
pattern proven by frontend.demo_runtime), the prefix-scoped reset, turn
realization through the agentic harness, goal planting, and the answering
model. Seeding happens ONCE per repetition and the resulting RunRecord is
shared by all conditions, so every competitor works from identical content.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

from evals.conditions import (
    BuiltContext,
    FullTranscriptCondition,
    MemcacheCondition,
    NoMemoryCondition,
    RunRecord,
)
from evals.scenarios import ALL_SCENARIOS, EvalScenario, interleave
from evals.scoring import count_tokens, fact_coverage

logger = logging.getLogger(__name__)

#: Every eval user id starts with this. It extends the demo runtime's
#: `demo-ui-` prefix so the demo's scoped-reset safety assertion holds, while
#: staying disjoint from the actual demo users (`demo-ui-<demo key>`).
EVAL_USER_PREFIX = "demo-ui-eval-"


@dataclass(frozen=True)
class ProbeOutcome:
    """One (condition, probe) measurement within a repetition."""

    condition: str
    question: str
    answer: str
    answer_coverage: float
    missing_facts: tuple[str, ...]
    context_tokens: int
    latency_seconds: float
    #: memcache only: coverage of required facts in the retrieved context.
    #: 0.0 for an empty retrieval — a total miss is a measurement, not a gap.
    retrieval_recall: float | None = None
    #: memcache only: distinct tiers among provenance sources.
    tiers_used: tuple[str, ...] = ()
    #: Degradation the retrieval endpoint reported for this measurement.
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RepetitionResult:
    scenario: str
    history_size: int
    repetition: int
    seeded_sessions: int
    outcomes: tuple[ProbeOutcome, ...]
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_matrix(
    scenarios: list[EvalScenario],
    history_sizes: list[int],
    repetitions: int,
    progress_cb=None,
    on_result=None,
) -> list[RepetitionResult]:
    """The full grid. A failed repetition is recorded, never silently dropped.

    `on_result` receives each RepetitionResult as it completes, so a caller
    can persist incrementally during a multi-hour run.
    """
    results = []
    for scenario in scenarios:
        for size in history_sizes:
            for rep in range(repetitions):
                if progress_cb:
                    progress_cb(scenario.name, size, rep)
                try:
                    result = _run_repetition(scenario, size, rep)
                except Exception as exc:
                    logger.exception(
                        "repetition failed: %s h%d r%d", scenario.name, size, rep
                    )
                    result = RepetitionResult(
                        scenario=scenario.name,
                        history_size=size,
                        repetition=rep,
                        seeded_sessions=0,
                        outcomes=(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)
                if on_result:
                    on_result(result)
    return results


def _run_repetition(scenario: EvalScenario, size: int, rep: int) -> RepetitionResult:
    from frontend import demo_runtime as rt

    handle = rt.bootstrap()
    user_id = f"{EVAL_USER_PREFIX}{scenario.name}-h{size}-r{rep}"
    _reset_eval_data(scenario)

    run = _seed(handle, scenario, user_id, size)
    if scenario.planted_goals:
        _plant_goals(scenario, run)

    def retrieve_fn(session_id: str, uid: str, query: str, max_tokens: int) -> dict:
        resp = handle.client.post(
            "/memory/retrieve",
            headers=handle.auth,
            json={
                "session_id": session_id,
                "user_id": uid,
                "query": query,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()

    conditions = [
        MemcacheCondition(retrieve_fn=retrieve_fn),
        FullTranscriptCondition(),
        NoMemoryCondition(),
    ]
    # Rotate the measurement order per repetition so no condition always
    # answers first: model warm-up and cache effects would otherwise bias
    # the latency metric toward whichever condition ran last.
    shift = rep % len(conditions)
    ordered_conditions = conditions[shift:] + conditions[:shift]
    outcomes = []
    for probe_index, probe in enumerate(scenario.probes):
        for condition in ordered_conditions:
            outcomes.append(_measure(condition, run, probe, probe_index))
    return RepetitionResult(
        scenario=scenario.name,
        history_size=size,
        repetition=rep,
        seeded_sessions=len(run.placed),
        outcomes=tuple(outcomes),
    )


def _measure(condition, run: RunRecord, probe, probe_index: int) -> ProbeOutcome:
    from evals.answering import ask

    try:
        built: BuiltContext = condition.build_context(run, probe, probe_index)
        answer = ask(probe.question, built.context)
        coverage = fact_coverage(answer.text, list(probe.required_facts))
        recall = None
        tiers: tuple[str, ...] = ()
        if condition.name == "memcache":
            # Gated on the CONDITION, not on sources: an empty retrieval is
            # a recall of 0.0 — exactly the failure this metric must expose.
            recall = fact_coverage(
                built.context or "", list(probe.required_facts)
            ).fraction
            tiers = tuple(sorted({s.get("tier", "?") for s in built.sources}))
        return ProbeOutcome(
            condition=condition.name,
            question=probe.question,
            answer=answer.text,
            answer_coverage=coverage.fraction,
            missing_facts=coverage.missing,
            context_tokens=count_tokens(built.context or ""),
            latency_seconds=round(answer.seconds, 2),
            retrieval_recall=recall,
            tiers_used=tiers,
            warnings=built.warnings,
        )
    except Exception as exc:
        logger.exception("condition %s failed", condition.name)
        return ProbeOutcome(
            condition=condition.name,
            question=probe.question,
            answer="",
            answer_coverage=0.0,
            missing_facts=tuple(probe.required_facts),
            context_tokens=0,
            latency_seconds=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ----------------------------------------------------------------- seeding


def _seed(handle, scenario: EvalScenario, user_id: str, size: int) -> RunRecord:
    """Ingest the interleaved history through the real pipeline, oldest first."""
    from app.config import settings
    from tests.agentic.ollama_agent import OllamaAgent

    placed = interleave(scenario, size)
    agent = OllamaAgent(
        base_url=settings.ollama_base_url, model=settings.ollama_model
    )
    session_ids, realized = [], []
    for i, item in enumerate(placed):
        sid = f"{user_id}-s{i:03d}"
        session_ids.append(sid)
        if item.session.fixed_messages:
            messages = [dict(m) for m in item.session.fixed_messages]
        else:
            messages = []
            for turn in item.session.turns:
                messages.extend(agent.exchange(turn))
        for tf in item.session.tool_failures:
            resp = handle.client.post(
                "/workbench/tool-call",
                headers=handle.auth,
                json={"session_id": sid, "user_id": user_id, **tf},
            )
            resp.raise_for_status()
        resp = handle.client.post(
            "/memory/ingest",
            headers=handle.auth,
            json={"session_id": sid, "user_id": user_id, "messages": messages},
        )
        assert resp.status_code == 202, resp.text
        realized.append(tuple(messages))
    return RunRecord(
        user_id=user_id,
        placed=tuple(placed),
        realized_messages=tuple(realized),
        session_ids=tuple(session_ids),
    )


def _plant_goals(scenario: EvalScenario, run: RunRecord) -> None:
    """Deterministic goal tree: goal i belongs to fact session i.

    Same contract as the demo frontend's planting: the measured 3B judge
    cannot build the tree, so tree shape is fixed and everything downstream
    (linking, activation, lineage retrieval) runs live.
    """
    from app.api import services as api_services
    from app.services.task_store import TaskStore

    driver = api_services.get_neo4j_driver()
    store = TaskStore(driver)
    with driver.session() as s:
        s.run(
            "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task) DETACH DELETE t",
            uid=run.user_id,
        )
    task_ids = [store.create_task(run.user_id, title) for title, _ in scenario.planted_goals]
    engine = api_services.get_postgres_engine()
    for i, tid in enumerate(task_ids):
        sid = run.fact_session_id(i)
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                "SELECT id FROM episodes WHERE session_id = %s", (sid,)
            ).fetchall()
        for (eid,) in rows:
            store.link_episode(tid, episode_id=eid)
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE tool_calls SET task_id = %s WHERE session_id = %s", (tid, sid)
            )
    for i, (_, parent_idx) in enumerate(scenario.planted_goals):
        if parent_idx is not None:
            store.set_parent(task_ids[i], task_ids[parent_idx])


# ------------------------------------------------------------------ reset


def _reset_eval_data(scenario: EvalScenario) -> None:
    """Wipe ALL prior eval data (prefix-wide), not just this repetition's user.

    Two isolation hazards force this beyond the demo's per-user reset:
    aliases are permanent by design, so the previous repetition's profile
    would own this scenario's person names and the new repetition would
    silently lose its profile step; and Entity nodes are global, so
    RELATED_TO observation counts would accumulate across repetitions and
    inflate later runs.

    Two operations are deliberately wider than the eval prefix, both bounded:
    the alias release is restricted to eval profiles but names any Entity;
    and the orphan prune deletes Entity nodes left with no MENTIONS from any
    surviving episode and no alias, WHOEVER created them. An entity referenced
    by real data keeps its MENTIONS edge and is untouched; a truly orphaned
    non-eval entity is dead data either way. This is the only place the eval
    touches anything outside its own prefix.
    """
    import redis as redis_lib

    from app.api import services as api_services
    from app.config import settings

    prefix = EVAL_USER_PREFIX

    r = redis_lib.from_url(settings.redis_url, decode_responses=True)
    try:
        for key in r.scan_iter(f"session:{prefix}*"):
            r.delete(key)
    finally:
        r.close()

    with api_services.get_postgres_engine().begin() as conn:
        conn.exec_driver_sql(
            "DELETE FROM tool_calls WHERE user_id LIKE %s", (f"{prefix}%",)
        )
        conn.exec_driver_sql(
            "DELETE FROM episodes WHERE user_id LIKE %s", (f"{prefix}%",)
        )

    driver = api_services.get_neo4j_driver()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile) WHERE p.user_id STARTS WITH $prefix
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            DETACH DELETE p, t, a
            """,
            prefix=prefix,
        )
        s.run(
            """
            MATCH (se:Session) WHERE se.id STARTS WITH $prefix
            OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall)
            DETACH DELETE se, ep, dp, tc
            """,
            prefix=prefix,
        )
        for name in scenario.alias_names:
            s.run(
                """
                MATCH (p:UserProfile)-[r:HAS_ALIAS]->(:Entity {name: $n})
                WHERE p.user_id STARTS WITH $prefix
                DELETE r
                """,
                n=name,
                prefix=prefix,
            )
        s.run(
            """
            MATCH (e:Entity)
            WHERE NOT ()-[:MENTIONS]->(e) AND NOT ()-[:HAS_ALIAS]->(e)
            DETACH DELETE e
            """
        )


# ------------------------------------------------------------ health check


def preflight() -> list[str]:
    """Names of unhealthy dependencies with remediation, empty when ready."""
    from frontend import demo_runtime as rt

    problems = []
    for name, (ok, detail) in rt.stack_status().items():
        if not ok:
            problems.append(f"{name}: {detail}")
    return problems


def scenarios_by_name(names: list[str] | None) -> list[EvalScenario]:
    if not names:
        return list(ALL_SCENARIOS)
    by_name = {s.name: s for s in ALL_SCENARIOS}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise SystemExit(
            f"Unknown scenarios: {unknown}; available: {sorted(by_name)}"
        )
    return [by_name[n] for n in names]


def utcnow_stamp() -> str:
    return time.strftime("%Y-%m-%d")
