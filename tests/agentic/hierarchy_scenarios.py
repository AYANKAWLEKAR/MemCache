"""Three-session goal-hierarchy scenarios with declared ground truth.

Same contract as `scenarios.py`: the model chooses the words, the scenario
chooses the facts. Each session is a plan (a Turn plus an optional failing tool
call recorded before ingest). Session 3 ends with a retrieval whose query
deliberately names none of the earlier facts — anything the agent learns, it
learned from the hierarchy.

`gate_*` fields are asserted. `metric_*` fields are printed under [metric].
Which is which was decided by the gate-promotion rule in the spec: a
behavioural claim gates only if it held on every trial during implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.agentic.ollama_agent import Turn


@dataclass(frozen=True)
class SessionPlan:
    turn: Turn
    tool_failure: dict | None = None   # {"tool_name","error","args"} recorded before ingest


@dataclass(frozen=True)
class HierarchyScenario:
    name: str
    sessions: list[SessionPlan]        # exactly three
    final_query: str
    #: substrings that MUST appear in S3's context (lowercased compare)
    gate_contains: list[str] = field(default_factory=list)
    #: substrings that MUST NOT appear in S3's context
    gate_absent: list[str] = field(default_factory=list)
    #: (child_title_fragment, parent_title_fragment) pairs — reported as [metric]
    metric_tree: list[tuple[str, str]] = field(default_factory=list)
    #: when non-empty, report every parented task (expected: none) under [metric]
    metric_roots: list[str] = field(default_factory=list)
    #: substrings reported (not asserted) in S3's context
    metric_contains: list[str] = field(default_factory=list)


ALEMBIC_FAILURE = {
    "tool_name": "alembic",
    "args": {"command": "upgrade", "revision": "0042"},
    "error": "DuplicateColumn: column user_id already exists on episodes",
}


TOP_DOWN = HierarchyScenario(
    name="top_down",
    sessions=[
        SessionPlan(Turn(
            intent="State plainly that your overall goal this quarter is to ship telemetry v2.",
            anchors=["ship telemetry v2"],
            fallback="My overall goal this quarter is to ship telemetry v2.",
        )),
        SessionPlan(Turn(
            intent=(
                "Say that as part of shipping telemetry v2 you are now migrating the "
                "telemetry schema to ClickHouse, and the alembic migration just failed."
            ),
            anchors=["part of shipping telemetry v2", "migrate the telemetry schema", "alembic"],
            fallback=(
                "As part of shipping telemetry v2 I need to migrate the telemetry schema "
                "to ClickHouse, and the alembic migration just failed."
            ),
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent=(
                "Say your goal right now is to fix the duplicate user_id column on the "
                "episodes table, a step of the schema migration."
            ),
            anchors=["fix the duplicate", "episodes", "step of the schema migration"],
            fallback=(
                "Right now my goal is to fix the duplicate user_id column on the episodes "
                "table, which is a step of the schema migration."
            ),
        )),
    ],
    final_query="Before I start, is there anything from earlier I should keep in mind?",
    gate_contains=["duplicatecolumn"],
    metric_tree=[("duplicate", "migrat"), ("migrat", "telemetry v2")],
    metric_contains=["telemetry v2", "under:"],
)


BOTTOM_UP = HierarchyScenario(
    name="bottom_up",
    sessions=[
        SessionPlan(Turn(
            intent=(
                "Say your goal is to fix the duplicate user_id column on the episodes "
                "table, and that the alembic upgrade just failed."
            ),
            anchors=["fix the duplicate", "episodes", "alembic"],
            fallback=(
                "My goal is to fix the duplicate user_id column on the episodes table; "
                "the alembic upgrade just failed."
            ),
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent=(
                "Say that the bigger goal, which the duplicate-column fix is one step of, "
                "is migrating the whole telemetry schema to ClickHouse."
            ),
            anchors=["bigger goal", "migrate", "telemetry schema"],
            fallback=(
                "The bigger goal here — the duplicate column fix is just one step of it — "
                "is to migrate the whole telemetry schema to ClickHouse."
            ),
        )),
        SessionPlan(Turn(
            intent="Say you are continuing the telemetry schema migration today.",
            anchors=["telemetry schema migration"],
            fallback="Continuing the telemetry schema migration today.",
        )),
    ],
    final_query="What should I watch out for?",
    gate_contains=["duplicatecolumn"],
    metric_tree=[("duplicate", "migrat")],
    metric_contains=["under:"],
)


UNRELATED_STAYS_FLAT = HierarchyScenario(
    name="unrelated_stays_flat",
    sessions=[
        SessionPlan(Turn(
            intent="State that your goal is to ship telemetry v2 and that an alembic migration failed.",
            anchors=["ship telemetry v2", "alembic"],
            fallback="My goal is to ship telemetry v2; an alembic migration failed today.",
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent="State that your goal is to plan the team offsite in Lisbon for October.",
            anchors=["plan the team offsite", "Lisbon"],
            fallback="My goal is to plan the team offsite in Lisbon for October.",
        )),
        SessionPlan(Turn(
            intent="Say you are continuing to plan the Lisbon offsite and looking at venues.",
            anchors=["Lisbon", "offsite"],
            fallback="Continuing to plan the Lisbon offsite; looking at venues.",
        )),
    ],
    final_query="Anything I should keep in mind?",
    # Precision: the telemetry goal must NOT be reported as an ancestor.
    gate_absent=["under: ship telemetry v2", "under: ship telemetry"],
    metric_roots=["telemetry", "offsite"],
)


SIBLING_SUBGOALS = HierarchyScenario(
    name="sibling_subgoals",
    sessions=[
        SessionPlan(Turn(
            intent="State plainly that your overall goal is to ship telemetry v2.",
            anchors=["ship telemetry v2"],
            fallback="My overall goal is to ship telemetry v2.",
        )),
        SessionPlan(Turn(
            intent=(
                "Say that one part of shipping telemetry v2 is migrating the telemetry "
                "schema with alembic, and it just failed."
            ),
            anchors=["part of shipping telemetry v2", "migrat", "alembic"],
            fallback=(
                "One part of shipping telemetry v2 is migrating the telemetry schema "
                "with alembic, and it just failed."
            ),
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent=(
                "Say that another part of shipping telemetry v2 is building the Grafana "
                "dashboards, and you are starting that now."
            ),
            anchors=["part of shipping telemetry v2", "Grafana dashboards"],
            fallback=(
                "Another part of shipping telemetry v2 is building the Grafana "
                "dashboards; starting that now."
            ),
        )),
    ],
    final_query="Anything I should know before I dig in?",
    gate_contains=[],
    metric_tree=[("migrat", "telemetry v2"), ("grafana", "telemetry v2")],
    metric_contains=["under:", "duplicatecolumn"],
)


HIERARCHY_SCENARIOS = [TOP_DOWN, BOTTOM_UP, UNRELATED_STAYS_FLAT, SIBLING_SUBGOALS]
