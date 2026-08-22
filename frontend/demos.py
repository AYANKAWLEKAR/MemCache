"""The four preset demos: scripted synthetic conversations + seeded failures.

Pure data. Conversations are scripted, not model-generated — this surface
shows MemCache, not traffic realism (the agentic harness covers that). Each
demo's `blurb` is user-facing copy and states honestly what is scripted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DemoSession:
    label: str
    messages: list[dict[str, str]]
    tool_failures: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    blurb: str
    sessions: list[DemoSession]
    retrieval_query: str
    agent_question: str
    plant_hierarchy: bool = False

    @property
    def user_id(self) -> str:
        return f"demo-ui-{self.key}"

    def session_id(self, index: int) -> str:
        return f"{self.user_id}-s{index + 1}"


_ALEMBIC_FAILURE = {
    "tool_name": "alembic",
    "status": "error",
    "args": {"command": "upgrade", "revision": "0042"},
    "error": "DuplicateColumn: column user_id already exists on episodes",
    "duration_ms": 412,
}

FAILURE_RECALL = Demo(
    key="failure-recall",
    title="Failure recall",
    blurb=(
        "Monday, session A: a migration fails (scripted, recorded to the L4 "
        "workbench for real) and the conversation is ingested. Thursday, a "
        "brand-new session resumes the work. Watch the with-memory agent "
        "address the exact DuplicateColumn error while the other starts blind."
    ),
    sessions=[
        DemoSession(
            label="Monday — session A",
            messages=[
                {"role": "user", "content": (
                    "I'm trying to migrate the telemetry schema to ClickHouse, "
                    "but the alembic migration just failed."
                )},
                {"role": "assistant", "content": "Noted — the migration errored."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
    ],
    retrieval_query="Picking the telemetry migration back up. What should I know?",
    agent_question=(
        "You are resuming work on the telemetry schema migration. "
        "What is your very first action, and why? Answer in one short sentence."
    ),
)

GOAL_HIERARCHY = Demo(
    key="goal-hierarchy",
    title="Goal hierarchy",
    blurb=(
        "Three sessions state a goal, a subgoal, and a sub-subgoal; the "
        "SUBGOAL_OF tree is planted by the demo (a 3B judge cannot infer "
        "direction — measured), while ingestion, retrieval, and ranking all "
        "run the live pipeline. Watch the Current-task path line and the "
        "parent goal's failure surface in the leaf's context."
    ),
    sessions=[
        DemoSession(
            label="Two weeks ago — session A",
            messages=[
                {"role": "user", "content": "My overall goal this quarter is to ship telemetry v2."},
                {"role": "assistant", "content": "Understood — telemetry v2 is the objective."},
            ],
        ),
        DemoSession(
            label="Last week — session B",
            messages=[
                {"role": "user", "content": (
                    "My goal is to migrate the telemetry schema to ClickHouse, "
                    "and the alembic migration just failed."
                )},
                {"role": "assistant", "content": "Recorded the failed migration."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
        DemoSession(
            label="Yesterday — session C",
            messages=[
                {"role": "user", "content": (
                    "My goal is to fix the duplicate user_id column on the episodes table."
                )},
                {"role": "assistant", "content": "On it — the duplicate column fix."},
            ],
        ),
    ],
    retrieval_query="Getting back to work. Where was I and what should I avoid?",
    agent_question=(
        "What are you working on right now, what larger goal does it serve, "
        "and what must you not repeat? Answer in at most three short sentences."
    ),
    plant_hierarchy=True,
)

IDENTITY_PREFERENCES = Demo(
    key="identity-preferences",
    title="Identity & preferences",
    blurb=(
        "Session A introduces Dana Whitfield of Northwind Robotics, a decision "
        "(Rust) and a preference (async standups). A fresh session then has to "
        "know who it is talking to — aliases collapse to one profile."
    ),
    sessions=[
        DemoSession(
            label="Session A",
            messages=[
                {"role": "user", "content": (
                    "Hi, I'm Dana Whitfield and I work at Northwind Robotics."
                )},
                {"role": "assistant", "content": "Nice to meet you, Dana."},
                {"role": "user", "content": "We decided to use Rust for the control backend."},
                {"role": "assistant", "content": "Rust for the backend — noted."},
                {"role": "user", "content": "I prefer async standups over daily video calls."},
                {"role": "assistant", "content": "Async standups it is."},
            ],
        ),
    ],
    retrieval_query="Who am I and how do we work together?",
    agent_question=(
        "Who are you speaking with, where do they work, and how should their "
        "standup update be run? Answer in two short sentences."
    ),
)

PASSING_MENTION = Demo(
    key="passing-mention",
    title="Passing mention",
    blurb=(
        "Session A ties ClickHouse to a failed alembic migration. Session B "
        "only mentions ClickHouse offhand, and the retrieval query names "
        "nothing at all — the failure must arrive through the weighted graph, "
        "with its activation path shown in the table below."
    ),
    sessions=[
        DemoSession(
            label="Earlier — session A",
            messages=[
                {"role": "user", "content": (
                    "The alembic migration for the ClickHouse telemetry schema just failed."
                )},
                {"role": "assistant", "content": "Logged the ClickHouse migration failure."},
            ],
            tool_failures=[_ALEMBIC_FAILURE],
        ),
        DemoSession(
            label="Today — session B",
            messages=[
                {"role": "user", "content": "Also, ClickHouse ingest looked slow yesterday."},
                {"role": "assistant", "content": "Noted about the ingest speed."},
            ],
        ),
    ],
    retrieval_query="anything else I should keep in mind before I continue?",
    agent_question=(
        "Anything the user should know before continuing their work? "
        "Answer in one short sentence."
    ),
)

DEMOS: list[Demo] = [FAILURE_RECALL, GOAL_HIERARCHY, IDENTITY_PREFERENCES, PASSING_MENTION]
