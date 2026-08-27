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
    #: Which seeded session the retrieval continues, or None for a brand-new
    #: empty session. Most demos retrieve from a fresh session (pure
    #: cross-session memory); passing-mention retrieves from its last session
    #: because the offhand mention must be LIVE in L1 — that is what seeds the
    #: graph walk this demo exists to show.
    retrieve_from_session: int | None = None
    #: When non-empty, the seeded tasks are REPLACED by this fixed goal TREE:
    #: one `(title, parent_index | None)` per session, parents before their
    #: children. Measured on this branch, a 3B judge can neither build the
    #: tree nor reliably keep the goals distinct, and these demos show
    #: retrieval over the tree, not inference. Session i's episodes and tool
    #: calls map onto goal i; the LAST session's goal ends up most recently
    #: advanced, i.e. active.
    planted_goals: list[tuple[str, int | None]] = field(default_factory=list)

    @property
    def plant_hierarchy(self) -> bool:
        return bool(self.planted_goals)

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

_INVOICES_FAILURE = {
    "tool_name": "alembic",
    "status": "error",
    "args": {"command": "upgrade", "revision": "0117"},
    "error": "DuplicateColumn: column account_id already exists on invoices",
    "duration_ms": 388,
}

GOAL_HIERARCHY = Demo(
    key="goal-hierarchy",
    title="Goal hierarchy",
    blurb=(
        "Three sessions state a goal, a subgoal, and a sub-subgoal; the "
        "three Task nodes and their SUBGOAL_OF tree are planted by the demo "
        "(a 3B judge cannot build them — measured), while ingestion, "
        "retrieval, and ranking all "
        "run the live pipeline. Watch the Current-task path line and the "
        "parent goal's failure surface in the leaf's context. (Each demo uses "
        "its own project vocabulary — Entity nodes are shared graph-wide, so "
        "overlapping words would let one demo's graph light another's.)"
    ),
    sessions=[
        DemoSession(
            label="Two weeks ago — session A",
            messages=[
                {"role": "user", "content": "My overall goal this quarter is to ship the billing revamp."},
                {"role": "assistant", "content": "Understood — the billing revamp is the objective."},
            ],
        ),
        DemoSession(
            label="Last week — session B",
            messages=[
                {"role": "user", "content": (
                    "My goal is to migrate the invoices schema to Postgres 16, "
                    "and the alembic migration just failed."
                )},
                {"role": "assistant", "content": "Recorded the failed migration."},
            ],
            tool_failures=[_INVOICES_FAILURE],
        ),
        DemoSession(
            label="Yesterday — session C",
            messages=[
                {"role": "user", "content": (
                    "My goal is to fix the duplicate account_id column on the invoices table."
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
    planted_goals=[
        ("Ship the billing revamp", None),
        ("Migrate the invoices schema to Postgres 16", 0),
        ("Fix the duplicate account_id column on invoices", 1),
    ],
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
    retrieval_query=(
        "Catching up: what do you know about me, where I work, and how I like "
        "to run my meetings?"
    ),
    agent_question=(
        "Who are you speaking with, what did their team decide about the "
        "backend, and how should their standup update be run? "
        "Answer in two short sentences."
    ),
)

_KAFKA_FAILURE = {
    "tool_name": "terraform",
    "status": "error",
    "args": {"command": "apply", "workspace": "kafka-prod"},
    "error": "QuotaExceeded: cannot create more than 12 brokers in region us-east-1",
    "duration_ms": 2140,
}

PASSING_MENTION = Demo(
    key="passing-mention",
    title="Passing mention",
    blurb=(
        "Session A ties Kafka to a failed terraform apply. Session B only "
        "mentions Kafka offhand, and the retrieval query names nothing at all "
        "— the failure must arrive through the weighted graph, with its "
        "activation path shown in the table below."
    ),
    sessions=[
        DemoSession(
            label="Earlier — session A",
            messages=[
                {"role": "user", "content": (
                    "The terraform apply for the Kafka cluster just failed."
                )},
                {"role": "assistant", "content": "Logged the Kafka cluster failure."},
            ],
            tool_failures=[_KAFKA_FAILURE],
        ),
        DemoSession(
            label="Today — session B",
            messages=[
                {"role": "user", "content": "Also, Kafka consumer lag looked high yesterday."},
                {"role": "assistant", "content": "Noted about the consumer lag."},
            ],
        ),
    ],
    retrieval_query="anything else I should keep in mind before I continue?",
    agent_question=(
        "Anything the user should know before continuing their work? "
        "Answer in one short sentence."
    ),
    retrieve_from_session=1,
)

_PYTEST_FAILURE = {
    "tool_name": "pytest",
    "status": "error",
    "args": {"target": "tests/test_retrieval.py", "flags": "-q"},
    "error": (
        "AssertionError: semantic recall returned 0 results — similarity "
        "threshold 0.7 sits above the max achievable score for this embedder"
    ),
    "duration_ms": 3120,
}

RECRUITING_ROADMAP = Demo(
    key="recruiting-roadmap",
    title="Recruiting roadmap (student)",
    blurb=(
        "A student persona: five sessions across a semester state a recruiting "
        "goal and the subgoals serving it — classwork, DSA prep, side "
        "projects, and shipping MemCache itself (yes, this repo). The goal "
        "TREE is planted by the demo (measured: a 3B judge cannot build it); "
        "everything else runs the live pipeline. Watch the leaf's context "
        "carry the recruiting goal it ladders up to and the exact test "
        "failure not to repeat."
    ),
    sessions=[
        DemoSession(
            label="Three weeks ago — the goal",
            messages=[
                {"role": "user", "content": (
                    "My big goal this semester is recruiting: land a summer "
                    "software engineering internship."
                )},
                {"role": "assistant", "content": "Recruiting is the objective — noted."},
            ],
        ),
        DemoSession(
            label="Three weeks ago — classwork",
            messages=[
                {"role": "user", "content": (
                    "Part of that is keeping my grades solid — my goal is to "
                    "stay on top of classwork, especially CS 162."
                )},
                {"role": "assistant", "content": "Classwork tracked, with CS 162 flagged."},
            ],
        ),
        DemoSession(
            label="Two weeks ago — DSA prep",
            messages=[
                {"role": "user", "content": (
                    "Another piece: finish my DSA interview prep — I'm working "
                    "through the Leetcode Grind75 list."
                )},
                {"role": "assistant", "content": "Grind75 progress noted."},
            ],
        ),
        DemoSession(
            label="Two weeks ago — side projects",
            messages=[
                {"role": "user", "content": (
                    "The third piece is building side projects that make my "
                    "resume stand out."
                )},
                {"role": "assistant", "content": "Side projects — the portfolio pillar."},
            ],
        ),
        DemoSession(
            label="Yesterday — shipping MemCache",
            messages=[
                {"role": "user", "content": (
                    "My goal is to ship MemCache, my memory layer for LLM "
                    "agents — but the pytest suite just failed on retrieval."
                )},
                {"role": "assistant", "content": "Recorded the failing suite."},
            ],
            tool_failures=[_PYTEST_FAILURE],
        ),
    ],
    retrieval_query="Sitting down to work. Where was I, and what should I not waste time on?",
    agent_question=(
        "What are you working on right now, how does it ladder up to your "
        "recruiting goal, and what exactly broke last time? "
        "Answer in at most three short sentences."
    ),
    planted_goals=[
        ("Land a summer software engineering internship", None),
        ("Stay on top of classwork (CS 162)", 0),
        ("Finish DSA prep — Grind75", 0),
        ("Build standout side projects", 0),
        ("Ship MemCache, a memory layer for LLM agents", 3),
    ],
)

STUDENT_COMPANION = Demo(
    key="student-companion",
    title="Student companion",
    blurb=(
        "A student persona built on identity + a passing mention: session A "
        "introduces Ayan Kawlekar, a CS student at Berkeley, with a decision "
        "(target backend/infra roles) and a preference (working after 9pm). "
        "Session B drops one offhand line about an upcoming interview — and "
        "the retrieval query names nothing. The companion should know who it "
        "is helping and what is coming up."
    ),
    sessions=[
        DemoSession(
            label="Last month — session A",
            messages=[
                {"role": "user", "content": (
                    "Hi, I'm Ayan Kawlekar, a CS student at Berkeley."
                )},
                {"role": "assistant", "content": "Nice to meet you, Ayan."},
                {"role": "user", "content": (
                    "I decided to target backend and infrastructure roles for "
                    "my internship search, not frontend."
                )},
                {"role": "assistant", "content": "Backend and infra roles — noted."},
                {"role": "user", "content": (
                    "I prefer deep work after 9pm, so schedule anything "
                    "demanding at night."
                )},
                {"role": "assistant", "content": "Night owl schedule it is."},
            ],
        ),
        DemoSession(
            label="Today — session B",
            messages=[
                {"role": "user", "content": (
                    "Also, my behavioral interview with a fintech startup is "
                    "next Friday."
                )},
                {"role": "assistant", "content": "Good luck — noted for Friday."},
            ],
        ),
    ],
    retrieval_query="what should I be getting ready for?",
    agent_question=(
        "Who are you helping, what roles are they targeting, and what should "
        "they start preparing for — and when should you schedule the prep? "
        "Answer in two short sentences."
    ),
    retrieve_from_session=1,
)

DEMOS: list[Demo] = [
    FAILURE_RECALL,
    GOAL_HIERARCHY,
    IDENTITY_PREFERENCES,
    PASSING_MENTION,
    RECRUITING_ROADMAP,
    STUDENT_COMPANION,
]
