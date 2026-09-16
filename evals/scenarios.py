"""Eval scenarios: fact-bearing sessions, probes, and distractor history.

Fact-bearing sessions reuse the agentic harness contract (`Turn`): the model
improvises the wording, mandatory anchors survive or the turn falls back to a
deterministic template. Distractor sessions are fixed text and never touch a
model. The four families together exercise every tier: L1 (the passing-mention
probe continues a live session), L2 (episodes in every family), L3 (profile,
decisions, preferences, entities, and a planted goal tree), and L4 (recorded
tool failures).

Fairness is enforced by unit test, not convention: no probe question contains
a required fact or an anchor, and no distractor session contains any
scenario's required fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.agentic.ollama_agent import Turn


@dataclass(frozen=True)
class EvalProbe:
    """One question asked after ingestion, with its deterministic ground truth."""

    question: str
    #: Substrings a correct answer must contain (case-insensitive, whitespace
    #: normalized). Only unambiguous strings: error fragments, names, terms.
    required_facts: tuple[str, ...]
    #: None: probe from a fresh session (context is purely cross-session
    #: memory). int: continue that fact-bearing session, for probes whose
    #: mechanism needs the live recent turns (the passing-mention family).
    probe_from_session: int | None = None


@dataclass(frozen=True)
class EvalSession:
    """One conversation to ingest.

    Either `turns` (model-improvised around anchors, harness style) or
    `fixed_messages` (used verbatim, no model) — never both.
    """

    label: str
    turns: tuple[Turn, ...] = ()
    fixed_messages: tuple[dict, ...] = ()
    #: Recorded to L4 via /workbench/tool-call before the session is ingested.
    tool_failures: tuple[dict, ...] = ()

    def __post_init__(self):
        assert bool(self.turns) != bool(self.fixed_messages), self.label


@dataclass(frozen=True)
class EvalScenario:
    """A family: ordered fact-bearing sessions plus probes."""

    name: str
    sessions: tuple[EvalSession, ...]
    probes: tuple[EvalProbe, ...]
    #: (title, parent index | None), parents before children; one goal per
    #: fact-bearing session, planted deterministically by the runner (the
    #: measured 3B limitation — a small judge cannot build the tree).
    planted_goals: tuple[tuple[str, int | None], ...] = ()
    #: Normalized person names the scenario introduces. Alias links are
    #: permanent by design (conflicts 409 rather than guess), so the runner
    #: releases these before each repetition — a new repetition's user would
    #: otherwise silently lose its profile to the previous repetition's.
    alias_names: tuple[str, ...] = ()

    def all_required_facts(self) -> list[str]:
        return [f for p in self.probes for f in p.required_facts]


# --------------------------------------------------------------- families

FAILURE_RECALL = EvalScenario(
    name="failure_recall",
    sessions=(
        EvalSession(
            label="starting the pipeline project",
            turns=(
                Turn(
                    intent="Say you are building a data pipeline for Project Meridian "
                    "and chose Apache Airflow to orchestrate it.",
                    anchors=["Project Meridian", "Apache Airflow"],
                    fallback="I'm building the data pipeline for Project Meridian "
                    "and chose Apache Airflow to orchestrate it.",
                ),
            ),
        ),
        EvalSession(
            label="the build breaks",
            turns=(
                Turn(
                    intent="Report that the dbt build for the pipeline just failed "
                    "and you had to stop for the day.",
                    anchors=["dbt build", "failed"],
                    fallback="The dbt build for the pipeline just failed, "
                    "so I'm stopping for the day.",
                ),
            ),
            tool_failures=(
                {
                    "tool_name": "dbt",
                    "status": "error",
                    "args": {"command": "dbt build", "project": "meridian"},
                    "error": "CompilationError: model orders_daily references "
                    "undefined column customer_tier",
                    "duration_ms": 4210,
                },
            ),
        ),
    ),
    probes=(
        EvalProbe(
            question="I'm sitting back down to work on my data pipeline. "
            "What exactly broke last time, and what should I fix first?",
            required_facts=("orders_daily", "customer_tier"),
        ),
    ),
)


IDENTITY_PREFERENCES = EvalScenario(
    name="identity_preferences",
    sessions=(
        EvalSession(
            label="introduction",
            turns=(
                Turn(
                    intent="Introduce yourself by name and employer.",
                    anchors=["Marta Okafor", "Helix Dynamics"],
                    fallback="Hi, I'm Marta Okafor and I work at Helix Dynamics.",
                ),
            ),
        ),
        EvalSession(
            label="decision and preference",
            turns=(
                Turn(
                    intent="Say which database your team settled on for the ledger "
                    "service.",
                    anchors=["decided to use PostgreSQL"],
                    fallback="We decided to use PostgreSQL for the ledger service.",
                ),
                Turn(
                    intent="State a working-style preference about when you take "
                    "code reviews.",
                    anchors=["prefer code reviews before noon"],
                    fallback="I prefer code reviews before noon, while I'm fresh.",
                ),
            ),
        ),
    ),
    probes=(
        EvalProbe(
            question="What did we settle on for the storage layer, and when do I "
            "like review work scheduled?",
            required_facts=("postgresql", "before noon"),
        ),
    ),
    alias_names=("marta okafor", "marta"),
)


GOAL_LINEAGE = EvalScenario(
    name="goal_lineage",
    sessions=(
        EvalSession(
            label="the root goal",
            turns=(
                Turn(
                    intent="State your big objective this quarter: launching the "
                    "analytics platform beta.",
                    anchors=["analytics platform beta"],
                    fallback="My big objective this quarter is launching the "
                    "analytics platform beta.",
                ),
            ),
        ),
        EvalSession(
            label="the middle goal",
            turns=(
                Turn(
                    intent="Say that part of that effort is migrating the reporting "
                    "warehouse.",
                    anchors=["migrating the reporting warehouse"],
                    fallback="Part of that effort is migrating the reporting "
                    "warehouse.",
                ),
            ),
        ),
        EvalSession(
            label="the leaf goal",
            turns=(
                Turn(
                    intent="Say you are currently fixing the nightly aggregation "
                    "job that keeps timing out.",
                    anchors=["nightly aggregation job"],
                    fallback="Right now I'm fixing the nightly aggregation job "
                    "that keeps timing out.",
                ),
            ),
        ),
    ),
    probes=(
        EvalProbe(
            question="What task am I in the middle of, and what larger goal is "
            "it ultimately serving?",
            required_facts=("nightly aggregation", "analytics platform beta"),
        ),
    ),
    planted_goals=(
        ("Launch the analytics platform beta", None),
        ("Migrate the reporting warehouse", 0),
        ("Fix the nightly aggregation job", 1),
    ),
)


PASSING_MENTION = EvalScenario(
    name="passing_mention",
    sessions=(
        EvalSession(
            label="weeks ago: the consumer incident",
            turns=(
                Turn(
                    intent="Say you spent the day debugging the Kafka consumers "
                    "for the events service.",
                    anchors=["Kafka consumers"],
                    fallback="I spent all day debugging the Kafka consumers for "
                    "the events service.",
                ),
            ),
            tool_failures=(
                {
                    "tool_name": "kafka-consumer",
                    "status": "error",
                    "args": {"group": "events-service", "partition": 3},
                    "error": "OffsetOutOfRangeError: partition 3 reset to earliest, "
                    "replaying 2.1M events",
                    "duration_ms": 950,
                },
            ),
        ),
        EvalSession(
            label="today: a different task, one offhand line",
            turns=(
                Turn(
                    intent="Say you are writing the on-call handbook today, and "
                    "mention in passing that you'll also touch the Kafka "
                    "consumers again.",
                    anchors=["on-call handbook", "Kafka consumers"],
                    fallback="Today I'm writing the on-call handbook, and I'll "
                    "also touch the Kafka consumers again.",
                ),
            ),
        ),
    ),
    probes=(
        EvalProbe(
            question="Anything else I should keep in mind before I start?",
            required_facts=("offsetoutofrange",),
            probe_from_session=1,
        ),
    ),
)


ALL_SCENARIOS: tuple[EvalScenario, ...] = (
    FAILURE_RECALL,
    IDENTITY_PREFERENCES,
    GOAL_LINEAGE,
    PASSING_MENTION,
)


# ------------------------------------------------------------- distractors

#: Topic templates for distractor sessions: realistic personal-agent exchanges
#: (scheduling, debugging help, drafting, planning). Each template is 4-6
#: user/assistant exchanges (8-12 messages). Slots {a}/{b} are filled from the
#: fixed lists below, rotated by session index, so any count of sessions is
#: deterministic and reasonably varied. None of this text may contain any
#: scenario's required facts — enforced by test_eval_scenarios.
_FILL_A = ("Rachel", "Tomas", "Yuki", "Omar", "Ingrid", "Wei", "Paulo")
_FILL_B = ("Lisbon", "Oslo", "Kyoto", "Austin", "Porto", "Zurich", "Nairobi")

_DISTRACTOR_TEMPLATES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("trip planning", (
        ("I need to plan the {b} trip for the offsite in March.",
         "Noted. Do you want flights first or the venue?"),
        ("Venue first, somewhere walkable from the station.",
         "I'll shortlist three options near the station."),
        ("Also remind me to renew my passport before booking.",
         "Reminder set for the passport renewal."),
        ("Budget cap is what we agreed with finance last quarter.",
         "Understood, I'll keep the shortlist under that cap."),
    )),
    ("expense report", (
        ("I still owe the expense report from the {b} conference.",
         "You have twelve receipts uploaded so far."),
        ("The taxi receipts are missing, I'll photograph them tonight.",
         "I'll hold the draft open until they arrive."),
        ("File it under the travel category, not team events.",
         "Recategorized to travel."),
        ("Submit it as soon as the taxi receipts are in.",
         "Will do, it goes out the moment they land."),
    )),
    ("meeting prep", (
        ("Help me prep the Thursday sync with {a}.",
         "Last week's notes list two open items."),
        ("Add the hiring update as a third agenda item.",
         "Added. Anything to drop?"),
        ("Drop the venue discussion, that's settled now.",
         "Dropped. The agenda has three items."),
        ("Send the agenda out an hour before the meeting.",
         "Scheduled to send one hour prior."),
    )),
    ("gift ideas", (
        ("I need a gift idea for {a}'s farewell next week.",
         "They mentioned enjoying trail running and coffee."),
        ("Coffee then, something from a local roaster.",
         "There are two well-reviewed roasters nearby."),
        ("Get a card the whole team can sign too.",
         "Card added to the list."),
        ("Wrap it before Friday so it doesn't slip.",
         "Reminder set for Thursday evening."),
    )),
    ("apartment errands", (
        ("The landlord finally replied about the radiator repair.",
         "Good. Do you want to confirm the proposed slot?"),
        ("Yes, Tuesday morning works if I move my standup with {a}.",
         "Standup moved, slot confirmed."),
        ("Also add lightbulbs and filters to the shopping list.",
         "Both added."),
        ("Remind me to be home by nine that day.",
         "Reminder set for Tuesday at nine."),
    )),
    ("reading list", (
        ("Add the distributed systems paper {a} recommended to my list.",
         "Added. Your queue has six items now."),
        ("Move it to the top, I want it before the reading group.",
         "Moved to the top."),
        ("Archive the two I finished last month.",
         "Archived. Four items remain."),
        ("Block Sunday afternoon for reading.",
         "Sunday afternoon blocked."),
    )),
    ("laptop trouble", (
        ("My laptop fan is roaring again during video calls.",
         "Last time clearing the browser tabs helped."),
        ("It's worse now, maybe the thermal paste is done.",
         "IT can reapply it; the form takes two minutes."),
        ("File the IT ticket for me with a Friday dropoff, and copy {a}.",
         "Ticket filed for Friday dropoff."),
        ("I'll use the loaner in the meantime.",
         "Noted, loaner reserved."),
    )),
    ("team lunch", (
        ("Organize the team lunch for the release next week.",
         "Ten people confirmed so far."),
        ("Pick somewhere with vegetarian options near the office.",
         "Two candidates within a five minute walk."),
        ("Book the quieter one for Wednesday at noon.",
         "Booked for Wednesday at noon."),
        ("Put it on everyone's calendar, including {a}.",
         "Invites sent."),
    )),
    ("workout plan", (
        ("I want to get back to a morning run routine; {a} may join on Saturdays.",
         "Your old plan was three runs a week."),
        ("Start with two, Tuesday and Saturday.",
         "Scheduled for Tuesday and Saturday mornings."),
        ("If it rains, swap in the rowing machine.",
         "Noted as the fallback."),
        ("Track it for a month before we change anything.",
         "Tracking enabled for four weeks."),
    )),
    ("visa paperwork", (
        ("The {b} work visa forms need my employment letter.",
         "HR issued one last year; it may be reusable."),
        ("It expired, request a fresh one from HR.",
         "Request sent to HR."),
        ("Print two copies once it arrives.",
         "Will do."),
        ("The appointment is on the second, keep that morning clear.",
         "Morning of the second blocked."),
    )),
    ("garden project", (
        ("The balcony planters need repotting before spring.",
         "You'll need soil and two larger pots."),
        ("Order those for weekend delivery to the {b} address.",
         "Ordered for Saturday delivery."),
        ("Remind me to water twice a week after repotting.",
         "Recurring reminder set."),
        ("The rosemary goes in the sunniest corner.",
         "Noted."),
    )),
    ("phone upgrade", (
        ("My phone battery barely lasts to lunch now, {a} noticed it on our call.",
         "It's twenty months old; a battery swap is an option."),
        ("Compare a swap against trading in for the newer model.",
         "Swap is cheaper; trade-in adds a better camera."),
        ("I'll do the battery swap and decide again next year.",
         "Booked at the repair shop for Saturday."),
        ("Back everything up the night before.",
         "Backup reminder set for Friday night."),
    )),
)


def distractor_sessions(count: int) -> list[EvalSession]:
    """`count` fixed-text distractor sessions, deterministic in `count`.

    Templates rotate with the fill lists so repeated topics still differ in
    their concrete details. Every session is 8-12 messages of ordinary
    personal-agent traffic.
    """
    sessions = []
    for i in range(count):
        topic, exchanges = _DISTRACTOR_TEMPLATES[i % len(_DISTRACTOR_TEMPLATES)]
        a = _FILL_A[i % len(_FILL_A)]
        b = _FILL_B[i % len(_FILL_B)]
        messages: list[dict] = []
        for user, assistant in exchanges:
            messages.append({"role": "user", "content": user.format(a=a, b=b)})
            messages.append({"role": "assistant", "content": assistant.format(a=a, b=b)})
        sessions.append(
            EvalSession(
                label=f"distractor {i + 1}: {topic}",
                fixed_messages=tuple(messages),
            )
        )
    return sessions


@dataclass(frozen=True)
class PlacedSession:
    """One session in a run's chronological order."""

    session: EvalSession
    #: Index into scenario.sessions for fact-bearing sessions (what
    #: `probe_from_session` and `planted_goals` refer to); None for distractors.
    fact_index: int | None


def interleave(scenario: EvalScenario, history_size: int) -> list[PlacedSession]:
    """Order all sessions for one run: `history_size` total, facts buried early.

    Fact-bearing sessions keep their relative order and sit spread across the
    earliest third of the run, so the probed facts are genuinely buried under
    later distractor history. The one exception: a fact session that a probe
    continues (`probe_from_session`) is chronologically "today" and is placed
    last instead.
    """
    n_fact = len(scenario.sessions)
    if history_size < max(n_fact, 1):
        raise ValueError(f"history_size {history_size} < {n_fact} fact sessions")

    continued = {p.probe_from_session for p in scenario.probes} - {None}
    early = [i for i in range(n_fact) if i not in continued]
    distractors = distractor_sessions(history_size - n_fact)

    n_early = len(early)
    early_third = max(n_early, (history_size + 2) // 3)
    # Even spread across the earliest third; increments >= 1 because
    # early_third >= n_early, so positions never collide.
    positions = {
        round(j * (early_third - 1) / max(n_early - 1, 1)): early[j]
        for j in range(n_early)
    }
    ordered: list[PlacedSession] = []
    d = 0
    for pos in range(history_size - len(continued)):
        if pos in positions:
            i = positions[pos]
            ordered.append(PlacedSession(session=scenario.sessions[i], fact_index=i))
        else:
            ordered.append(PlacedSession(session=distractors[d], fact_index=None))
            d += 1
    for i in sorted(continued):
        ordered.append(PlacedSession(session=scenario.sessions[i], fact_index=i))
    return ordered
