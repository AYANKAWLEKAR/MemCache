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
        ("I need to start planning the {b} trip for the team offsite in March, and I want the big pieces settled before the end of this week so it stops hanging over me.",
         "Understood. The big pieces are flights, the venue, and the group dinner. Do you want to start with flights or the venue?"),
        ("Venue first. It should be walkable from the main station, have a room that fits fourteen people, and decent coffee nearby, because last year's spot failed on all three counts.",
         "I will shortlist three venues near the station with a fourteen-seat room and note the coffee situation for each."),
        ("Also remind me to renew my passport before we book anything international, since it expires in June and the renewal office has been slow lately.",
         "Reminder set for the passport renewal, flagged as blocking the flight booking."),
        ("The budget cap is whatever we agreed with finance last quarter, and I would rather come in under it than have that conversation again.",
         "Noted. I will keep the running total visible on the shortlist so nothing sneaks past the cap."),
        ("Once the venue is picked, send {a} the details so they can plan the workshop sessions around the room layout.",
         "Will do. The venue summary goes to {a} the same day it is booked."),
        ("One more thing: check whether the March dates collide with the school holidays, because half the team has kids and last year we learned that the hard way.",
         "Checked. The proposed week avoids the holidays in every region the team lives in, so the dates are safe to book."),
    )),
    ("expense report", (
        ("I still owe the expense report from the {b} conference and the deadline email this morning had a distinctly final tone, so let's get it done today.",
         "You have twelve receipts uploaded so far: the hotel, two flights, the registration fee, and eight meals."),
        ("The taxi receipts are the missing piece. I have four paper ones in my jacket pocket and I will photograph them tonight after dinner.",
         "I will hold the draft open and slot the four taxi receipts in as soon as the photos arrive."),
        ("Make sure the whole thing is filed under the travel category and not team events, because the {b} trip got miscategorized last time and took three emails to fix.",
         "Recategorized to travel. I also checked the hotel entry, which was sitting in the wrong bucket again."),
        ("The registration fee should be split against the education budget, the way we did it for the spring workshop.",
         "Split applied: registration to education, everything else to travel."),
        ("Submit it the moment the taxi receipts are in, and forward me the confirmation so I can stop thinking about it.",
         "It goes out the moment they land, with the confirmation forwarded to you."),
        ("While you are in there, check whether the per-diem rate changed this year, because the policy page I found looked suspiciously stale.",
         "The rate did change in January, upward by a small amount. I have applied the current figure to the eight meal entries."),
    )),
    ("meeting prep", (
        ("Help me prepare for the Thursday sync with {a}, because last week we ran out of time and two items rolled over yet again.",
         "Last week's notes list two open items: the vendor contract question and the intern onboarding plan."),
        ("Add the hiring update as a third item, and put it first, since {a} specifically asked for progress on the two open roles.",
         "Added and moved to the top. The agenda is hiring update, vendor contract, then intern onboarding."),
        ("Drop anything about the venue discussion, that got settled in the hallway on Monday and does not need airtime.",
         "Dropped. The agenda stays at three items, which fits the thirty minutes."),
        ("Attach the one-page summary I wrote about the vendor options so nobody reads it for the first time during the meeting.",
         "Attached to the invite, with a note asking everyone to read it beforehand."),
        ("Send the final agenda out an hour before the meeting, and remind me fifteen minutes ahead so I can reread my own summary.",
         "Scheduled: agenda one hour prior, your reminder fifteen minutes before the start."),
        ("Last thing, book the small conference room instead of the glass one, because the glass one echoes and the recording was unusable last time.",
         "The small room is booked for Thursday. I also moved the recording device request over to the new room."),
    )),
    ("gift ideas", (
        ("I need a proper gift idea for {a}'s farewell next week, something better than the usual mug and card routine we always fall back on.",
         "They have mentioned enjoying trail running and being particular about coffee, which gives two good directions."),
        ("Coffee then. Find something from a local roaster, a subscription if that exists, because a one-off bag feels thin for someone leaving after four years.",
         "There are two well-reviewed roasters nearby and one runs a three-month subscription that can be gifted."),
        ("The three-month subscription is right. Also get a card, but circulate it early so the whole team actually signs it this time.",
         "Subscription noted and the card will circulate starting tomorrow morning, giving everyone three days."),
        ("Collect contributions the usual way, and cover whatever gap is left from me quietly.",
         "Contribution link sent. I will tell you the gap privately before the purchase."),
        ("Have everything wrapped and ready by Thursday evening, because Friday morning will be chaos and I will forget.",
         "Wrapping reminder set for Thursday evening, with the card and subscription voucher together."),
        ("Also find out {a}'s last day precisely, because if it is Thursday rather than Friday the whole schedule moves up a day.",
         "Their last day is Friday, confirmed against the team calendar, so the Thursday evening wrap deadline stands."),
    )),
    ("apartment errands", (
        ("The landlord finally replied about the radiator repair after three weeks of silence, and the proposed slot is Tuesday morning between eight and eleven.",
         "Good news at last. Do you want me to confirm the Tuesday slot as proposed?"),
        ("Yes, confirm it, but that means moving my standup with {a}, so shift that to the afternoon and add an apology note.",
         "Standup moved to two in the afternoon with a short note, and the repair slot is confirmed."),
        ("While we are on the apartment, add lightbulbs, water filters, and the bathroom sealant to the shopping list before I forget them all again.",
         "All three added to the list, grouped under the hardware store stop."),
        ("The building management also wants the parking form back this month, which has been sitting on my desk since it arrived.",
         "I have added a task to scan and return the parking form by the twentieth."),
        ("Remind me to be home by nine on Tuesday so I do not leave the repair person standing in the hallway.",
         "Reminder set for Tuesday at eight forty-five, before the window opens."),
        ("And check the lease renewal date while you have the folder open, because I have a vague feeling it lands sometime this spring.",
         "The lease renews at the end of May, with a sixty day notice window. I have added a reminder well ahead of it."),
    )),
    ("reading list", (
        ("Add the distributed systems paper {a} recommended to my reading list, the one about consensus tradeoffs they would not stop talking about at lunch.",
         "Added. Your queue now has six items, and this one is the only paper among the books."),
        ("Move it to the top, because the reading group meets in two weeks and I refuse to be the person who skimmed the abstract at breakfast.",
         "Moved to the top, with the reading group date attached as the deadline."),
        ("Archive the two novels I finished last month so the list stops lying about my backlog.",
         "Archived both. Four items remain, with the paper first."),
        ("Estimate how long the paper actually takes, because if it is a forty-page monster I need to split it across evenings.",
         "It runs thirty-two pages. Two evenings at your usual pace, three with note-taking."),
        ("Then block two Sunday afternoons for it and keep those blocks safe from meeting invites.",
         "Two Sunday blocks created and marked as busy."),
        ("If the reading group picks a second paper, decline on my behalf, one paper in two weeks is my honest capacity.",
         "Understood. If a second paper appears, I will reply that you are covering the consensus paper only."),
    )),
    ("laptop trouble", (
        ("My laptop fan has started roaring during every video call again, and {a} noticed it on our one-on-one, which is mildly embarrassing.",
         "Last time this happened, closing the browser profile with forty tabs helped for about a month."),
        ("It is worse now, the whole chassis gets hot within ten minutes of a call, so I suspect the thermal paste has finally given up after three years.",
         "That fits the symptoms. IT can reapply thermal paste, and their form takes about two minutes to fill."),
        ("File the IT ticket for me, request a Friday dropoff, and mention the fan noise plus the heat so they do not just blow the dust out and call it fixed.",
         "Ticket filed for Friday dropoff with both symptoms described and the age of the machine noted."),
        ("Reserve one of the loaner machines for Friday and Monday, because I have back-to-back calls both days.",
         "Loaner reserved for both days and the confirmation is in your inbox."),
        ("Back up the working directory tonight, just in case they decide to reimage the thing without asking.",
         "Backup scheduled for tonight, covering the working directory and your local notes."),
        ("If IT says the repair needs more than a week, escalate to a replacement request instead, this machine is past its refresh date anyway.",
         "Noted as the escalation path. The refresh eligibility date passed two months ago, which strengthens the case."),
    )),
    ("team lunch", (
        ("Organize the team lunch for the release next week, because the release itself is the easy part and the lunch is somehow always the thing that slips.",
         "Ten people have confirmed so far, with two waiting on their calendars."),
        ("Pick somewhere with real vegetarian options near the office, not the place where the vegetarian option is a side salad and an apology.",
         "Two candidates within a five minute walk fit: one Lebanese, one Italian, both with proper vegetarian mains."),
        ("The Lebanese place, and book the quieter back room for Wednesday at noon if they still have it.",
         "Back room booked for Wednesday at twelve for up to fourteen people."),
        ("Ask {a} to collect dietary notes from the two remote folks who are visiting, since they always get forgotten.",
         "Request sent to {a}, and I will attach the notes to the booking."),
        ("Put it on everyone's calendar with the address and a walking map, so nobody arrives at twelve twenty pretending they knew the way.",
         "Invites sent with the address, map link, and the back room mentioned."),
        ("If the release slips past Wednesday, keep the booking anyway, the team earned the lunch either way.",
         "Agreed. The booking stands regardless of the release date, and I will only move it if fewer than six can attend."),
    )),
    ("workout plan", (
        ("I want to get back to a morning run routine before the {b} half marathon registration opens, and this time make it survive longer than two weeks.",
         "Your old plan was three runs a week, which historically collapsed at week three."),
        ("Start with two then, Tuesday and Saturday, and keep them short enough that skipping feels sillier than going.",
         "Scheduled: thirty minutes Tuesday and forty-five Saturday, both in the morning."),
        ("If it rains, swap the run for the rowing machine instead of cancelling, because cancelling is how the whole thing died last time.",
         "Noted as the standing fallback, and I will suggest it automatically on rainy mornings."),
        ("{a} mentioned wanting to join the Saturday runs, so send them the plan and the meeting point.",
         "Plan and meeting point sent to {a}, who confirmed for this Saturday."),
        ("Track everything for a month before we change any variable, distance, pace, or schedule.",
         "Tracking enabled for four weeks, with a review scheduled at the end."),
        ("And if a work trip lands on a run day, move the run rather than skip it, even a short one keeps the streak honest.",
         "Standing rule recorded: travel moves a run to the nearest free morning instead of cancelling it."),
    )),
    ("visa paperwork", (
        ("The {b} work visa forms came back and they need my employment letter, plus two other documents I have never heard of in fifteen years of paperwork.",
         "HR issued an employment letter last year. It may be reusable if the date requirements are loose."),
        ("It expired in January, so request a fresh one from HR and ask them to include the salary line this time, since the consulate bounced the last one for omitting it.",
         "Request sent to HR with the salary line requirement spelled out explicitly."),
        ("The other two documents are a bank statement and proof of address, both of which I can download tonight.",
         "Noted. I will add both to the document checklist so nothing is missing at the appointment."),
        ("Print two copies of everything once the letter arrives, because the consulate keeps one set and I have learned not to hand over my only copy.",
         "Two full sets will be printed and clipped separately."),
        ("The appointment is on the second at nine in the morning, so keep that whole morning clear and remind me the evening before.",
         "Morning of the second blocked, with a checklist reminder the evening before."),
        ("If the consulate asks for anything beyond the checklist, forward it to me immediately rather than batching it with the weekly summary.",
         "Understood. Anything unexpected from the consulate reaches you the moment it arrives."),
    )),
    ("garden project", (
        ("The balcony planters need repotting before spring actually arrives, and the rosemary is looking at me reproachfully every morning.",
         "You will need two larger pots, a bag of soil, and drainage gravel for the repotting."),
        ("Order all of that for Saturday delivery to the {b} address, and add a small trowel because mine vanished sometime last autumn.",
         "Ordered for Saturday delivery: two pots, soil, gravel, and a hand trowel."),
        ("The rosemary goes in the sunniest corner by the railing, and the mint goes wherever the rosemary is not, since it takes over everything.",
         "Noted: rosemary in the sun corner, mint isolated in its own pot on the far side."),
        ("Set a recurring reminder to water twice a week after the repotting, because my watering history speaks for itself.",
         "Recurring reminder set for Wednesdays and Sundays."),
        ("If the weather turns cold again next week, remind me to pull the pots against the wall overnight.",
         "I will watch the forecast and remind you on any night below five degrees."),
        ("If the pots arrive damaged like last time, photograph them before unboxing further and start the return straight away.",
         "Will do. Photos first, then the return claim, and I will reorder the same day so the schedule holds."),
    )),
    ("phone upgrade", (
        ("My phone battery barely lasts to lunch now, and {a} noticed it die mid-sentence on our call, which settles the question of whether it is getting worse.",
         "The phone is twenty months old. A battery replacement is the cheap option, a trade-in the expensive one."),
        ("Compare the battery swap against trading in for the newer model, including what my current one is actually worth in trade.",
         "The swap costs about a tenth of the upgrade. The trade-in value drops sharply next quarter, but the only real gain is the camera."),
        ("Then I will do the battery swap now and revisit the upgrade question next year when the trade-in math changes.",
         "Sensible. Booked at the repair shop for Saturday at ten, with a ninety minute estimate."),
        ("Back everything up the night before, photos included, because the last repair wiped a colleague's phone without warning.",
         "Full backup scheduled for Friday night, photos and messages included."),
        ("And put the repair receipt in the warranty folder afterwards, since the swap extends the battery warranty by a year.",
         "I will file the receipt in the warranty folder as soon as it arrives."),
        ("If the shop finds anything else wrong during the swap, tell them to call me before doing extra work, not after.",
         "Noted on the booking: no additional work without a phone call to you first."),
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
