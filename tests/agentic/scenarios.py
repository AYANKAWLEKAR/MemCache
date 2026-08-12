"""Conversation scenarios with declared ground truth.

Each scenario is a plan: the turns an agent will improvise around, plus exactly
what MemCache must have learned once those turns are ingested. Tests assert
against `expected_*`, never against whatever the model happened to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.agentic.ollama_agent import Turn


@dataclass(frozen=True)
class Scenario:
    """A planned conversation and the memory state it should produce."""

    name: str
    turns: list[Turn]

    #: Normalized entity names expected in Neo4j (see `normalize_entity_name`).
    #: Hard gate — only names spaCy extracts reliably belong here.
    expected_entities: list[str] = field(default_factory=list)
    #: Names whose extraction is known to be unreliable with `en_core_web_sm`
    #: (single-token invented proper nouns detect ~75% of the time). Measured and
    #: reported, never gated, so model weakness cannot turn CI red — while the
    #: measurement keeps the weakness visible instead of hiding it.
    optional_entities: list[str] = field(default_factory=list)
    #: Entity pairs expected to be connected by RELATED_TO (normalized, unordered).
    expected_related: list[tuple[str, str]] = field(default_factory=list)
    #: Substrings expected within some Decision node's text.
    expected_decision_fragments: list[str] = field(default_factory=list)
    #: Substrings expected within some Preference node's text.
    expected_preference_fragments: list[str] = field(default_factory=list)
    #: (question, substring that must appear in retrieved context)
    recall_probes: list[tuple[str, str]] = field(default_factory=list)


ONBOARDING = Scenario(
    name="onboarding",
    turns=[
        Turn(
            intent="Introduce yourself by name and employer to your assistant.",
            anchors=["Dana Whitfield", "Northwind Robotics"],
            fallback=(
                "Hi, I'm Dana Whitfield and I work at Northwind Robotics."
            ),
        ),
        Turn(
            intent="Say which programming language your team settled on for the backend.",
            anchors=["decided to use Rust"],
            fallback="We decided to use Rust for the control backend.",
        ),
        Turn(
            intent="State a working-style preference about meetings.",
            anchors=["prefer async standups"],
            fallback="I prefer async standups over daily video calls.",
        ),
    ],
    # "Rust" is deliberately absent: en_core_web_sm does not tag programming
    # languages as entities. It is captured as a Decision instead, which is the
    # right home for it — asserted via expected_decision_fragments below.
    expected_entities=["dana whitfield", "northwind robotics"],
    expected_related=[("dana whitfield", "northwind robotics")],
    expected_decision_fragments=["rust"],
    expected_preference_fragments=["async standups"],
    recall_probes=[
        ("Where does Dana work?", "northwind robotics"),
        ("What language did the team choose?", "rust"),
    ],
)


PROJECT_CONTEXT = Scenario(
    name="project_context",
    turns=[
        Turn(
            intent="Name the project you are starting and the database you picked.",
            anchors=["Project Halyard", "chose ClickHouse"],
            fallback="On Project Halyard we chose ClickHouse for analytics.",
        ),
        Turn(
            intent=(
                "State that a colleague is consulting on the project. Write it as "
                "a statement about them in the third person, not as a greeting."
            ),
            anchors=["Priya Raman", "Lumenwave"],
            fallback="Priya Raman from Lumenwave is consulting with us.",
        ),
    ],
    expected_entities=["project halyard", "clickhouse", "priya raman"],
    # "Lumenwave" is a single invented token; en_core_web_sm tags it in only
    # ~75% of sentence frames. Measured rather than gated.
    optional_entities=["lumenwave"],
    expected_related=[("priya raman", "lumenwave")],
    expected_decision_fragments=["clickhouse"],
    expected_preference_fragments=[],
    recall_probes=[
        ("Which database was chosen?", "clickhouse"),
        ("Who is consulting on the project?", "priya raman"),
    ],
)


#: Punctuation-heavy phrasing that used to split one entity into two nodes.
PUNCTUATION_STRESS = Scenario(
    name="punctuation_stress",
    turns=[
        Turn(
            intent=(
                "Mention the same company twice, once at the end of a sentence "
                "and once mid-sentence."
            ),
            anchors=["Vertex Labs"],
            fallback=(
                "I just joined Vertex Labs. Everyone at Vertex Labs uses Go."
            ),
        ),
    ],
    expected_entities=["vertex labs"],
    expected_related=[],
    expected_decision_fragments=[],
    expected_preference_fragments=[],
    recall_probes=[],
)


ALL_SCENARIOS = [ONBOARDING, PROJECT_CONTEXT, PUNCTUATION_STRESS]
