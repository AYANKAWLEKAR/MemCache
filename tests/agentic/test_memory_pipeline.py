"""End-to-end memory pipeline tests driven by an Ollama agent.

Flow under test, with nothing mocked:

    agent turns -> POST /memory/ingest -> Redis (L1)
                                       -> process_conversation
                                          -> Ollama summarize
                                          -> MiniLM embed -> Postgres (L2)
                                          -> spaCy NER -> Neo4j (L3)
                -> POST /memory/retrieve -> hybrid context

Assertions run against ground truth declared in `scenarios.py` and are verified
with independent Cypher/SQL probes, never through the write-side API.
"""

from __future__ import annotations

import pytest

from tests.agentic import graph_probe as probe
from tests.agentic.ollama_agent import Turn, summarize_anchor_recall
from tests.agentic.scenarios import ALL_SCENARIOS, Scenario

pytestmark = [pytest.mark.agentic, pytest.mark.integration]


def _run_scenario(scenario: Scenario, agent, ingest, sid: str) -> float:
    """Play the scenario into MemCache; return anchor-recall for diagnostics."""
    agent.transcript.clear()
    for turn in scenario.turns:
        messages = agent.exchange(turn)
        ingest(sid, messages, {"scenario": scenario.name})
    return summarize_anchor_recall(scenario.turns, agent.transcript)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_scenario_builds_expected_memory(
    scenario, agent, ingest, retrieve, session_id, neo4j_driver, pg_engine
):
    """The whole pipeline learns what the scenario says it should learn."""
    anchor_recall = _run_scenario(scenario, agent, ingest, session_id)

    # L2: one durable, embedded episode per ingested exchange.
    rows = probe.episode_rows(pg_engine, session_id)
    assert rows, "no episode reached L2 — summarization or insert failed"
    assert len(rows) == len(scenario.turns), (
        f"expected {len(scenario.turns)} episodes, got {len(rows)}"
    )
    assert all(r["has_embedding"] for r in rows), "episode stored without embedding"
    assert all(r["summary"].strip() for r in rows), "episode stored with empty summary"

    # L3: every episode is reachable from its session.
    assert probe.episode_ids(neo4j_driver, session_id) == {r["id"] for r in rows}, (
        "L2 episode ids and L3 Episode nodes disagree"
    )

    # L3: the entity-episode relation graph — the core structure.
    found = probe.mentioned_entities(neo4j_driver, session_id)
    missing = [e for e in scenario.expected_entities if e not in found]
    assert not missing, (
        f"entities missing from Session->Episode->MENTIONS->Entity: {missing}\n"
        f"found: {sorted(found)}\nanchor_recall={anchor_recall:.2f}"
    )

    # Extraction-quality metric for names spaCy handles unreliably. Reported so
    # the weakness stays visible; never asserted, so it cannot flake the gate.
    if scenario.optional_entities:
        detected = [e for e in scenario.optional_entities if e in found]
        print(
            f"\n[metric] {scenario.name}: optional entity extraction "
            f"{len(detected)}/{len(scenario.optional_entities)} "
            f"(missed: {sorted(set(scenario.optional_entities) - set(detected))})"
        )

    related = probe.related_pairs(neo4j_driver, session_id)
    for a, b in scenario.expected_related:
        # Only assert the edge when both endpoints were actually extracted —
        # otherwise this re-tests NER reliability rather than edge construction.
        if a not in found or b not in found:
            print(f"[metric] {scenario.name}: RELATED_TO {a!r}-{b!r} unverified (endpoint missing)")
            continue
        assert frozenset((a, b)) in related, (
            f"both entities extracted but no RELATED_TO between {a!r} and {b!r}"
        )

    decisions = " | ".join(probe.decision_texts(neo4j_driver, session_id)).lower()
    for fragment in scenario.expected_decision_fragments:
        assert fragment.lower() in decisions, (
            f"decision fragment {fragment!r} not recorded; got {decisions!r}"
        )

    preferences = " | ".join(probe.preference_texts(neo4j_driver, session_id)).lower()
    for fragment in scenario.expected_preference_fragments:
        assert fragment.lower() in preferences, (
            f"preference fragment {fragment!r} not recorded; got {preferences!r}"
        )

    # Retrieval must surface the facts again.
    for question, must_contain in scenario.recall_probes:
        result = retrieve(session_id, question, 1500)
        assert must_contain.lower() in result["context"].lower(), (
            f"retrieval for {question!r} lost {must_contain!r}\n"
            f"context: {result['context'][:400]}"
        )
        assert result["sources"], "retrieval returned context with no provenance"


def test_retrieval_provenance_spans_all_tiers(
    agent, ingest, retrieve, session_id, neo4j_driver
):
    """A retrieval after a full ingest cites L1, L2 and L3."""
    from tests.agentic.scenarios import ONBOARDING

    _run_scenario(ONBOARDING, agent, ingest, session_id)
    result = retrieve(session_id, "What do you know about Dana and the backend?", 2000)

    tiers = {s["tier"] for s in result["sources"]}
    assert {"L1", "L2", "L3"} <= tiers, f"missing tiers, got {sorted(tiers)}"
    assert result["status"] in {"ok", "degraded"}
    for source in result["sources"]:
        assert source["type"], "source missing type"


# --------------------------------------------------------------------------
# Deterministic regression tests. No Ollama in the ingest path for these — the
# text is fixed so a failure means MemCache regressed, not that the model drifted.
# --------------------------------------------------------------------------


def test_repeated_entity_with_trailing_punctuation_is_one_node(
    ingest, session_id, neo4j_driver, monkeypatch
):
    """Regression: 'Vertex Labs.' and 'Vertex Labs' must not become two nodes."""
    monkeypatch.setattr(
        "app.workers.tasks.summarize_conversation_ollama",
        lambda messages, settings=None: "Dana joined Vertex Labs and uses Go.",
    )
    ingest(
        session_id,
        [
            {"role": "user", "content": "I just joined Vertex Labs."},
            {"role": "assistant", "content": "Everyone at Vertex Labs uses Go."},
        ],
    )

    variants = probe.entity_nodes_like(neo4j_driver, "vertex lab")
    assert variants == {"vertex labs"}, (
        f"entity fragmented into {sorted(variants)} — normalization regressed"
    )


def test_decision_does_not_swallow_following_clause():
    """Regression: clause boundary ends a captured decision span."""
    from app.services.graph_extraction import extract_decisions_preferences_regex

    decisions, preferences = extract_decisions_preferences_regex(
        "We decided to use Rust for the backend and I prefer async standups."
    )
    assert decisions == ["use Rust for the backend"], decisions
    assert preferences == ["async standups"], preferences


def test_dates_and_numbers_are_not_stored_as_entities():
    """Regression: temporal/numeric spans must not become Entity nodes."""
    import spacy

    from app.config import settings
    from app.services.graph_extraction import entity_cooccurrence_pairs, ner_entity_texts

    nlp = spacy.load(settings.spacy_model)
    doc = nlp("Priya Raman from Lumenwave will catch up later today about the 3 reports.")

    entities = {e.lower() for e in ner_entity_texts(doc)}
    assert "lumenwave" in entities, f"real entity lost: {entities}"
    for noise in ("today", "later today", "3"):
        assert noise not in entities, f"{noise!r} stored as an entity: {entities}"

    paired = {t.lower() for pair in entity_cooccurrence_pairs(doc) for t in pair}
    assert not (paired & {"today", "later today", "3"}), (
        f"co-occurrence edge built to a date/number: {paired}"
    )


def test_episode_id_collision_is_rejected(neo4j_driver, session_id):
    """Regression: an Episode owned by another session is never silently re-linked."""
    from app.services.neo4j_store import EpisodeCollisionError, Neo4jStore

    store = Neo4jStore(neo4j_driver)
    other = f"{session_id}-other"
    episode_id = -424242  # negative: cannot collide with a real Postgres id

    store.upsert_episode(session_id, episode_id, "Owned by the first session.")
    try:
        with pytest.raises(EpisodeCollisionError):
            store.upsert_episode(other, episode_id, "Hijack attempt.")
    finally:
        with neo4j_driver.session() as s:
            s.run("MATCH (e:Episode {id: $eid}) DETACH DELETE e", eid=episode_id)
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=other)


def test_ingest_rejects_missing_api_key(api):
    """Auth is enforced on the memory endpoints."""
    response = api.post(
        "/memory/ingest",
        json={"session_id": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code in {401, 403}, response.text


def test_ingest_rejects_empty_message_list(api, auth):
    """Schema validation rejects a degenerate ingest."""
    response = api.post(
        "/memory/ingest", headers=auth, json={"session_id": "s", "messages": []}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "turn",
    [Turn(intent="say hello", anchors=["Northwind Robotics"], fallback="Hi from Northwind Robotics.")],
    ids=["anchor_enforced"],
)
def test_agent_turns_always_contain_anchors(agent, turn):
    """Harness self-check: generated traffic never loses its ground-truth anchors."""
    text = agent.user_turn(turn)
    assert turn.satisfied_by(text), f"anchor lost in generated turn: {text!r}"
