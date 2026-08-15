"""Seed building and context assembly for proactive retrieval. Pure — hand-built
neighborhoods and fake stores; no Neo4j, no Postgres, no model."""

from __future__ import annotations

import pytest

from app.services.activation import Edge, Neighborhood
from app.services.proactive import (
    ActivatedNode,
    assemble_activated,
    build_seeds,
    explain_path,
)


def _nb(*edges, labels=None) -> Neighborhood:
    return Neighborhood(
        edges=[Edge(s, r, d, c) for s, r, d, c in edges],
        labels=labels or {},
    )


# ------------------------------------------------------------------ seeds


def test_live_entities_seed_at_full_activation():
    seeds = build_seeds(
        live_entities=["clickhouse", "alembic"],
        task_entities=[],
        alias_to_profile={},
    )
    assert seeds == {"Entity:clickhouse": 1.0, "Entity:alembic": 1.0}


def test_task_entities_seed_lower_than_live():
    seeds = build_seeds(
        live_entities=["clickhouse"],
        task_entities=["kafka"],
        alias_to_profile={},
        task_seed=0.6,
    )
    assert seeds["Entity:clickhouse"] == 1.0
    assert seeds["Entity:kafka"] == pytest.approx(0.6)


def test_live_evidence_beats_inherited_on_the_same_entity():
    """An entity both live and task-touched keeps the live 1.0, never the 0.6."""
    seeds = build_seeds(
        live_entities=["clickhouse"],
        task_entities=["clickhouse"],
        alias_to_profile={},
        task_seed=0.6,
    )
    assert seeds["Entity:clickhouse"] == 1.0


def test_aliases_collapse_to_the_profile():
    """'dana' and 'dana whitfield' seed ONE UserProfile node, not two entities."""
    seeds = build_seeds(
        live_entities=["dana", "dana whitfield", "clickhouse"],
        task_entities=[],
        alias_to_profile={"dana": "u-1", "dana whitfield": "u-1"},
    )
    assert seeds == {"UserProfile:u-1": 1.0, "Entity:clickhouse": 1.0}


def test_empty_inputs_yield_no_seeds():
    assert build_seeds(live_entities=[], task_entities=[], alias_to_profile={}) == {}


# --------------------------------------------------------------- assembly


def test_assembly_ranks_by_activation_and_types_by_label():
    nb = _nb(
        ("Entity:clickhouse", "MENTIONS", "Episode:41", 3),
        ("Episode:41", "INVOKED", "ToolCall:9", 1),
        ("Entity:clickhouse", "RELATED_TO", "Entity:alembic", 7),
        labels={
            "Entity:clickhouse": "Entity",
            "Episode:41": "Episode",
            "ToolCall:9": "ToolCall",
            "Entity:alembic": "Entity",
        },
    )
    activated = {"Entity:clickhouse": 1.0, "Episode:41": 0.5, "Entity:alembic": 0.3, "ToolCall:9": 0.2}
    nodes = assemble_activated(nb, activated, seeds={"Entity:clickhouse": 1.0})

    assert [n.node_id for n in nodes] == [
        "Entity:clickhouse", "Episode:41", "Entity:alembic", "ToolCall:9"
    ]
    kinds = {n.node_id: n.label for n in nodes}
    assert kinds["Episode:41"] == "Episode"
    assert kinds["ToolCall:9"] == "ToolCall"
    # Every node knows its key so callers can hydrate from L2/L4 by id.
    assert next(n for n in nodes if n.node_id == "Episode:41").key == "41"


def test_seeds_are_marked_and_kept():
    nb = _nb(("Entity:a", "MENTIONS", "Episode:1", 1),
             labels={"Entity:a": "Entity", "Episode:1": "Episode"})
    nodes = assemble_activated(nb, {"Entity:a": 1.0, "Episode:1": 0.4}, seeds={"Entity:a": 1.0})
    by_id = {n.node_id: n for n in nodes}
    assert by_id["Entity:a"].is_seed is True
    assert by_id["Episode:1"].is_seed is False


def test_explain_path_traces_the_strongest_chain_back_to_a_seed():
    """Provenance: the edge chain that lit a node, most-activated route."""
    nb = _nb(
        ("Entity:clickhouse", "RELATED_TO", "Entity:alembic", 7),
        ("Entity:alembic", "MENTIONS", "Episode:41", 2),
        ("Episode:41", "INVOKED", "ToolCall:9", 1),
    )
    activated = {
        "Entity:clickhouse": 1.0,
        "Entity:alembic": 0.35,
        "Episode:41": 0.2,
        "ToolCall:9": 0.12,
    }
    path = explain_path(nb, activated, target="ToolCall:9", seeds={"Entity:clickhouse": 1.0})
    assert path == [
        ("Entity:clickhouse", "RELATED_TO", 7, "Entity:alembic"),
        ("Entity:alembic", "MENTIONS", 2, "Episode:41"),
        ("Episode:41", "INVOKED", 1, "ToolCall:9"),
    ]


def test_explain_path_of_a_seed_is_empty():
    nb = _nb(("Entity:a", "MENTIONS", "Episode:1", 1))
    assert explain_path(nb, {"Entity:a": 1.0}, target="Entity:a", seeds={"Entity:a": 1.0}) == []


def test_activated_node_renders_a_stable_key_and_label():
    n = ActivatedNode(node_id="Task:abc-123", label="Task", key="abc-123", activation=0.7, is_seed=False)
    assert (n.label, n.key) == ("Task", "abc-123")
