"""Proactive context assembly: seeds → activation → ranked, explainable nodes.

The pure half of proactive retrieval. Given what the conversation *surfaced*
(entities in recent messages and the query, plus the active task's entities),
build seeds, spread activation over a pulled neighborhood, and return the
activated nodes ranked and typed so `retrieval.py` can hydrate each one from
the tier that owns its payload (L2 for episodes, L4 for tool calls, the graph
for the rest).

Nothing here touches a database or a model. Node ids are the `Label:key`
strings produced by `Neo4jStore.fetch_neighborhood`.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.activation import Neighborhood

#: Activation for entities the user's active Task already touches. Lower than
#: live evidence (1.0): something said in *this* conversation outranks something
#: inherited from the goal it is pursuing.
DEFAULT_TASK_SEED = 0.6


@dataclass(frozen=True)
class ActivatedNode:
    """One node that activation reached, ready to be hydrated and rendered."""

    node_id: str
    label: str
    key: str
    activation: float
    is_seed: bool


def build_seeds(
    *,
    live_entities: list[str],
    task_entities: list[str],
    alias_to_profile: dict[str, str],
    task_seed: float = DEFAULT_TASK_SEED,
) -> dict[str, float]:
    """Turn surfaced entity names into `{node_id: activation}` seeds.

    - Live entities (recent messages + query) seed at 1.0.
    - Task-touched entities seed at `task_seed`; a name that is both keeps 1.0.
    - Names that are aliases of a profile seed the `UserProfile` node instead
      of the fragment `Entity` — that is the alias work paying off: "dana" and
      "dana whitfield" light one identity, not two.

    Names are expected already normalized (see `normalize_entity_name`).
    """
    seeds: dict[str, float] = {}

    def _node_for(name: str) -> str:
        profile = alias_to_profile.get(name)
        return f"UserProfile:{profile}" if profile else f"Entity:{name}"

    for name in task_entities:
        if name:
            nid = _node_for(name)
            seeds[nid] = max(seeds.get(nid, 0.0), task_seed)
    for name in live_entities:
        if name:
            nid = _node_for(name)
            seeds[nid] = 1.0
    return seeds


def _split(node_id: str) -> tuple[str, str]:
    label, _, key = node_id.partition(":")
    return label, key


def assemble_activated(
    neighborhood: Neighborhood,
    activated: dict[str, float],
    *,
    seeds: dict[str, float],
) -> list[ActivatedNode]:
    """Rank activated nodes (desc) and attach label/key for hydration.

    Labels come from the neighborhood when it saw the node; a seed the
    neighborhood never returned (isolated entity) still resolves from its id.
    """
    out: list[ActivatedNode] = []
    for node_id, act in sorted(activated.items(), key=lambda kv: kv[1], reverse=True):
        label_from_id, key = _split(node_id)
        label = neighborhood.labels.get(node_id, label_from_id)
        out.append(
            ActivatedNode(
                node_id=node_id,
                label=label,
                key=key,
                activation=act,
                is_seed=node_id in seeds,
            )
        )
    return out


def explain_path(
    parents: dict[str, tuple[str, str, int | None]],
    *,
    target: str,
    seeds: dict[str, float],
) -> list[tuple[str, str, int | None, str]]:
    """The edge chain that actually lit `target`, from `ActivationResult.parents`.

    Walks the recorded parent of each node back to a seed. Returned forward
    (seed -> target) as `(src, rel, count, dst)` so a source line reads:
    `clickhouse -RELATED_TO(7)-> alembic -MENTIONS(2)-> episode 41`.

    This used to search backwards for the "strongest-looking" neighbour, which
    is not equivalent: where two routes tie, or where a route was superseded,
    the search reported chains that never happened (174 divergences across 4000
    random graphs; the minimal case is a two-hop route tying exactly with the
    one-hop route that actually won). Reading the recorded parent cannot lie.

    A chain that does not reach a seed — possible only if `parents` is partial —
    stops where the record ends and returns what it has.
    """
    if target in seeds or target not in parents:
        return []
    chain: list[tuple[str, str, int | None, str]] = []
    current = target
    visited = {target}
    while current not in seeds:
        record = parents.get(current)
        if record is None:
            break
        parent, rel, count = record
        chain.append((parent, rel, count, current))
        if parent in visited:
            break  # defensive: a cycle in the record cannot be walked
        visited.add(parent)
        current = parent
    chain.reverse()
    return chain
