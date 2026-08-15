"""Edge weights and activation spreading over an in-memory neighborhood.

Pure functions. No database, no model, no settings lookups — the neighborhood
is pulled once by the caller (see `neo4j_store.fetch_neighborhood`) and handed
in as plain data, so everything here is testable on hand-built graphs.

Why not Cypher: activation is iterative multiplication with a floor, not a path
pattern. Without GDS (this deployment has none) Cypher expresses it badly and
needs a round-trip per hop. Keeping it here also makes the algorithm swappable —
personalized PageRank would take the same `Neighborhood` and return the same
`dict[node, activation]`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

#: Static evidence-quality prior per edge type. Observation *count* scales
#: within the prior; the prior itself says how much one observation is worth.
#: A task-confirmed ADVANCES edge (LLM adjudicated) is trusted more than a
#: co-occurrence RELATED_TO edge (two names within ten tokens).
EDGE_PRIORS: dict[str, float] = {
    "ADVANCES": 1.0,
    "MENTIONS": 0.9,
    "HAS_ALIAS": 0.9,
    "PURSUES": 0.9,
    "INVOKED": 0.9,
    "HAS_EPISODE": 0.9,
    "PARTICIPATED_IN": 0.9,
    "DECIDED": 0.8,
    "PREFERS": 0.8,
    "RELATED_TO": 0.5,
}

#: Anything unlisted spreads conservatively; a new edge type must earn its prior.
_UNKNOWN_PRIOR = 0.3

#: Edges whose count is *evidence*: they can be observed repeatedly and each
#: observation strengthens the link. Everything else is structural — a fact that
#: either holds or does not (a tool call has exactly one INVOKED edge, forever)
#: — and carries its full prior regardless of count. Log-scaling a count that is
#: always 1 would strangle every structural hop to ~23% of its prior; measured
#: on the live graph, a ToolCall two hops from a seed died at 0.027 under a
#: 0.05 floor for exactly that reason.
COUNTED_EDGES: frozenset[str] = frozenset({"RELATED_TO", "MENTIONS"})

#: Count at which an edge reaches its full prior. Logarithmic below it so the
#: fiftieth observation adds little; clamped above it so one saturating session
#: cannot dominate a link.
DEFAULT_COUNT_CAP = 20


def edge_weight(rel_type: str, count: int | None, *, cap: int = DEFAULT_COUNT_CAP) -> float:
    """Effective weight in [0, prior].

    Counted edges (`COUNTED_EDGES`): `prior * log(1+count) / log(1+cap)`.
    Structural edges: the prior itself — repetition is not evidence for them.

    A missing or non-positive count reads as 1 — legacy edges written before
    counting existed represent one observation, not zero.
    """
    prior = EDGE_PRIORS.get(rel_type, _UNKNOWN_PRIOR)
    if rel_type not in COUNTED_EDGES:
        return prior
    n = count if (count is not None and count > 0) else 1
    n = min(n, cap)
    return prior * (math.log1p(n) / math.log1p(cap))


@dataclass(frozen=True)
class Edge:
    """One relationship pulled from the graph. Node ids are opaque strings."""

    src: str
    rel: str
    dst: str
    count: int | None = 1


@dataclass
class Neighborhood:
    """The subgraph activation may spread over, plus optional node labels."""

    edges: list[Edge] = field(default_factory=list)
    #: node id -> label ("Entity", "Episode", ...). Not needed for spreading;
    #: assembly uses it to decide how to render each activated node.
    labels: dict[str, str] = field(default_factory=dict)

    def adjacency(self) -> dict[str, list[tuple[str, str, int | None]]]:
        """Undirected adjacency: node -> [(neighbor, rel, count)].

        Relations are evidence in both directions for spreading — an episode
        that MENTIONS an entity lights up when the entity does, and vice versa.
        """
        adj: dict[str, list[tuple[str, str, int | None]]] = defaultdict(list)
        for e in self.edges:
            adj[e.src].append((e.dst, e.rel, e.count))
            adj[e.dst].append((e.src, e.rel, e.count))
        return adj


def spread_activation(
    neighborhood: Neighborhood,
    *,
    seeds: dict[str, float],
    floor: float,
    decay: float,
    cap: int = DEFAULT_COUNT_CAP,
) -> dict[str, float]:
    """Spread activation from `seeds` across the neighborhood.

        activation[child] = max(activation[child],
                                activation[parent] * edge_weight * decay)

    Breadth-first while any frontier node still exceeds `floor`. There is no
    hop limit: depth is a consequence of weight. Termination is guaranteed
    because `decay < 1` and every weight is <= 1, so activation strictly
    decreases along any path, and a node re-enters the frontier only when its
    activation *increases* — which can happen finitely many times before it
    falls under the floor.

    Returns nodes at or above `floor`, sorted by activation descending.
    """
    if not seeds:
        return {}
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1), got {decay}")

    adj = neighborhood.adjacency()
    activation: dict[str, float] = {}
    frontier: list[str] = []
    for node, a in seeds.items():
        if a >= floor:
            activation[node] = max(activation.get(node, 0.0), a)
            frontier.append(node)

    while frontier:
        next_frontier: list[str] = []
        for parent in frontier:
            parent_act = activation[parent]
            for child, rel, count in adj.get(parent, ()):
                proposed = parent_act * edge_weight(rel, count, cap=cap) * decay
                if proposed < floor:
                    continue
                if proposed > activation.get(child, 0.0):
                    activation[child] = proposed
                    next_frontier.append(child)
        frontier = next_frontier

    return dict(sorted(activation.items(), key=lambda kv: kv[1], reverse=True))
