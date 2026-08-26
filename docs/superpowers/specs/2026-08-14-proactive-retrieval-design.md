# Weighted Graph + Proactive Retrieval — Design

**Date:** 2026-08-14 · **Status:** Approved, implementing · **Branch:** `workbench`

## Problem

The graph is structural, not weighted: 13 edge types, one carries a number.
`RELATED_TO` is a bare MERGE — one co-occurrence and fifty are identical. Graph
retrieval is session-locked (`query_session_entities(session_id)`) and bounded
by a hard-coded `max_hops=2`. Retrieval is driven by a query string. None of
that is "proactive": nothing lights up because the conversation *mentioned*
something.

## Decisions taken (user-chosen)

| # | Question | Decision |
|---|----------|----------|
| 1 | How is "related enough" decided without a hop limit? | **Weight-decayed activation spreading** — depth emerges from weight |
| 2 | What makes an edge heavy? | **Raw co-occurrence count, no decay** |
| 3 | What seeds activation? | **Live entities from the conversation + the active task's entities at lower activation** |

Deferred (recorded, not built): weight decay over time (a read-time formula
change, not a migration); GDS algorithms — Neo4j here has **0 GDS procedures**
(190 APOC), so PageRank/community detection would need the plugin or a library;
the activation interface is shaped so personalized PageRank can replace it.

## 1. Weights

`RELATED_TO` and `MENTIONS` gain `count` (int):

```
MERGE (a)-[r:RELATED_TO]->(b)
ON CREATE SET r.count = 1
ON MATCH  SET r.count = r.count + 1
```

Pre-existing edges without `count` read as 1 (existence was one observation) —
`coalesce(r.count, 1)` on read, no migration.

Every edge type has a static **prior** (evidence quality, config):

| Edge | Prior |
|---|---|
| `ADVANCES` | 1.0 |
| `MENTIONS`, `HAS_ALIAS`, `PURSUES`, `INVOKED`, `HAS_EPISODE`, `PARTICIPATED_IN` | 0.9 |
| `DECIDED`, `PREFERS` | 0.8 |
| `RELATED_TO` | 0.5 |

Effective weight: `prior × log(1 + count) / log(1 + cap)`, `cap = 20`, clamped
to `[0, prior]`. Logarithmic so the 50th observation adds little; capped so one
saturating session cannot dominate a link.

## 2. Activation spreading — pure function in Python

One Cypher round-trip pulls the neighborhood (nodes + edges with type and count)
into an adjacency map; spreading runs as a pure function over it:

```
activation[child] = max(activation[child],
                        activation[parent] × w_eff(edge) × decay_per_hop)
```

Propagate breadth-first while any frontier node exceeds `floor`. **No hop
limit** — depth is a consequence of weight. Cycles terminate because activation
strictly decreases per hop (`decay_per_hop < 1`) and a node only re-enters the
frontier when its activation *increases*.

Why not Cypher: activation is iterative multiplication with a floor, not a path
pattern; without GDS, Cypher expresses it badly and needs one round-trip per
hop. A pure function is testable on hand-built graphs with no database, and is
swappable.

Neighborhood pull is bounded by a *safety* radius (config, default 4) purely to
cap the fetch size; the algorithm never sees it as a semantic limit — anything
strong enough to reach hop 4 would need the radius raised, which the tests
document.

## 3. Seeds

- **1.0** — entities extracted (spaCy) at read time from the L1 recent messages
  and the query, normalized, resolved through `HAS_ALIAS` so aliases collapse
  to the `UserProfile`. Unresolvable names are ignored (nothing to light).
- **0.6** — entities the user's active `Task` already touches
  (`Task<-ADVANCES-Episode-MENTIONS->Entity`). Live evidence outranks inherited.

## 4. Assembly

Nodes above `floor`, ranked by activation (ties: recency), rendered by kind:
`Episode` → its L2 row; `ToolCall` → its L4 row (errors first); `Entity`,
`Task`, `Decision`, `Preference` → graph-fact lines. Rendered as one
**Proactive Context** section that *replaces* the old session-locked
`Graph Facts` section. Recent Conversation, User Profile, Known Failures, and
Relevant Past Episodes are unchanged.

Every source carries `activation` and `path` (the edge chain that lit it), so
any line is explainable. The path comes from `ActivationResult.parents`, which
records the winning parent *at the moment of the max-wins update* — an earlier
version reconstructed it by walking back to the strongest-looking neighbour,
which reported chains that never happened wherever two routes tied or one was
superseded (measured: 174 divergences across 4000 random graphs). Example: `clickhouse -RELATED_TO(7)-> alembic -INVOKED-> tool
call 49`.

## 5. Calibration, not choice

`floor` and `decay_per_hop` are **measured** before being set: activation
distributions on real graphs (the demo's, the agentic scenarios') printed and
inspected, then chosen so genuine two-hop links survive and noise dies at hop
one. Defaults ship with the measurement recorded in config comments, exactly as
the similarity threshold was.

## 6. Contracts under test (TDD)

**Deterministic, no DB:** weight formula (monotone, log-shaped, saturates at
cap, prior-scaled); spreading on hand-built graphs — strong path reaches hop 3,
weak path dies at hop 1, cycle terminates, floor respected, seed max wins on
convergence, empty seeds → empty result; assembly ranking; alias-collapse of
seeds.

**Integration (live Neo4j/Postgres):** `count` increments on re-observation;
legacy edges read as 1; neighborhood pull returns types + counts; seeds resolve
through aliases to the profile.

**Agentic (Ollama):** session B mentions ClickHouse in passing; the alembic
failure and the ClickHouse episode surface with the query naming neither.
Severed-code check: with spreading disabled the same test must fail.

**Regression:** existing 205 stay green; provenance types remain valid.
