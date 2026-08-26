"""Measure activation distributions on a realistic graph so floor/decay are
calibrated, not chosen. Prints, for a few seed choices, the activation reached
at each hop and by each node kind. Run against the live stack after a demo or
agentic run has populated the graph:

    PYTHONPATH=. .venv/bin/python scripts/calibrate_activation.py [seed-entity ...]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

from app.config import settings
from app.db.neo4j import create_driver_from_settings
from app.services.activation import spread_activation
from app.services.neo4j_store import Neo4jStore, normalize_entity_name


def main() -> None:
    task_ids = [a.split("task:", 1)[1] for a in sys.argv[1:] if a.startswith("task:")]
    entity_args = [a for a in sys.argv[1:] if not a.startswith("task:")]
    seeds_in = [normalize_entity_name(a) for a in entity_args] or None
    if task_ids:
        _calibrate_task_lineages(task_ids)
        if not entity_args:
            return
    driver = create_driver_from_settings()
    g = Neo4jStore(driver)
    if seeds_in is None:
        with driver.session() as s:
            # Most-connected entities make the most informative seeds.
            seeds_in = [
                r["n"] for r in s.run(
                    "MATCH (e:Entity)-[r]-() RETURN e.name AS n, count(r) AS d "
                    "ORDER BY d DESC LIMIT 5"
                )
            ]
    print(f"seeds: {seeds_in}")
    print(f"floor={settings.proactive_activation_floor} decay={settings.proactive_decay_per_hop} "
          f"radius={settings.proactive_fetch_radius}\n")

    for seed in seeds_in:
        nb = g.fetch_neighborhood([seed], radius=settings.proactive_fetch_radius)
        act = spread_activation(nb, seeds={f"Entity:{seed}": 1.0},
                                floor=settings.proactive_activation_floor,
                                decay=settings.proactive_decay_per_hop).scores
        by_kind: dict[str, list[float]] = defaultdict(list)
        for nid, a in act.items():
            by_kind[nid.split(":", 1)[0]].append(a)
        print(f"== seed '{seed}': {len(nb.edges)} edges pulled, {len(act)} nodes activated")
        for kind, vals in sorted(by_kind.items(), key=lambda kv: -max(kv[1])):
            vals.sort(reverse=True)
            print(f"   {kind:12} n={len(vals):3}  max={vals[0]:.3f}  median={vals[len(vals)//2]:.3f}  min={vals[-1]:.3f}")
        # Distribution buckets: how much lives just above the floor?
        buckets = Counter(
            "≥0.5" if a >= 0.5 else "0.2–0.5" if a >= 0.2 else "0.1–0.2" if a >= 0.1 else "floor–0.1"
            for a in act.values()
        )
        print(f"   buckets: {dict(buckets)}\n")
    driver.close()


def _calibrate_task_lineages(task_ids: list[str]) -> None:
    """Seed the given tasks' lineages the way retrieval does (spec §4c) and
    print what lights up, by kind — the Task-seed counterpart of the entity
    calibration above."""
    from app.services.proactive import lineage_task_seeds
    from app.services.task_store import TaskStore

    driver = create_driver_from_settings()
    g = Neo4jStore(driver)
    store = TaskStore(driver)
    print(f"task-node seed={settings.proactive_task_node_seed} "
          f"depth-decay={settings.proactive_task_depth_decay} "
          f"floor={settings.proactive_activation_floor} decay={settings.proactive_decay_per_hop}\n")
    for tid in task_ids:
        lineage = store.get_lineage_ids(tid)
        seeds = lineage_task_seeds(
            lineage,
            base=settings.proactive_task_node_seed,
            decay=settings.proactive_task_depth_decay,
        )
        nb = g.fetch_neighborhood([], task_ids=lineage, radius=settings.proactive_fetch_radius)
        act = spread_activation(nb, seeds=seeds,
                                floor=settings.proactive_activation_floor,
                                decay=settings.proactive_decay_per_hop).scores
        by_kind: dict[str, list[float]] = defaultdict(list)
        for nid, a in act.items():
            by_kind[nid.split(":", 1)[0]].append(a)
        print(f"== task lineage ({len(lineage)} deep) from {tid}: "
              f"{len(nb.edges)} edges pulled, {len(act)} nodes activated")
        for kind, vals in sorted(by_kind.items(), key=lambda kv: -max(kv[1])):
            vals.sort(reverse=True)
            print(f"   {kind:12} n={len(vals):3}  max={vals[0]:.3f}  median={vals[len(vals)//2]:.3f}  min={vals[-1]:.3f}")
        print()
    driver.close()


if __name__ == "__main__":
    main()
