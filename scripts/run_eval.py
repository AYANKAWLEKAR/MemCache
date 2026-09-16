"""Run the memory impact evaluation and write a dated report.

Usage (stack + both Ollama models required):

    .venv/bin/python scripts/run_eval.py
    .venv/bin/python scripts/run_eval.py --scenarios failure_recall \
        --history-sizes 10 --repetitions 1

Design: docs/superpowers/specs/2026-09-15-memory-eval-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.report import render_markdown  # noqa: E402
from evals.runner import (  # noqa: E402
    preflight,
    run_matrix,
    scenarios_by_name,
    utcnow_stamp,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="scenario names (default: all four families)")
    parser.add_argument("--history-sizes", nargs="*", type=int, default=[10, 25, 50])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out-dir", default="evals/results")
    args = parser.parse_args()

    problems = preflight()
    if problems:
        print("The eval needs the full stack. Unhealthy dependencies:")
        for p in problems:
            print(f"  - {p}")
        print("\nRemediation:")
        print("  docker compose up -d redis postgres neo4j")
        print("  ollama pull qwen2.5:3b && ollama pull qwen3:4b")
        return 1

    scenarios = scenarios_by_name(args.scenarios)
    total = len(scenarios) * len(args.history_sizes) * args.repetitions
    done = 0

    def progress(name: str, size: int, rep: int) -> None:
        nonlocal done
        done += 1
        print(f"[{done}/{total}] {name} history={size} repetition={rep + 1}")

    results = run_matrix(scenarios, args.history_sizes, args.repetitions, progress)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = utcnow_stamp()
    md_path = _fresh_path(out_dir, base, "md")
    json_path = md_path.with_suffix(".json")

    md_path.write_text(render_markdown(results, f"Memory impact evaluation, {base}"))
    json_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print(f"\nReport: {md_path}\nRaw results: {json_path}")

    failed = sum(1 for r in results if r.error) + sum(
        1 for r in results for o in r.outcomes if o.error
    )
    if failed:
        print(f"WARNING: {failed} failed measurement(s), see the report footer.")
    return 0


def _fresh_path(out_dir: Path, base: str, ext: str) -> Path:
    """Dated filename that never overwrites an earlier run."""
    candidate = out_dir / f"{base}.{ext}"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{base}-{n}.{ext}"
        n += 1
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
