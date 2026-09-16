"""Render eval results as a markdown report. Pure: results in, text out."""

from __future__ import annotations

from collections import defaultdict

from evals.runner import ProbeOutcome, RepetitionResult
from evals.scoring import aggregate

CONDITION_ORDER = ("memcache", "full_transcript", "no_memory")


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:,.0f}"


def _agg_line(outcomes: list[ProbeOutcome]) -> dict:
    ok = [o for o in outcomes if o.error is None]
    coverage = aggregate([o.answer_coverage for o in ok])
    recalls = [o.retrieval_recall for o in ok if o.retrieval_recall is not None]
    return {
        "coverage": coverage,
        "recall": aggregate(recalls),
        "tokens": aggregate([float(o.context_tokens) for o in ok]),
        "latency": aggregate([o.latency_seconds for o in ok]),
        "ok": len(ok),
        "total": len(outcomes),
    }


def render_markdown(results: list[RepetitionResult], title: str) -> str:
    """One report: a table per (scenario, history size), then a summary."""
    grouped: dict[tuple[str, int], dict[str, list[ProbeOutcome]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failed_reps: dict[tuple[str, int], int] = defaultdict(int)
    failures: list[str] = []
    degraded: list[str] = []
    for rep in results:
        if rep.error:
            failed_reps[(rep.scenario, rep.history_size)] += 1
            failures.append(
                f"{rep.scenario} h{rep.history_size} r{rep.repetition}: {rep.error}"
            )
            continue
        for o in rep.outcomes:
            grouped[(rep.scenario, rep.history_size)][o.condition].append(o)
            if o.error:
                failures.append(
                    f"{rep.scenario} h{rep.history_size} r{rep.repetition} "
                    f"[{o.condition}]: {o.error}"
                )
            if o.warnings:
                degraded.append(
                    f"{rep.scenario} h{rep.history_size} r{rep.repetition} "
                    f"[{o.condition}]: {'; '.join(o.warnings)}"
                )

    lines = [f"# {title}", ""]
    lines += [
        "Conditions share identical seeded sessions within each repetition; "
        "only the context mechanism differs. The memcache context is capped "
        "at its production default of 1,200 tokens while full_transcript is "
        "deliberately uncapped: paying whatever the history costs is that "
        "approach's nature, and the token column records the price. Cells "
        "are mean (min..max) over repetitions. Coverage is the fraction of "
        "required facts present in the answer; retrieval recall is those "
        "facts' presence in the retrieved context itself, memcache only.",
        "",
    ]

    for (scenario, size), by_condition in sorted(grouped.items()):
        lines += [f"## {scenario}, {size} sessions of history", ""]
        lines += [
            "| Condition | Answer coverage | Retrieval recall | Context tokens | Latency (s) | Runs |",
            "|-----------|-----------------|------------------|----------------|-------------|------|",
        ]
        for condition in CONDITION_ORDER:
            outcomes = by_condition.get(condition, [])
            if not outcomes:
                continue
            a = _agg_line(outcomes)
            cov, rec, tok, lat = a["coverage"], a["recall"], a["tokens"], a["latency"]
            cov_cell = (
                f"{_fmt(cov.mean, pct=True)} ({_fmt(cov.minimum, pct=True)}"
                f"..{_fmt(cov.maximum, pct=True)})"
                if cov
                else "n/a"
            )
            tok_cell = (
                f"{_fmt(tok.mean)} ({_fmt(tok.minimum)}..{_fmt(tok.maximum)})"
                if tok
                else "n/a"
            )
            lat_cell = (
                f"{lat.mean:.1f} ({lat.minimum:.1f}..{lat.maximum:.1f})"
                if lat
                else "n/a"
            )
            lines.append(
                f"| {condition} | {cov_cell} "
                f"| {_fmt(rec.mean, pct=True) if rec else 'n/a'} "
                f"| {tok_cell} "
                f"| {lat_cell} "
                f"| {a['ok']}/{a['total']} |"
            )
        if failed_reps.get((scenario, size)):
            lines.append("")
            lines.append(
                f"Failed repetitions not in the table above: "
                f"{failed_reps[(scenario, size)]} (see the footer)."
            )
        lines.append("")

    lines += _summary(grouped)
    if degraded:
        lines += ["## Degraded retrievals", ""]
        lines += [
            "These measurements ran against a retrieval endpoint that "
            "reported partial failure; their numbers are included above.",
            "",
        ]
        lines += [f"- {d}" for d in degraded]
        lines.append("")
    if failures:
        lines += ["## Failed measurements", ""]
        lines += [f"- {f}" for f in failures]
        lines.append("")
    return "\n".join(lines)


def _summary(grouped) -> list[str]:
    """Condition x history size, averaged across scenarios."""
    by_cond_size: dict[tuple[str, int], list[ProbeOutcome]] = defaultdict(list)
    for (_, size), by_condition in grouped.items():
        for condition, outcomes in by_condition.items():
            by_cond_size[(condition, size)].extend(outcomes)
    if not by_cond_size:
        return []
    sizes = sorted({size for _, size in by_cond_size})
    lines = ["## Summary across scenarios", ""]
    header = "| Condition | " + " | ".join(
        f"coverage @ {s} | tokens @ {s}" for s in sizes
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + 2 * len(sizes)))
    for condition in CONDITION_ORDER:
        cells = []
        for size in sizes:
            a = _agg_line(by_cond_size.get((condition, size), []))
            cov, tok = a["coverage"], a["tokens"]
            cells.append(_fmt(cov.mean, pct=True) if cov else "n/a")
            cells.append(_fmt(tok.mean) if tok else "n/a")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines
