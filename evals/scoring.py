"""Pure scoring functions for the memory impact evaluation.

Everything here is deterministic and store-free: text in, numbers out. The
matching contract is the repository's anchor rule — case-insensitive substring
presence after whitespace normalization — so a fact "counts" under exactly the
conditions the agentic harness already relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


def normalize(text: str) -> str:
    """Casefold and collapse all whitespace runs to single spaces."""
    return " ".join((text or "").split()).casefold()


@dataclass(frozen=True)
class FactCoverage:
    """Which required facts a text contains."""

    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def fraction(self) -> float:
        total = len(self.present) + len(self.missing)
        return len(self.present) / total if total else 1.0


def fact_coverage(text: str, required_facts: list[str]) -> FactCoverage:
    """Score a text against required facts by normalized substring presence."""
    haystack = normalize(text)
    present, missing = [], []
    for fact in required_facts:
        (present if normalize(fact) in haystack else missing).append(fact)
    return FactCoverage(present=tuple(present), missing=tuple(missing))


def count_tokens(text: str) -> int:
    """Token count, identical to app.services.retrieval._count_tokens.

    Mirrored rather than imported so the eval does not depend on a private
    function; test_eval_scoring pins the two implementations together.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


@dataclass(frozen=True)
class Aggregate:
    """Mean and range over a metric's repetitions."""

    mean: float
    minimum: float
    maximum: float
    n: int


def aggregate(values: list[float]) -> Aggregate | None:
    """Aggregate repetition values; None when every repetition failed."""
    if not values:
        return None
    return Aggregate(
        mean=sum(values) / len(values),
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )
