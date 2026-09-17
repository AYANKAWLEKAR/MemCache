"""Conditions: the ways of supplying (or not supplying) memory to the agent.

Each condition builds the context block for one probe from the same underlying
run record, so all three competitors see identical seeded content. Only the
mechanism differs: `memcache` queries the live retrieval endpoint,
`full_transcript` replays everything verbatim, `no_memory` supplies nothing.

`full_transcript` and `no_memory` are pure over the run record and unit
tested; `memcache` takes an injected retrieve callable so the runner owns the
HTTP side effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from evals.scenarios import EvalProbe, PlacedSession


@dataclass(frozen=True)
class RunRecord:
    """Everything a condition may draw on: the run's exact ingested content.

    `realized_messages[i]` holds the messages actually ingested for
    `placed[i]` (improvised wording included), and `session_ids[i]` its
    session id — so every condition works from the same seeded data.
    """

    user_id: str
    placed: tuple[PlacedSession, ...]
    realized_messages: tuple[tuple[dict, ...], ...]
    session_ids: tuple[str, ...]

    def fact_session_id(self, fact_index: int) -> str:
        for p, sid in zip(self.placed, self.session_ids):
            if p.fact_index == fact_index:
                return sid
        raise KeyError(fact_index)


@dataclass(frozen=True)
class BuiltContext:
    """One condition's context for one probe."""

    condition: str
    context: str | None
    #: Provenance sources, memcache only (used for tier accounting).
    sources: tuple[dict, ...] = ()
    #: Degradation reported by the retrieval endpoint, memcache only. A
    #: degraded retrieval is still measured, but the report must say so.
    warnings: tuple[str, ...] = ()


class NoMemoryCondition:
    """The agent gets nothing beyond the question."""

    name = "no_memory"

    def build_context(
        self, run: RunRecord, probe: EvalProbe, probe_index: int = 0
    ) -> BuiltContext:
        return BuiltContext(condition=self.name, context=None)


class FullTranscriptCondition:
    """Every prior session's raw turns, verbatim, chronological, uncapped.

    Tool failures are included as bracketed lines in their session: a real
    thread would have seen those tool results, and omitting them would make
    the failure-recall family unanswerable for this condition by construction
    rather than by memory quality.
    """

    name = "full_transcript"

    def build_context(
        self, run: RunRecord, probe: EvalProbe, probe_index: int = 0
    ) -> BuiltContext:
        blocks = []
        for i, (placed, messages) in enumerate(zip(run.placed, run.realized_messages)):
            lines = [f"## Session {i + 1}: {placed.session.label}"]
            for tf in placed.session.tool_failures:
                lines.append(f"[tool {tf['tool_name']} {tf['status']}: {tf['error']}]")
            for m in messages:
                lines.append(f"{m['role'].capitalize()}: {m['content']}")
            blocks.append("\n".join(lines))
        return BuiltContext(condition=self.name, context="\n\n".join(blocks))


@dataclass
class NaiveRagCondition:
    """Plain vector search over raw turns: no summarization, no graph.

    The stage-2 baseline that isolates what MemCache's summarization, graph,
    and profile machinery add beyond embeddings alone. The corpus is every
    raw message of every session plus its recorded tool-result lines (the
    same fairness rule as full_transcript), each embedded with the SAME
    model MemCache uses. Documents are ranked by cosine similarity against
    the probe question and packed under the SAME token cap as MemCache, in
    chronological order once selected, so the two retrieval approaches
    compete at an identical budget.
    """

    #: texts -> unit-length embedding vectors. Injected: the runner supplies
    #: the app's SentenceTransformer; tests supply a fake.
    embed_fn: Callable[[list[str]], list[list[float]]]
    max_tokens: int = 1200
    name: str = field(default="naive_rag", init=False)

    def build_context(
        self, run: RunRecord, probe: EvalProbe, probe_index: int = 0
    ) -> BuiltContext:
        from evals.scoring import count_tokens

        docs: list[tuple[int, str]] = []  # (position, text) in corpus order
        for i, (placed, messages) in enumerate(zip(run.placed, run.realized_messages)):
            label = f"Session {i + 1} ({placed.session.label})"
            for tf in placed.session.tool_failures:
                docs.append(
                    (len(docs),
                     f"{label} [tool {tf['tool_name']} {tf['status']}: {tf['error']}]")
                )
            for m in messages:
                docs.append(
                    (len(docs), f"{label} {m['role'].capitalize()}: {m['content']}")
                )
        vectors = self.embed_fn([text for _, text in docs])
        query_vec = self.embed_fn([probe.question])[0]
        scored = sorted(
            zip(docs, vectors),
            key=lambda pair: -_dot(pair[1], query_vec),
        )
        selected: list[tuple[int, str]] = []
        used = 0
        for (pos, text), _vec in scored:
            tokens = count_tokens(text)
            if used + tokens > self.max_tokens:
                continue
            selected.append((pos, text))
            used += tokens
        selected.sort()  # chronological presentation of the selected set
        return BuiltContext(
            condition=self.name,
            context="\n".join(text for _, text in selected),
        )


def _dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class MemcacheCondition:
    """The document returned by POST /memory/retrieve at the production cap.

    A probe runs from a fresh session by default (empty L1, purely
    cross-session memory); a probe with `probe_from_session` continues that
    seeded session so its live recent turns participate.
    """

    #: (session_id, user_id, query, max_tokens) -> retrieve response dict.
    #: Injected by the runner; carries the HTTP call and auth.
    retrieve_fn: Callable[[str, str, str, int], dict]
    max_tokens: int = 1200
    name: str = field(default="memcache", init=False)

    def build_context(
        self, run: RunRecord, probe: EvalProbe, probe_index: int = 0
    ) -> BuiltContext:
        if probe.probe_from_session is not None:
            session_id = run.fact_session_id(probe.probe_from_session)
        else:
            session_id = f"{run.user_id}-probe-{probe_index}"
        response = self.retrieve_fn(session_id, run.user_id, probe.question, self.max_tokens)
        warnings = tuple(response.get("warnings", ()))
        if response.get("status") not in (None, "ok"):
            warnings = (f"retrieval status: {response.get('status')}",) + warnings
        return BuiltContext(
            condition=self.name,
            context=response.get("context", ""),
            sources=tuple(response.get("sources", ())),
            warnings=warnings,
        )
