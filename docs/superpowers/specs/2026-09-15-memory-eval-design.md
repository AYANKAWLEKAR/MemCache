# Memory impact evaluation: design

Date: 2026-09-15
Status: proposed

## Goal

Produce defensible numbers for the question the README currently answers only with examples: how much does MemCache improve an agent's recall and response quality compared to the realistic alternative, and at what token cost. The primary competitor is a regular agent thread that carries the full transcript of all prior sessions in its context window. The output is a reproducible evaluation run that emits a results table suitable for the README.

## Staging

The build is staged. This document specifies stage 1 in full and records the intended shape of the later stages so that stage 1 interfaces do not have to change.

- **Stage 1 (this spec):** internal evaluation suite. Conditions: MemCache, full transcript, no memory. Metrics: retrieval recall, answer fact coverage, context tokens, latency.
- **Stage 2a (delivered 2026-09-17):** the naive RAG condition: plain vector search over raw turns, no summarization, no graph. The corpus is every raw message plus each session's recorded tool-result lines (the same fairness rule as full_transcript), embedded with the same MiniLM model MemCache uses; documents are ranked by cosine similarity against the probe question and packed under the same 1,200-token cap as MemCache, presented chronologically once selected, so the two retrieval approaches compete at an identical budget.
- **Stage 2b (later):** a calibrated pairwise LLM judge for answer quality beyond fact coverage.
- **Stage 3 (later):** adapt a subset of a public long-term-memory benchmark (LongMemEval or LoCoMo) for externally citable numbers.

## Definitions

**Condition.** A way of supplying (or not supplying) memory to the answering agent. All conditions use the same model (`qwen3:4b`, the demo agent), the same temperature (0), the same `/no_think` switch, and the same question text. Only the context differs.

| Condition | Context given to the agent |
|-----------|---------------------------|
| `memcache` | The document returned by `POST /memory/retrieve` (`max_tokens=1200`), queried from a fresh session |
| `full_transcript` | Every prior session's raw turns plus its recorded tool results, verbatim, in chronological order, uncapped. Tool results are included because a real thread would have seen them; omitting them would make the failure-recall family unanswerable for this condition by construction rather than by memory quality |
| `naive_rag` | Top raw messages and tool-result lines by embedding similarity to the question, packed under the same 1,200-token cap as `memcache` |
| `no_memory` | None |

The `full_transcript` condition is deliberately uncapped: its nature is that it pays whatever the history costs. The token metric records that cost.

**Eval scenario.** An extension of the existing `tests/agentic` `Scenario` contract. A scenario declares planned multi-session conversations (model-improvised wording around mandatory anchor strings, with deterministic fallbacks) and, new for the eval, one or more probes:

```python
@dataclass(frozen=True)
class EvalProbe:
    question: str              # asked after all ingestion
    required_facts: list[str]  # substrings that a correct answer must contain
    probe_from_session: int | None = None  # None: fresh session (empty L1);
                                           # int: continue that seeded session
```

`probe_from_session` mirrors the demo runtime's `retrieve_from_session`. Most probes run from a fresh session so the context is purely cross-session memory. A probe whose mechanism depends on the live conversation (the passing-mention family, where the recent turns seed the graph walk) continues its seeded session instead; the `full_transcript` condition includes all sessions either way, so the comparison stays fair.

`required_facts` follow the same rule as the existing anchor contract: only strings that are unambiguous under case-insensitive substring matching (an error message, a proper name, a technology term). No fact that requires interpretation to verify.

**History size.** The total number of sessions ingested per run: the scenario's fact-bearing sessions interleaved among distractor sessions, with the fact-bearing sessions placed in the earliest third of the ordering so the probed facts are genuinely buried. The eval runs each scenario at history sizes of 10, 25, and 50 total sessions. There is no small configuration: the floor of 10 sessions exists so that every condition, including the smallest, mirrors sustained personal-agent use rather than a toy exchange.

Distractor sessions are realistic personal-agent conversations, not filler lines: 8 to 12 turns each (roughly 230 to 310 tokens, about 260 on average, enforced by unit test), covering the kinds of exchanges an assistant accumulates in ordinary use (scheduling, debugging help, drafting, planning), written as fixed deterministic text and sharing no anchors with any probe. At these sizes the full transcript is on the order of 3k tokens at 10 sessions, 7k at 25, and 13k or more at 50, so the top size presses against the practical context handling of the answering model. The answering call pins its context window explicitly (32k by default) so that an install-dependent Ollama truncation default cannot silently delete the buried facts and masquerade as a result. The full transcript still contains every probed fact by construction at all sizes; the informative results are the token cost at every size and the coverage trend as the history grows.

## Metrics

Per (scenario, condition, history size), aggregated over N repetitions (default 5) of the full seed-and-query cycle:

1. **Answer fact coverage.** Fraction of `required_facts` present in the agent's answer, case-insensitive, whitespace-normalized. The primary quality metric. Deterministic; no judge involved.
2. **Retrieval recall** (`memcache` condition only). Fraction of `required_facts` present in the retrieved context document, before any agent involvement. Separates retrieval failures from generation failures: if a fact is absent from the answer but present in the context, the retrieval tier is exonerated.
3. **Context tokens.** Token count of the supplied context, measured with tiktoken (already a dependency). Zero for `no_memory`.
4. **Latency.** Seconds for the generation call.

Reported as mean and range across repetitions. Repetition matters because the seeded traffic is model-improvised; temperature 0 does not remove that source of variance.

## Architecture

New top-level package `evals/`, importable and unit-testable, with a thin script entry point. The `tests/agentic` harness modules (`ollama_agent`, planned-turn generation) are reused as imports. The eval's own unit and smoke tests under `tests/` import from `evals/`; nothing in `app/` or `frontend/` does.

```
evals/
    __init__.py
    scenarios.py     # EvalProbe, eval scenarios, distractor generator
    conditions.py    # Condition protocol + the three stage-1 implementations
    scoring.py       # pure functions: fact_coverage, token counts, aggregation
    report.py        # results -> markdown table
scripts/run_eval.py  # CLI: orchestrates runs, writes evals/results/<date>.md
```

**Condition protocol.** One method: `build_context(run, probe, probe_index) -> BuiltContext`, where `run` is the shared record of the ingested sessions and `BuiltContext` carries the context string plus, for `memcache`, the provenance sources and any degradation warnings the endpoint reported (degraded measurements are included in the results and listed in their own report section). `memcache` calls the live retrieve endpoint; `full_transcript` concatenates the stored session turns with session-boundary markers; `no_memory` returns None. Stage 2's naive RAG implements the same protocol, which is the reason the protocol exists now.

**Isolation.** Each repetition uses a fresh user id under the eval's own `demo-ui-eval-` prefix. Before every repetition the runner wipes ALL prior eval data prefix-wide (Redis keys, Postgres rows, Neo4j profiles and sessions), because a per-user reset is not enough: aliases are permanent by design, so the previous repetition's profile would silently keep this scenario's person names, and Entity nodes are global, so RELATED_TO observation counts would accumulate across repetitions and inflate later runs. The scenario declares its person names and the runner releases them from eval-prefixed profiles; entities left with no MENTIONS from any surviving episode and no alias are pruned, which resets eval-only entities while any entity referenced by real data keeps its edges and is untouched. This prune is the only operation that reaches beyond the eval prefix, and it only ever deletes fully orphaned nodes.

**Execution.** In-process API with eager Celery, the pattern already proven by `demo_runtime.bootstrap()`. The run requires the docker-compose stack and both Ollama models; `run_eval.py` checks health first and exits with a specific message naming what is missing. A failed repetition (model timeout, ingest error) is recorded in the results as failed and excluded from aggregation with its count shown; it never silently disappears.

**Stage-1 scenarios.** Four families, each with probes whose facts live in early sessions:

1. Failure recall: a recorded tool failure (error string as the required fact), probed from a fresh session.
2. Identity and preferences: a fresh persona (name, employer, decision, preference) following the onboarding scenario's shape. A fresh persona rather than the onboarding scenario's own names, so eval alias ownership can never collide with the agentic suite's fixtures; the scenario declares its person names and the runner releases them from eval profiles before each repetition.
3. Goal lineage: a planted goal tree, probing for the root goal from a leaf context (tree planted deterministically, consistent with the measured 3B limitation documented in the README).
4. Passing mention: the probe continues the session containing an offhand mention (`probe_from_session`); the probe question names nothing, and the required fact is reachable only through the graph walk seeded by that mention. This family is expected to show the largest gap over `full_transcript` at high history sizes, and its result is reported either way.

## Fairness rules

- Same model, sampling settings, and prompt template across conditions; only the context block differs.
- The probe question never contains a required fact or an anchor.
- Distractor content is fixed text, identical across conditions within a repetition.
- MemCache's context stays capped at its production default (1200 tokens) while `full_transcript` is uncapped; the asymmetry is the point of the comparison and is stated in the report.

## Error handling

- The full matrix (4 families x 3 conditions x 3 history sizes x 5 repetitions) is on the order of 180 seed-and-query cycles, and with 10 to 50 sessions ingested per cycle it runs for hours on local models. `run_eval.py` therefore takes `--scenarios`, `--history-sizes`, and `--repetitions` flags, and seeding is shared across the three conditions within a repetition (one ingest, three question passes), which divides the seeding cost by three. Distractor sessions are identical fixed text across repetitions, so their summaries and graph writes are the only per-repetition cost that scales with history size. The measurement order of the three conditions rotates per repetition so no condition systematically answers first, which would bias the latency metric. The runner reports results incrementally and the CLI persists the raw JSON after every repetition, rendering a partial report on interrupt, so a crash or Ctrl-C loses at most the repetition in flight.
- Missing stack or models: fail before any run, with the exact `docker compose` / `ollama pull` remediation printed.
- Mid-run failures: per-repetition capture, reported in the table footer.
- The script is idempotent: each run writes a new dated results file and never overwrites a previous one.

## Testing

- `evals/scoring.py` and `evals/report.py` are pure and get plain unit tests (no marker, run in CI).
- `evals/conditions.py` context assembly for `full_transcript` and `no_memory` is pure given stored sessions and gets unit tests.
- Smoke tests, marked `agentic`, run the failure-recall family at the minimum history size (10) with N=1 through the orchestration function and assert a well-formed results structure and multi-tier provenance, plus one pass over the goal-planting and continued-probe paths (goal lineage and passing mention), shape assertions only.
- The full eval run is not a CI gate. It is a measurement, run manually, consistent with the repository's existing treatment of model-dependent measurements.

## Out of scope for stage 1

- Naive RAG condition (stage 2).
- LLM-judged answer quality (stage 2, and only after judge calibration on known-winner pairs).
- Public benchmark adaptation (stage 3).
- Any change to retrieval behavior itself; the eval observes the system, it does not tune it. If results motivate tuning, that is separate work measured by rerunning the eval.
- Time depth. Burial is positional: all sessions are ingested minutes apart, so recency decay over calendar time is not exercised. Backdating episode timestamps to give the history real age is a stage 2 candidate.

## Acceptance

Stage 1 is done when `scripts/run_eval.py` completes on the live stack and produces a markdown table of the four scenario families under three conditions at three history sizes, with fact coverage, retrieval recall, context tokens, and latency, aggregated over five repetitions, and the unit and smoke tests pass.
