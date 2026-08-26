# MemCache — Portfolio Review (recruiter/evaluator lens)

*An assessment of this repository as a work sample: what it demonstrates, what
the evidence is, and what an interviewer should probe.*

## What the system is

A hand-built, four-tier memory infrastructure for LLM agents, on boring,
inspectable technology — FastAPI, Redis, Postgres+pgvector, Neo4j, Celery,
spaCy, and local Ollama models. No memory framework was imported; every tier
and every edge type is original design:

| Tier | Store | Holds |
|------|-------|-------|
| L1 | Redis | Raw recent turns, TTL'd |
| L2 | Postgres + pgvector | Summarized episodes with embeddings, user-scoped, recency-ranked |
| L3 | Neo4j | The relation graph: sessions, episodes, entities, decisions, preferences, user profiles, **tasks** |
| L4 | Postgres | Tool calls with outcomes, byte-capped payloads, content-hash dedup |

The differentiating claim is the **relation graph across all four**: one Cypher
traversal answers "for this user's goal, which episode advanced it, which
entities were involved, and which action failed" —

```
(:UserProfile)-[:PURSUES]->(:Task)<-[:ADVANCES]-(:Episode)
      (:Episode)-[:MENTIONS]->(:Entity)
      (:Episode)-[:INVOKED]->(:ToolCall {status:'error'})
```

Captured live from the shipped demo (fresh session, same user, zero prompting):

```
User Profile:
User: Dana Whitfield
Current task: Resolve alembic migration issue with telemetry pipeline

Known Failures:
Failed action: alembic — DuplicateColumn: column user_id already exists

Relevant Past Episodes:
Episode 39: Dana Whitfield aims to migrate the telemetry pipeline to ClickHouse...

goal     : 'Resolve alembic migration issue with telemetry pipeline' [open]
entities : ['clickhouse', 'dana whitfield']
failures : ['alembic #46']
```

The agent is warned about the failed migration *unprompted* — failure memory,
which is the product thesis, working end-to-end.

## What the evidence shows (strengths)

**1. Claims are measured, not asserted.** The repo's recurring pattern is
"measure before deciding": the L2 similarity threshold was recalibrated after
measuring real MiniLM score distributions (the shipped 0.7 sat *above the
maximum achievable score* — the entire semantic tier was dead and all tests
were green); NER reliability was quantified per-name (~75% on single-token
invented names) and turned into a metric rather than a flaky gate; the
real-LLM task-inference gate was justified by a 40/40 measured trial, not
optimism.

**2. Tests attack the system, not the mocks.** 203 tests, 5/5 stable
full-suite runs. Integration tests assert via independent SQL/Cypher probes —
never through the API that wrote the data — and the tri-tier tests require
L2/L3 id agreement, a check that caught real cross-linking corruption twice.
An Ollama-driven agent harness generates realistic traffic under a
"model chooses the words, scenario chooses the facts" contract (anchored
strings with deterministic fallback), so model variance can degrade realism
but never test correctness. Tests that passed first-run were verified by
severing the feature and watching them fail.

**3. Failure-mode engineering is explicit.** Identity ambiguity fails loudly
(`EpisodeCollisionError`, `ProfileAliasConflictError`, HTTP 409) instead of
silently merging people or sessions. Inference tiers are additive by contract:
a dead Ollama, malformed JSON, or hallucinated task id degrades to "no
attachment, logged" and can never fail an ingest. A hallucinated id
deliberately does *not* create a new task — the parser treats the model's
worst output as noise, not signal.

**4. The audit trail is unusual for a personal project.** Ten defects found
against a live stack, each with severity, reproduction evidence, and fix;
design docs recording the road not taken (overlay-vs-absorb aliases, column
vs. graph ownership, why reranking lives in Python to keep IVFFlat useful);
an obstacles log distinguishing self-inflicted bugs from environmental ones.

## What an interviewer should probe (honest weaknesses)

- **Extraction ceiling.** Entity/decision extraction is spaCy small-model NER
  plus regex. It was measured and its misses are contained, but coreference
  beyond the user (e.g., third parties) and one-word invented names remain
  weak. The candidate should be able to defend "ship rule-based, measure,
  upgrade to LLM extraction later" as sequencing rather than avoidance.
- **Unmeasured knobs.** Recency half-life (30d) and over-fetch factor (4) are
  admitted guesses; contrast with the calibrated threshold and ask why the
  same rigor wasn't applied (fair answer: no real usage history yet).
- **Single-box scale.** IVFFlat index creation is still manual; task
  candidate lists cap at 20 with no staleness sweep (a user choice, logged
  with its consequence); no retention policy on L4 yet. Growth paths exist on
  paper only.
- **Sequential-session assumption.** Claim-unlinked attribution attaches all
  unlinked tool calls to the next episode; concurrent agents on one session
  would mis-attribute. Known and documented, not solved.
- **Silent alias-conflict path.** In the ingest path, an alias conflict logs
  and skips; only the manual endpoint is loud. Flagged in-repo as an open
  operational question.

## Verdict, as a work sample

This reads like infrastructure built by someone who treats a solo project
with production discipline: specs before code, TDD with observed-red evidence,
measured thresholds, adversarial tests, honest documentation of what is guessed
versus known. The four-way episode–action–entity–goal relation is genuinely
the interesting artifact — it is the part that is *not* reproducible by wiring
up an off-the-shelf RAG stack, and the live demo proves it answers the question
it was built for: *an agent that already knows what failed last time.*
