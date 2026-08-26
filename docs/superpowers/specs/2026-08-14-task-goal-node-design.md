# Task/Goal Node — Design

**Date:** 2026-08-14
**Status:** Approved, implementing
**Branch:** `workbench`

## Problem

The user's stated goal for the L4 workbench is that an agent should *already
know* which tool calls failed **for a task** and avoid repeating that workflow.
But tasks do not exist in the graph. Entities and episodes are only proxies —
"what failed while ClickHouse was being discussed" is topical adjacency, not
task identity. Two different ClickHouse migrations are indistinguishable.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | How does a task come into existence? | Inferred from conversation (user chose inference over explicit ids) |
| 2 | How are two goal statements matched to one task? | LLM adjudication against the user's open tasks |
| 3 | When does a task close? | The model decides, from the same adjudication call (no staleness backstop, no explicit endpoint) |

Consequence accepted by the user: with no backstop, if the model never says
"done", open tasks accumulate. The candidate cap (below) bounds the prompt, not
the graph.

## Schema (Neo4j)

```
(:Task {id, title, status, created_at, updated_at, closed_at})
(:UserProfile)-[:PURSUES]->(:Task)
(:Episode)-[:ADVANCES]->(:Task)
```

- `id`: UUID assigned at creation. Uniqueness constraint.
- `status`: `open` | `done`.
- `title`: the model's extracted goal statement.
- `updated_at` advances whenever an episode links, giving "most recently
  active" ordering for the candidate list.

Task lives in Neo4j, not Postgres: it is a relational concept connecting
episodes, decisions, and (next spec) tool calls, and needs no vector search.

## Adjudication

A **separate** Ollama call after summarization — NOT folded into the summary
call. Asking a 3B model for prose and strict JSON in one response risks a
corrupted summary, damaging L2 to save a round trip the async worker doesn't
need to save.

Input: the episode summary plus the user's **20 most-recently-active open
tasks** (`task_candidate_limit`, enforced in code). Output, strict JSON:

```json
{"goal": "<short imperative phrase or null>",
 "matches_task_id": "<uuid or null>",
 "task_complete": false}
```

Worker flow (only when `user_id` present), after L3 + profile writes:

1. `goal` null → nothing happens; not every episode advances a task.
2. `matches_task_id` valid → `(:Episode)-[:ADVANCES]->(:Task)`; if
   `task_complete`, task closes.
3. Otherwise → new Task created with `PURSUES` from the profile, episode linked.

## Failure handling

Parsing is defensive: malformed JSON, fenced/prose-wrapped JSON, missing
fields, wrong types, or a hallucinated `matches_task_id` all degrade to **no
task attachment**, logged. A hallucinated id does not fall back to creating a
new task — that would mint spurious duplicates from the model's worst outputs.

**An ingest never fails because task inference failed.** The entire task step
is wrapped defensively; L1–L3 are unaffected.

## Retrieval

When `user_id` is present, the profile section gains one line for the most
recently active open task: `Current task: <title>`. Source type `task`,
tier L3.

## Testing

Follows the established rule: LLM judgement is a reported metric, never a gate.

**Deterministic (gates):** JSON parsing across malformed/wrapped/partial
responses; hallucinated-id rejection; candidate capping; store CRUD and status
transitions against live Neo4j; worker linkage with the adjudicator mocked;
ingest survives adjudicator failure; retrieval line.

**Metric (not gates):** with real Ollama, do two paraphrases of one goal merge
into one task or fragment into two? Reported as `[metric]` output.

**Empirical gate decision:** whether the real-Ollama end-to-end test (explicit
goal statement → Task node exists) is stable enough to gate is measured during
implementation, not assumed. If it flakes, it demotes to a metric and the
mocked-path test remains the gate. The decision and evidence are documented.

## Scope

Task node, inference, linkage, retrieval line. Tool-call linkage is the
workbench spec, which gains a `task_id` column and inherits precision from this
tier via `(:Task)<-[:ADVANCES]-(:Episode)-[:INVOKED]->(:ToolCall)`.
