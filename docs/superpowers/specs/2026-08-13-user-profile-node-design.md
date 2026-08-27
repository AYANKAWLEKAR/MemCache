# UserProfile Node — Design

**Date:** 2026-08-13
**Status:** Approved, ready for implementation planning
**Branch:** `api-testing`

## Problem

MemCache has no notion of who is speaking. `MemoryIngestRequest` carries
`session_id`, `messages`, `metadata` — nothing identifies the person. "Dana
Whitfield" is stored as an `Entity` node structurally identical to "ClickHouse".

Two consequences, both observed live during the audit (`steps taken/audit_report.md`):

1. **Identity fragments.** `Dana` and `Dana Whitfield` become two `Entity` nodes,
   and co-occurrence links them with a spurious `RELATED_TO` — the same person
   appearing as two things that happen to be related.
2. **Nothing persists across sessions.** A second session retrieving context got
   back only its own L1 messages. Identity, decisions, and preferences from the
   first session were absent, not degraded.

## Goal

A canonical `UserProfile` node that:

- anchors identity on a stable key that does not depend on NER,
- absorbs surface-form aliases so one person is one identity,
- carries `name`, `gender`, `title`, `role`, `location` with provenance,
- connects to the sessions, decisions, and preferences that belong to the user.

## Decisions taken

| # | Question | Decision |
|---|----------|----------|
| 1 | Identity source | Explicit `user_id` supplied by the caller |
| 2 | Alias relationship | Overlay — profile links to `Entity` nodes, does not absorb them |
| 3 | Alias learning | Rule-based (self-reference + subset) **and** explicit registration API |
| 4 | Gender | Inferred from stated evidence, with explicit override |

### Rationale

**Explicit `user_id`** makes the profile the one element of the graph that does
not depend on extraction being correct. Inferring the anchor from conversation
would make identity as unreliable as the NER it exists to fix — measured at ~75%
detection for single-token proper nouns.

**Overlay rather than absorb.** Because extraction is unreliable, identity
decisions must be cheap to reverse. An overlay makes a bad alias one edge to
unlink. Absorbing rewrites `MENTIONS` and destroys the distinction between two
people if the merge was wrong.

**Gender inferred from stated evidence only.** Inference is drawn from pronoun
declarations and explicit self-identification, never from names. Every inferred
value stores the sentence it came from, so a wrong inference is visible and
correctable rather than an unexplained property. Explicit values always win.

## Schema

```
(:UserProfile  {user_id, display_name, created_at, updated_at})
(:ProfileAttribute {id, key, value, source, confidence, observed_at, evidence})

(:UserProfile)-[:HAS_ATTRIBUTE]->(:ProfileAttribute)
(:UserProfile)-[:HAS_ALIAS {source, confidence, linked_at}]->(:Entity)
(:UserProfile)-[:PARTICIPATED_IN]->(:Session)
(:UserProfile)-[:DECIDED]->(:Decision)
(:UserProfile)-[:PREFERS]->(:Preference)
```

Constraints: `UserProfile.user_id` unique, `ProfileAttribute.id` unique. Added to
`ensure_constraints` in `app/db/neo4j.py`.

### Attributes as nodes, not properties

`key` ∈ `{name, gender, title, role, location}`. `source` ∈ `{explicit, inferred}`.

The five requested fields are not homogeneous: `name` is stable, `title`/`role`/
`location` change over time, and `gender` is inferred. Flat properties cannot
express "Staff Engineer at Northwind, now Principal at Vertex", nor record that a
value was inferred with 0.6 confidence from a particular sentence.

`ProfileAttribute.id` is `f"{user_id}:{key}:{sha256(value)[:16]}"`, matching the
stable-synthetic-id pattern already used for `Decision`/`Preference`, so reruns
stay idempotent.

**Current value resolution** — a single function, one read path:

1. `source == "explicit"` beats `source == "inferred"`
2. then most recent `observed_at`
3. then highest `confidence`

## Attribute extraction

All five keys populate the same way: pattern-matched from `role: "user"` messages
as `source: "inferred"`, or set via `PATCH` as `source: "explicit"`. Explicit
always wins. Every inferred value stores the matched sentence in `evidence`.

| Key | Inferred from | Confidence |
|-----|---------------|------------|
| `name` | Rule 1 self-reference patterns (shared with alias resolution) | 0.9 |
| `title` | `I'm a/an <title>`, `I work as <title>`, `my title is <title>` — captured span must not be a spaCy `PERSON`/`ORG` | 0.6 |
| `role` | `I'm the <role> on/for <X>`, `my role is <role>`, `I lead <X>` | 0.6 |
| `location` | `I'm based in <X>`, `I live in <X>`, `I'm in <X>` — captured span must be a spaCy `GPE` or `LOC` | 0.7 |
| `gender` | Pronoun declarations and explicit self-identification only (below) | 0.8 |

`title` and `role` overlap in natural speech ("I'm a staff engineer" is arguably
both). They are kept as separate keys because the user asked for both; when a
sentence matches both patterns, `title` wins and `role` is not written, so one
utterance never produces two competing attributes.

Requiring spaCy label agreement on `title` and `location` is what keeps
"I'm a bit lost" from becoming a title and "I'm in trouble" from becoming a
location. Where no label check is possible (`role`), confidence stays low and the
value remains overridable.

### Gender

Inferred from **stated evidence only** — never from a name, and never from
honorifics:

```
my pronouns are <X>  |  I use <X>/<Y>  |  I'm a woman|man|nonbinary person
```

The stored `value` is the stated string as given, not a normalized enum, so
self-descriptions that do not fit a fixed vocabulary are preserved rather than
coerced. `PATCH` overrides permanently, and an explicit value is never
re-inferred over.

## Connecting to related nodes

`Decision` and `Preference` are promoted from episode scope to profile scope.
Existing `(:Episode)-[:DECIDED]->` edges remain as provenance; the new profile
edges make "everything Dana has ever decided" a single-hop query across all
sessions.

Related **entities** are derived by traversal through aliases, not materialized.
Materializing an edge to every entity the user mentions turns the profile into a
hub connected to everything, which carries no information.

## Alias resolution

Two pure functions over message lists, in `profile_extraction.py`:

**Rule 1 — self-reference.** A first-person introduction in a `role: "user"`
message aliases that entity to the speaking profile. The person is stating who
they are; this is the strongest available evidence.

An introduction is a match of one of these leading patterns, where the captured
span must also be a `PERSON` entity that spaCy extracted from the same message:

```
I'm <X>  |  I am <X>  |  my name is <X>  |  this is <X>  |  call me <X>
```

Requiring the span to coincide with a `PERSON` entity is what prevents
"I'm exhausted" from aliasing the profile to "exhausted". Messages with
`role != "user"` are never considered.

**Rule 2 — subset.** A shorter name aliases only when its tokens are a subset of
an already-confirmed alias (`dana` ⊂ `dana whitfield`) **and** it appears within
that profile's own sessions. Deliberately narrow.

**Explicit registration.** `POST /profile/{user_id}/alias` writes
`source: "explicit"`.

### Known limits

- Nicknames and initials ("Dan", "DW") are not matched. Rule 2 is token-subset
  only.
- If the user never introduces themselves, rules yield nothing and the profile
  sits empty. Explicit registration is the escape hatch.
- A genuinely different person sharing a first name within the same profile's
  sessions can be wrongly aliased. The overlay makes this one unlink to fix.

### Conflicts

Aliasing one `Entity` to two profiles raises `ProfileAliasConflictError`,
following the `EpisodeCollisionError` precedent: identity ambiguity fails loudly
rather than silently merging two people. Not retried by the worker — a conflict
is permanent, not transient.

## Wiring

**API.** `user_id: str | None` added to `MemoryIngestRequest`. When absent, no
profile work occurs and behavior is byte-identical to today. This keeps the
change additive: existing sessions, callers, and the current 74-test suite are
unaffected.

New endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/profile/{user_id}` | Resolved profile: current attribute values, aliases |
| `POST` | `/profile/{user_id}/alias` | Register an explicit alias |
| `PATCH` | `/profile/{user_id}` | Set attributes explicitly (overrides inference) |

**Worker.** `process_conversation` calls profile resolution after `_write_l3`,
inside the existing Neo4j try/except so failures follow the established retry
path.

**Retrieval.** `_format_graph_facts` resolves through the profile, emitting one
canonical identity block instead of fragments. Profile-scoped decisions and
preferences are included, which is what delivers cross-session recall for graph
facts.

## Scope boundary

This delivers cross-session recall for **graph facts** — identity, decisions,
preferences — because those resolve through the profile.

It does **not** change L2. `search_episodes` still filters by `session_id`, so
semantic episode retrieval remains session-locked. After this work a new session
knows who you are, what you decided, and what you prefer, but still cannot
semantically retrieve past conversations. That is a separate change and is
deliberately excluded here.

## Testing

**Deterministic unit tests** (no Ollama — pure functions):

- Rule 1 fires on first-person introductions; does not fire on `role: "assistant"`
  messages or third-person mentions.
- Rule 2 matches token subsets of confirmed aliases only; rejects unrelated names.
- Attribute resolution precedence: explicit > inferred; recency; confidence.
- `ProfileAliasConflictError` raised when an entity is claimed by two profiles.
- Gender inferred from a pronoun declaration; never from a name alone; explicit
  value overrides an existing inference.
- `title` and `location` reject spans that fail their spaCy label check
  ("I'm a bit lost" yields no title; "I'm in trouble" yields no location).
- A sentence matching both `title` and `role` patterns writes only `title`.
- `title`/`role`/`location` supersede by recency: a later "I'm now at Vertex Labs"
  resolves over the earlier value, with both retained as history.

**Agentic scenarios** (`tests/agentic/`):

- *Identity collapse*: a conversation producing both `Dana` and `Dana Whitfield`
  yields one profile with two aliases, and retrieval shows one canonical block.
- *Cross-session*: session A states a decision; session B, same `user_id`,
  retrieves it. This is the scenario that fails today.

Names used in scenarios must be multi-word — single-token invented names detect
at only ~75% and belong in `optional_entities`, per the harness convention.

**Regression:** the existing 74 tests must stay green, since `user_id` is optional.

## Files

**New:** `app/services/profile_store.py` (Neo4j operations),
`app/services/profile_extraction.py` (rules, pure functions).

**Modified:** `app/api/models.py`, `app/api/routes.py`, `app/workers/tasks.py`,
`app/services/retrieval.py`, `app/db/neo4j.py`.

## Migration

Entirely additive. Existing graphs have no `UserProfile` nodes and every new
label, relationship, and field is optional. No backfill is required; a graph
written before this change continues to work with profile features simply
inactive.
