# UserProfile Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical `UserProfile` node that anchors user identity on an explicit `user_id`, absorbs surface-form aliases so one person is never several entities, and carries `name`/`gender`/`title`/`role`/`location` with provenance.

**Architecture:** An overlay on the existing graph. `UserProfile` links to existing `Entity` nodes via `HAS_ALIAS` rather than replacing them, so a wrong identity merge is one edge to unlink. Attributes are separate `ProfileAttribute` nodes carrying source/confidence/evidence, resolved to a current value by a single precedence function. Everything is additive: `user_id` is optional on ingest, and when absent behaviour is byte-identical to today.

**Tech Stack:** Python 3.11+, FastAPI, Neo4j (Bolt), spaCy `en_core_web_sm`, Celery, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-user-profile-node-design.md`

## Global Constraints

- `user_id` is **optional** on `MemoryIngestRequest`. When absent, no profile work occurs and existing behaviour is unchanged. The existing 74 tests must stay green throughout.
- Entity names are stored normalized via `normalize_entity_name` from `app/services/neo4j_store.py`. All alias comparisons use normalized form.
- Attribute keys are exactly `{"name", "gender", "title", "role", "location"}`. Sources are exactly `{"explicit", "inferred"}`.
- Gender is inferred **only** from pronoun declarations and explicit self-identification — never from a name or honorific.
- Identity ambiguity fails loudly, following the existing `EpisodeCollisionError` precedent. Never silently merge two people.
- Stable synthetic ids use `hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]`, matching `record_decisions_and_preferences`.
- Profile extraction reads **only** messages with `role == "user"`.
- Run tests with `.venv/bin/python -m pytest`. The stack must be up: `docker compose up -d redis postgres neo4j`.

## File Structure

| File | Responsibility |
|------|----------------|
| `app/services/profile_extraction.py` (new) | Pure functions: parse messages → candidate names, aliases, attributes. No I/O. |
| `app/services/profile_store.py` (new) | All Neo4j reads/writes for `UserProfile`, `ProfileAttribute`, `HAS_ALIAS`, and profile-scoped edges. |
| `app/db/neo4j.py` (modify) | Add uniqueness constraints. |
| `app/api/models.py` (modify) | `user_id` on ingest; profile request/response models. |
| `app/api/routes.py` (modify) | `GET`/`PATCH` profile, `POST` alias. |
| `app/workers/tasks.py` (modify) | Thread `user_id` through; call profile resolution. |
| `app/services/retrieval.py` (modify) | Resolve graph facts through the profile. |
| `tests/test_profile_extraction.py` (new) | Unit tests for the pure functions. |
| `tests/test_profile_store.py` (new) | Integration tests against live Neo4j. |
| `tests/agentic/scenarios.py` (modify) | Profile scenarios. |
| `tests/agentic/test_memory_pipeline.py` (modify) | Identity-collapse and cross-session tests. |

Extraction is split from storage because the rules are the part most likely to need tuning, and pure functions are cheap to test exhaustively without a database.

---

### Task 1: Graph constraints and profile upsert

**Files:**
- Modify: `app/db/neo4j.py:28-42` (the `stmts` list in `ensure_constraints`)
- Create: `app/services/profile_store.py`
- Test: `tests/test_profile_store.py`

**Interfaces:**
- Consumes: `create_driver_from_settings`, `ensure_constraints` from `app/db/neo4j.py`
- Produces: `ProfileStore(driver)`, `ProfileStore.upsert_profile(user_id: str, display_name: str | None = None) -> None`, `ProfileStore.get_profile(user_id: str) -> ProfileRow | None`, dataclass `ProfileRow(user_id: str, display_name: str | None)`, exception `ProfileAliasConflictError`

- [ ] **Step 1: Write the failing test**

```python
"""L3 UserProfile: identity node, attributes, aliases, profile-scoped edges."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_store import ProfileRow, ProfileStore

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def user_id(driver):
    """Unique profile id, removed with everything it owns afterwards."""
    uid = f"u-{uuid.uuid4().hex[:12]}"
    yield uid
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            DETACH DELETE p, a
            """,
            uid=uid,
        )


@pytest.fixture
def store(driver):
    return ProfileStore(driver)


def test_upsert_profile_is_idempotent(store, user_id):
    store.upsert_profile(user_id, display_name="Dana Whitfield")
    store.upsert_profile(user_id, display_name="Dana Whitfield")

    row = store.get_profile(user_id)
    assert row == ProfileRow(user_id=user_id, display_name="Dana Whitfield")


def test_get_profile_returns_none_when_absent(store):
    assert store.get_profile("does-not-exist") is None


def test_upsert_profile_preserves_display_name_when_omitted(store, user_id):
    store.upsert_profile(user_id, display_name="Dana Whitfield")
    store.upsert_profile(user_id)

    row = store.get_profile(user_id)
    assert row is not None
    assert row.display_name == "Dana Whitfield"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.profile_store'`

- [ ] **Step 3: Add the constraints**

In `app/db/neo4j.py`, append two entries to the `stmts` list inside `ensure_constraints`:

```python
        "CREATE CONSTRAINT user_profile_id_unique IF NOT EXISTS FOR (p:UserProfile) REQUIRE p.user_id IS UNIQUE",
        "CREATE CONSTRAINT profile_attribute_id_unique IF NOT EXISTS FOR (a:ProfileAttribute) REQUIRE a.id IS UNIQUE",
```

Also update that function's docstring first line to:

```python
    """Create uniqueness constraints for Session, Episode, Entity, and UserProfile if missing.
```

- [ ] **Step 4: Write minimal implementation**

Create `app/services/profile_store.py`:

```python
"""L3 UserProfile: canonical identity, attributes, aliases, profile-scoped edges.

The profile is an overlay. It links to existing `Entity` nodes rather than
replacing them, so an identity decision made from unreliable extraction stays
cheap to reverse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver


class ProfileAliasConflictError(RuntimeError):
    """Raised when one Entity is claimed as an alias by two different profiles.

    Follows the `EpisodeCollisionError` precedent: identity ambiguity fails
    loudly rather than silently merging two people.
    """


@dataclass(frozen=True)
class ProfileRow:
    """Identity fields of a UserProfile node."""

    user_id: str
    display_name: str | None = None


def _stable_suffix(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProfileStore:
    """Graph operations for the canonical user identity."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def upsert_profile(self, user_id: str, display_name: str | None = None) -> None:
        """MERGE the profile. A None `display_name` leaves any existing value intact."""
        q = """
        MERGE (p:UserProfile {user_id: $user_id})
        ON CREATE SET p.created_at = $now
        SET p.updated_at = $now,
            p.display_name = CASE
                WHEN $display_name IS NULL THEN p.display_name
                ELSE $display_name
            END
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, display_name=display_name, now=_now())

    def get_profile(self, user_id: str) -> ProfileRow | None:
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        RETURN p.user_id AS user_id, p.display_name AS display_name
        """
        with self._driver.session() as session:
            record = session.run(q, user_id=user_id).single()
        if record is None:
            return None
        return ProfileRow(
            user_id=record["user_id"],
            display_name=record["display_name"],
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Verify nothing regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 77 passed

- [ ] **Step 7: Commit**

```bash
git add app/db/neo4j.py app/services/profile_store.py tests/test_profile_store.py
git commit -m "feat: add UserProfile node with uniqueness constraints"
```

---

### Task 2: Profile attributes with precedence resolution

**Files:**
- Modify: `app/services/profile_store.py`
- Test: `tests/test_profile_store.py`

**Interfaces:**
- Consumes: `ProfileStore`, `_stable_suffix`, `_now` from Task 1
- Produces: dataclass `AttributeRow(key: str, value: str, source: str, confidence: float, observed_at: str, evidence: str | None)`; `ProfileStore.set_attribute(user_id: str, key: str, value: str, *, source: str, confidence: float, evidence: str | None = None) -> None`; `ProfileStore.get_attributes(user_id: str) -> list[AttributeRow]`; `resolve_attributes(rows: list[AttributeRow]) -> dict[str, AttributeRow]`; `ATTRIBUTE_KEYS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_store.py`:

```python
def test_explicit_attribute_beats_inferred_regardless_of_recency(store, user_id):
    from app.services.profile_store import resolve_attributes

    store.upsert_profile(user_id)
    store.set_attribute(user_id, "title", "Staff Engineer", source="explicit", confidence=1.0)
    store.set_attribute(user_id, "title", "Intern", source="inferred", confidence=0.9)

    current = resolve_attributes(store.get_attributes(user_id))
    assert current["title"].value == "Staff Engineer"
    assert current["title"].source == "explicit"


def test_more_recent_inferred_value_supersedes_older(store, user_id):
    from app.services.profile_store import resolve_attributes

    store.upsert_profile(user_id)
    store.set_attribute(user_id, "location", "Boston", source="inferred", confidence=0.7)
    store.set_attribute(user_id, "location", "Seattle", source="inferred", confidence=0.7)

    current = resolve_attributes(store.get_attributes(user_id))
    assert current["location"].value == "Seattle"
    # History is retained, not overwritten.
    assert {r.value for r in store.get_attributes(user_id) if r.key == "location"} == {
        "Boston",
        "Seattle",
    }


def test_set_attribute_rejects_unknown_key(store, user_id):
    store.upsert_profile(user_id)
    with pytest.raises(ValueError, match="unknown attribute key"):
        store.set_attribute(user_id, "favourite_colour", "blue", source="explicit", confidence=1.0)


def test_set_attribute_rejects_unknown_source(store, user_id):
    store.upsert_profile(user_id)
    with pytest.raises(ValueError, match="unknown source"):
        store.set_attribute(user_id, "title", "Staff Engineer", source="guessed", confidence=1.0)


def test_reasserting_same_value_is_idempotent(store, user_id):
    store.upsert_profile(user_id)
    store.set_attribute(user_id, "role", "tech lead", source="inferred", confidence=0.6)
    store.set_attribute(user_id, "role", "tech lead", source="inferred", confidence=0.6)

    rows = [r for r in store.get_attributes(user_id) if r.key == "role"]
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v -k attribute`
Expected: FAIL with `ImportError: cannot import name 'resolve_attributes'`

- [ ] **Step 3: Write the implementation**

Add to `app/services/profile_store.py`, after `ProfileRow`:

```python
#: The attribute keys this profile schema supports.
ATTRIBUTE_KEYS = frozenset({"name", "gender", "title", "role", "location"})

#: Where a value came from. Explicit statements always win over inference.
ATTRIBUTE_SOURCES = frozenset({"explicit", "inferred"})


@dataclass(frozen=True)
class AttributeRow:
    """One observed value for a profile attribute, with provenance."""

    key: str
    value: str
    source: str
    confidence: float
    observed_at: str
    evidence: str | None = None


def resolve_attributes(rows: list[AttributeRow]) -> dict[str, AttributeRow]:
    """Reduce observation history to the current value per key.

    Precedence: explicit beats inferred, then most recent, then most confident.
    Keeping this a pure function over rows means there is exactly one place that
    decides what a profile currently says.
    """
    current: dict[str, AttributeRow] = {}
    for row in rows:
        incumbent = current.get(row.key)
        if incumbent is None or _rank(row) > _rank(incumbent):
            current[row.key] = row
    return current


def _rank(row: AttributeRow) -> tuple[int, str, float]:
    return (1 if row.source == "explicit" else 0, row.observed_at, row.confidence)
```

Then add these methods to `ProfileStore`:

```python
    def set_attribute(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        source: str,
        confidence: float,
        evidence: str | None = None,
    ) -> None:
        """Record an observed attribute value. Re-asserting a value is idempotent."""
        if key not in ATTRIBUTE_KEYS:
            raise ValueError(f"unknown attribute key {key!r}; expected one of {sorted(ATTRIBUTE_KEYS)}")
        if source not in ATTRIBUTE_SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {sorted(ATTRIBUTE_SOURCES)}")
        cleaned = value.strip()
        if not cleaned:
            return

        # Id keys on (profile, key, value) so the same assertion collapses while a
        # genuinely new value becomes a new node and history is preserved.
        attr_id = f"{user_id}:{key}:{_stable_suffix(cleaned)}"
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (a:ProfileAttribute {id: $attr_id})
        ON CREATE SET a.observed_at = $now
        SET a.key = $key,
            a.value = $value,
            a.source = $source,
            a.confidence = $confidence,
            a.evidence = $evidence
        MERGE (p)-[:HAS_ATTRIBUTE]->(a)
        """
        with self._driver.session() as session:
            session.run(
                q,
                user_id=user_id,
                attr_id=attr_id,
                key=key,
                value=cleaned,
                source=source,
                confidence=float(confidence),
                evidence=evidence,
                now=_now(),
            )

    def get_attributes(self, user_id: str) -> list[AttributeRow]:
        """All observed attribute values for a profile, oldest first."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
        RETURN a.key AS key, a.value AS value, a.source AS source,
               a.confidence AS confidence, a.observed_at AS observed_at,
               a.evidence AS evidence
        ORDER BY a.observed_at
        """
        with self._driver.session() as session:
            return [
                AttributeRow(
                    key=r["key"],
                    value=r["value"],
                    source=r["source"],
                    confidence=float(r["confidence"]),
                    observed_at=r["observed_at"],
                    evidence=r["evidence"],
                )
                for r in session.run(q, user_id=user_id)
            ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v`
Expected: PASS (8 passed)

> If `test_more_recent_inferred_value_supersedes_older` is flaky, two writes landed in the same ISO timestamp. `_now()` has microsecond resolution so this should not occur; if it does, the precedence rule is still correct and the fix is in the test, not `_rank`.

- [ ] **Step 5: Commit**

```bash
git add app/services/profile_store.py tests/test_profile_store.py
git commit -m "feat: profile attributes with provenance and precedence resolution"
```

---

### Task 3: Self-reference extraction (Rule 1)

**Files:**
- Create: `app/services/profile_extraction.py`
- Test: `tests/test_profile_extraction.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure functions, spaCy `Doc` passed in by caller)
- Produces: `user_messages(messages: list[dict]) -> list[str]`; `extract_self_reference_names(text: str, doc: Any) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
"""Pure extraction rules for the user profile. No database, no network."""

from __future__ import annotations

import pytest
import spacy

from app.config import settings
from app.services.profile_extraction import (
    extract_self_reference_names,
    user_messages,
)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load(settings.spacy_model)


def test_user_messages_keeps_only_user_role():
    messages = [
        {"role": "user", "content": "I'm Dana Whitfield."},
        {"role": "assistant", "content": "Hello Dana Whitfield."},
        {"role": "system", "content": "You are helpful."},
    ]
    assert user_messages(messages) == ["I'm Dana Whitfield."]


@pytest.mark.parametrize(
    "text",
    [
        "I'm Dana Whitfield and I work on robots.",
        "I am Dana Whitfield.",
        "My name is Dana Whitfield.",
        "This is Dana Whitfield.",
        "Call me Dana Whitfield.",
    ],
)
def test_self_reference_matches_introduction_forms(text, nlp):
    assert extract_self_reference_names(text, nlp(text)) == ["Dana Whitfield"]


def test_self_reference_ignores_non_person_spans(nlp):
    """'I'm exhausted' must not alias the profile to 'exhausted'."""
    text = "I'm exhausted and I'm behind on the report."
    assert extract_self_reference_names(text, nlp(text)) == []


def test_self_reference_ignores_third_party_mentions(nlp):
    text = "Priya Raman from Lumenwave is consulting with us."
    assert extract_self_reference_names(text, nlp(text)) == []


def test_self_reference_deduplicates(nlp):
    text = "I'm Dana Whitfield. My name is Dana Whitfield."
    assert extract_self_reference_names(text, nlp(text)) == ["Dana Whitfield"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.profile_extraction'`

- [ ] **Step 3: Write the implementation**

Create `app/services/profile_extraction.py`:

```python
"""Pure extraction rules for the canonical user profile.

Every function here takes text (and optionally a spaCy `Doc`) and returns plain
data. No database, no network, no settings lookups — which is what makes the
rules cheap to test exhaustively and safe to tune.

Only `role == "user"` messages are ever considered: the profile describes the
person speaking, not the assistant and not third parties they mention.
"""

from __future__ import annotations

import re
from typing import Any

#: An introduction is one of these phrases immediately before a PERSON span.
#: Anchored to the end so it must directly precede the name.
_INTRO_TAIL = re.compile(
    r"(?i)(?:\bi'?m|\bi am|\bmy name is|\bthis is|\bcall me)\s*$",
)


def user_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Contents of messages authored by the user, in order."""
    return [
        str(m.get("content", ""))
        for m in messages
        if str(m.get("role", "")).lower() == "user"
    ]


def extract_self_reference_names(text: str, doc: Any) -> list[str]:
    """Names the speaker gives for themselves, in order, deduped.

    A candidate must be a spaCy PERSON span *and* be directly preceded by an
    introduction phrase. Requiring both is what stops "I'm exhausted" from
    aliasing the profile to "exhausted", and stops a third-party mention like
    "Priya Raman is consulting" from being read as self-reference.
    """
    found: list[str] = []
    seen: set[str] = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        preceding = text[: ent.start_char]
        if not _INTRO_TAIL.search(preceding):
            continue
        name = ent.text.strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        found.append(name)
    return found
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_extraction.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/profile_extraction.py tests/test_profile_extraction.py
git commit -m "feat: self-reference name extraction for user profile"
```

---

### Task 4: Attribute extraction (title, role, location, gender)

**Files:**
- Modify: `app/services/profile_extraction.py`
- Test: `tests/test_profile_extraction.py`

**Interfaces:**
- Consumes: `_INTRO_TAIL`, `user_messages` from Task 3
- Produces: dataclass `ExtractedAttribute(key: str, value: str, confidence: float, evidence: str)`; `extract_attributes(text: str, doc: Any) -> list[ExtractedAttribute]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_extraction.py`:

```python
def _by_key(items):
    return {item.key: item.value for item in items}


def test_extracts_title(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm a staff engineer at Northwind Robotics."
    assert _by_key(extract_attributes(text, nlp(text)))["title"] == "staff engineer"


def test_extracts_location(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm based in Seattle these days."
    assert _by_key(extract_attributes(text, nlp(text)))["location"] == "Seattle"


def test_rejects_non_place_as_location(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm in trouble with the deadline."
    assert "location" not in _by_key(extract_attributes(text, nlp(text)))


def test_rejects_person_as_title(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm a Dana Whitfield fan."
    assert "title" not in _by_key(extract_attributes(text, nlp(text)))


def test_extracts_gender_from_pronoun_declaration(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "My pronouns are she/her."
    assert _by_key(extract_attributes(text, nlp(text)))["gender"] == "she/her"


def test_does_not_infer_gender_from_name_alone(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm Dana Whitfield."
    assert "gender" not in _by_key(extract_attributes(text, nlp(text)))


def test_title_wins_when_sentence_matches_title_and_role(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm a tech lead and my role is tech lead."
    keys = [item.key for item in extract_attributes(text, nlp(text))]
    assert "title" in keys
    assert "role" not in keys


def test_every_extraction_carries_evidence(nlp):
    from app.services.profile_extraction import extract_attributes

    text = "I'm based in Seattle these days."
    for item in extract_attributes(text, nlp(text)):
        assert item.evidence.strip()
        assert 0.0 < item.confidence <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_extraction.py -v -k "title or location or gender or role or evidence"`
Expected: FAIL with `ImportError: cannot import name 'extract_attributes'`

- [ ] **Step 3: Write the implementation**

Append to `app/services/profile_extraction.py`:

```python
from dataclasses import dataclass

#: spaCy labels that confirm a span really is a place.
_PLACE_LABELS = frozenset({"GPE", "LOC", "FAC"})
#: spaCy labels that disqualify a span from being a job title.
_NOT_TITLE_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"})

_TITLE_PATTERNS = [
    re.compile(r"(?i)\b(?:i'?m|i am)\s+an?\s+([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|\s+at\s|\s+for\s|$)"),
    re.compile(r"(?i)\b(?:i work as|my title is)\s+(?:an?\s+)?([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|$)"),
]

_ROLE_PATTERNS = [
    re.compile(r"(?i)\bmy role is\s+(?:an?\s+)?([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|$)"),
    re.compile(r"(?i)\bi'?m the\s+([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|\s+on\s|\s+for\s|$)"),
]

_LOCATION_PATTERNS = [
    re.compile(r"(?i)\b(?:i'?m based in|i live in|i'?m located in|i'?m in)\s+([^.,;!?\n]{2,60})"),
]

#: Gender is read only from what the speaker states. Never from a name, never
#: from an honorific. The stated string is preserved verbatim rather than being
#: coerced into an enum.
_GENDER_PATTERNS = [
    re.compile(r"(?i)\bmy pronouns are\s+([a-z]+(?:\s*/\s*[a-z]+)+)"),
    re.compile(r"(?i)\bi use\s+([a-z]+\s*/\s*[a-z]+)\s*(?:pronouns)?"),
    re.compile(r"(?i)\bi'?m a\s+(woman|man|nonbinary person|non-binary person)\b"),
]

_CONFIDENCE = {"name": 0.9, "gender": 0.8, "location": 0.7, "title": 0.6, "role": 0.6}


@dataclass(frozen=True)
class ExtractedAttribute:
    """One attribute value inferred from a sentence, with the sentence kept."""

    key: str
    value: str
    confidence: float
    evidence: str


def _spans_overlapping(doc: Any, start: int, end: int) -> list[Any]:
    return [e for e in doc.ents if e.start_char < end and e.end_char > start]


def _first_match(patterns: list[re.Pattern[str]], text: str):
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def extract_attributes(text: str, doc: Any) -> list[ExtractedAttribute]:
    """Infer profile attributes from one user message.

    `title` and `location` require agreement from spaCy's labels, which is what
    keeps "I'm a bit lost" from becoming a job title and "I'm in trouble" from
    becoming a place. `role` has no such check available, so it carries low
    confidence and stays overridable.
    """
    out: list[ExtractedAttribute] = []

    title_match = _first_match(_TITLE_PATTERNS, text)
    if title_match:
        value = title_match.group(1).strip()
        overlapping = _spans_overlapping(doc, title_match.start(1), title_match.end(1))
        if value and not any(e.label_ in _NOT_TITLE_LABELS for e in overlapping):
            out.append(
                ExtractedAttribute("title", value, _CONFIDENCE["title"], text.strip())
            )

    # A sentence that reads as both a title and a role yields only the title, so
    # one utterance never writes two competing attributes.
    if not any(item.key == "title" for item in out):
        role_match = _first_match(_ROLE_PATTERNS, text)
        if role_match:
            value = role_match.group(1).strip()
            if value:
                out.append(
                    ExtractedAttribute("role", value, _CONFIDENCE["role"], text.strip())
                )

    location_match = _first_match(_LOCATION_PATTERNS, text)
    if location_match:
        value = location_match.group(1).strip()
        overlapping = _spans_overlapping(doc, location_match.start(1), location_match.end(1))
        place = next((e for e in overlapping if e.label_ in _PLACE_LABELS), None)
        if place is not None:
            out.append(
                ExtractedAttribute(
                    "location", place.text.strip(), _CONFIDENCE["location"], text.strip()
                )
            )

    gender_match = _first_match(_GENDER_PATTERNS, text)
    if gender_match:
        value = re.sub(r"\s*/\s*", "/", gender_match.group(1).strip())
        if value:
            out.append(
                ExtractedAttribute("gender", value, _CONFIDENCE["gender"], text.strip())
            )

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_extraction.py -v`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/profile_extraction.py tests/test_profile_extraction.py
git commit -m "feat: infer title, role, location, and gender from stated evidence"
```

---

### Task 5: Aliases with subset rule and conflict detection

**Files:**
- Modify: `app/services/profile_store.py`, `app/services/profile_extraction.py`
- Test: `tests/test_profile_store.py`, `tests/test_profile_extraction.py`

**Interfaces:**
- Consumes: `ProfileStore`, `ProfileAliasConflictError` (Task 1); `normalize_entity_name` from `app/services/neo4j_store.py`
- Produces: `ProfileStore.link_alias(user_id: str, entity_name: str, *, source: str, confidence: float) -> str`; `ProfileStore.get_aliases(user_id: str) -> list[str]`; `ProfileStore.unlink_alias(user_id: str, entity_name: str) -> None`; `subset_alias_candidates(confirmed: list[str], candidates: list[str]) -> list[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_extraction.py`:

```python
def test_subset_rule_matches_token_subset():
    from app.services.profile_extraction import subset_alias_candidates

    assert subset_alias_candidates(["dana whitfield"], ["dana"]) == ["dana"]


def test_subset_rule_rejects_unrelated_name():
    from app.services.profile_extraction import subset_alias_candidates

    assert subset_alias_candidates(["dana whitfield"], ["priya raman"]) == []


def test_subset_rule_rejects_partial_token_overlap():
    """'dan' is not a token of 'dana whitfield'; substring matching is wrong here."""
    from app.services.profile_extraction import subset_alias_candidates

    assert subset_alias_candidates(["dana whitfield"], ["dan"]) == []


def test_subset_rule_ignores_already_confirmed():
    from app.services.profile_extraction import subset_alias_candidates

    assert subset_alias_candidates(["dana whitfield"], ["dana whitfield"]) == []
```

Append to `tests/test_profile_store.py`:

```python
@pytest.fixture
def second_user(driver):
    uid = f"u2-{uuid.uuid4().hex[:12]}"
    yield uid
    with driver.session() as s:
        s.run("MATCH (p:UserProfile {user_id: $uid}) DETACH DELETE p", uid=uid)


def test_link_alias_normalizes_and_roundtrips(store, user_id):
    store.upsert_profile(user_id)
    assert store.link_alias(user_id, "Dana Whitfield", source="explicit", confidence=1.0) == "dana whitfield"
    assert store.get_aliases(user_id) == ["dana whitfield"]


def test_linking_same_alias_twice_is_idempotent(store, user_id):
    store.upsert_profile(user_id)
    store.link_alias(user_id, "Dana Whitfield", source="explicit", confidence=1.0)
    store.link_alias(user_id, "dana whitfield", source="inferred", confidence=0.9)
    assert store.get_aliases(user_id) == ["dana whitfield"]


def test_alias_claimed_by_two_profiles_raises(store, user_id, second_user):
    from app.services.profile_store import ProfileAliasConflictError

    store.upsert_profile(user_id)
    store.upsert_profile(second_user)
    store.link_alias(user_id, "Dana Whitfield", source="explicit", confidence=1.0)

    with pytest.raises(ProfileAliasConflictError, match="already an alias"):
        store.link_alias(second_user, "Dana Whitfield", source="inferred", confidence=0.9)


def test_unlink_alias_removes_only_that_edge(store, user_id):
    store.upsert_profile(user_id)
    store.link_alias(user_id, "Dana Whitfield", source="explicit", confidence=1.0)
    store.link_alias(user_id, "Dana", source="inferred", confidence=0.8)

    store.unlink_alias(user_id, "Dana")
    assert store.get_aliases(user_id) == ["dana whitfield"]
```

Update the `user_id` fixture teardown in `tests/test_profile_store.py` to also drop alias entities:

```python
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py tests/test_profile_extraction.py -v -k "alias or subset"`
Expected: FAIL with `AttributeError: 'ProfileStore' object has no attribute 'link_alias'`

- [ ] **Step 3: Implement the subset rule**

Append to `app/services/profile_extraction.py`:

```python
def subset_alias_candidates(
    confirmed: list[str],
    candidates: list[str],
) -> list[str]:
    """Candidates whose tokens are a strict subset of a confirmed alias.

    Token-subset rather than substring: "dan" must not match "dana whitfield".
    Deliberately narrow — it will not catch nicknames or initials, and that is
    preferred over merging two people who share a first name.

    Both arguments must already be normalized via `normalize_entity_name`.
    """
    confirmed_tokens = [set(name.split()) for name in confirmed if name]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        tokens = set(candidate.split())
        if not tokens:
            continue
        if any(tokens < whole for whole in confirmed_tokens):
            seen.add(candidate)
            out.append(candidate)
    return out
```

- [ ] **Step 4: Implement the alias store methods**

Add to `ProfileStore` in `app/services/profile_store.py`, and add the import at the top of the file:

```python
from app.services.neo4j_store import normalize_entity_name
```

```python
    def link_alias(
        self,
        user_id: str,
        entity_name: str,
        *,
        source: str,
        confidence: float,
    ) -> str:
        """Alias an Entity to this profile. Returns the normalized name.

        Raises `ProfileAliasConflictError` when the entity is already aliased to
        a different profile — that means identity is ambiguous, and guessing is
        how two people get silently merged.
        """
        if source not in ATTRIBUTE_SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {sorted(ATTRIBUTE_SOURCES)}")
        norm = normalize_entity_name(entity_name)
        if not norm:
            raise ValueError(f"entity name {entity_name!r} normalizes to empty")

        q_owner = """
        MATCH (other:UserProfile)-[:HAS_ALIAS]->(:Entity {name: $name})
        WHERE other.user_id <> $user_id
        RETURN other.user_id AS owner
        LIMIT 1
        """
        q_link = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.display_name = $display_name
        MERGE (p)-[r:HAS_ALIAS]->(e)
        ON CREATE SET r.source = $source, r.confidence = $confidence, r.linked_at = $now
        """
        with self._driver.session() as session:
            conflict = session.run(q_owner, name=norm, user_id=user_id).single()
            if conflict is not None:
                raise ProfileAliasConflictError(
                    f"Entity {norm!r} is already an alias of profile "
                    f"{conflict['owner']!r}; refusing to also alias it to {user_id!r}."
                )
            session.run(
                q_link,
                user_id=user_id,
                name=norm,
                display_name=entity_name.strip(),
                source=source,
                confidence=float(confidence),
                now=_now(),
            )
        return norm

    def get_aliases(self, user_id: str) -> list[str]:
        """Normalized entity names aliased to this profile."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:HAS_ALIAS]->(e:Entity)
        RETURN e.name AS name ORDER BY name
        """
        with self._driver.session() as session:
            return [r["name"] for r in session.run(q, user_id=user_id)]

    def unlink_alias(self, user_id: str, entity_name: str) -> None:
        """Remove one alias edge. The Entity node itself is left intact."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[r:HAS_ALIAS]->(:Entity {name: $name})
        DELETE r
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, name=normalize_entity_name(entity_name))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py tests/test_profile_extraction.py -v`
Expected: PASS (33 passed)

- [ ] **Step 6: Commit**

```bash
git add app/services/profile_store.py app/services/profile_extraction.py tests/
git commit -m "feat: profile aliases with subset rule and conflict detection"
```

---

### Task 6: Profile-scoped session, decision, and preference edges

**Files:**
- Modify: `app/services/profile_store.py`
- Test: `tests/test_profile_store.py`

**Interfaces:**
- Consumes: `ProfileStore` (Tasks 1–5)
- Produces: `ProfileStore.link_session(user_id: str, session_id: str) -> None`; `ProfileStore.promote_episode_facts(user_id: str, episode_id: int) -> None`; `ProfileStore.get_profile_decisions(user_id: str) -> list[str]`; `ProfileStore.get_profile_preferences(user_id: str) -> list[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_store.py`:

```python
def test_promoted_decisions_are_visible_across_sessions(store, driver, user_id):
    """The payoff: a decision made in one session is reachable from the profile."""
    from app.services.neo4j_store import Neo4jStore

    graph = Neo4jStore(driver)
    session_a = f"{user_id}-sess-a"
    episode_id = -910001  # negative ids cannot collide with real Postgres ids
    try:
        graph.upsert_session(session_a)
        graph.upsert_episode(session_a, episode_id, "Chose Rust for motion control.")
        graph.record_decisions_and_preferences(
            episode_id,
            decisions=["use Rust for motion control"],
            preferences=["async standups"],
        )

        store.upsert_profile(user_id)
        store.link_session(user_id, session_a)
        store.promote_episode_facts(user_id, episode_id)

        assert store.get_profile_decisions(user_id) == ["use Rust for motion control"]
        assert store.get_profile_preferences(user_id) == ["async standups"]
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (e:Episode {id: $eid})
                OPTIONAL MATCH (e)-[:DECIDED|PREFERS]->(dp)
                DETACH DELETE e, dp
                """,
                eid=episode_id,
            )
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_a)


def test_promote_is_idempotent(store, driver, user_id):
    from app.services.neo4j_store import Neo4jStore

    graph = Neo4jStore(driver)
    session_a = f"{user_id}-sess-b"
    episode_id = -910002
    try:
        graph.upsert_session(session_a)
        graph.upsert_episode(session_a, episode_id, "Chose Rust.")
        graph.record_decisions_and_preferences(episode_id, ["use Rust"], [])

        store.upsert_profile(user_id)
        store.promote_episode_facts(user_id, episode_id)
        store.promote_episode_facts(user_id, episode_id)

        assert store.get_profile_decisions(user_id) == ["use Rust"]
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (e:Episode {id: $eid})
                OPTIONAL MATCH (e)-[:DECIDED|PREFERS]->(dp)
                DETACH DELETE e, dp
                """,
                eid=episode_id,
            )
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v -k "promote or across_sessions"`
Expected: FAIL with `AttributeError: 'ProfileStore' object has no attribute 'link_session'`

- [ ] **Step 3: Write the implementation**

Add to `ProfileStore`:

```python
    def link_session(self, user_id: str, session_id: str) -> None:
        """Record that this profile participated in a session."""
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MERGE (s:Session {id: $session_id})
        MERGE (p)-[:PARTICIPATED_IN]->(s)
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, session_id=session_id)

    def promote_episode_facts(self, user_id: str, episode_id: int) -> None:
        """Attach an episode's decisions and preferences to the profile.

        The `(:Episode)-[:DECIDED]->` edges stay as provenance; these additional
        profile edges make "everything this person has ever decided" a one-hop
        query spanning every session.
        """
        q = """
        MATCH (p:UserProfile {user_id: $user_id})
        MATCH (ep:Episode {id: $episode_id})
        OPTIONAL MATCH (ep)-[:DECIDED]->(d:Decision)
        OPTIONAL MATCH (ep)-[:PREFERS]->(pr:Preference)
        FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
            MERGE (p)-[:DECIDED]->(d))
        FOREACH (_ IN CASE WHEN pr IS NULL THEN [] ELSE [1] END |
            MERGE (p)-[:PREFERS]->(pr))
        """
        with self._driver.session() as session:
            session.run(q, user_id=user_id, episode_id=episode_id)

    def get_profile_decisions(self, user_id: str) -> list[str]:
        """Decision texts attached directly to the profile, across all sessions."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:DECIDED]->(d:Decision)
        RETURN DISTINCT d.text AS text ORDER BY text
        """
        with self._driver.session() as session:
            return [r["text"] for r in session.run(q, user_id=user_id) if r["text"]]

    def get_profile_preferences(self, user_id: str) -> list[str]:
        """Preference texts attached directly to the profile, across all sessions."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:PREFERS]->(p:Preference)
        RETURN DISTINCT p.text AS text ORDER BY text
        """
        with self._driver.session() as session:
            return [r["text"] for r in session.run(q, user_id=user_id) if r["text"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_store.py -v`
Expected: PASS (35 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/profile_store.py tests/test_profile_store.py
git commit -m "feat: promote decisions and preferences to profile scope"
```

---

### Task 7: Profile resolution orchestrator

**Files:**
- Modify: `app/services/profile_extraction.py`
- Test: `tests/test_profile_resolution.py` (new)

**Interfaces:**
- Consumes: everything from Tasks 1–6
- Produces: `resolve_profile_from_messages(store, *, user_id, session_id, episode_id, messages, nlp) -> dict[str, Any]` returning `{"aliases": list[str], "attributes": list[str]}`

- [ ] **Step 1: Write the failing test**

```python
"""Profile resolution: messages in, graph writes out."""

from __future__ import annotations

import uuid

import pytest
import spacy

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_extraction import resolve_profile_from_messages
from app.services.profile_store import ProfileStore, resolve_attributes

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def nlp():
    return spacy.load(settings.spacy_model)


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def user_id(driver):
    uid = f"ur-{uuid.uuid4().hex[:12]}"
    yield uid
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )


def test_resolution_creates_profile_alias_and_attributes(driver, user_id, nlp):
    store = ProfileStore(driver)
    messages = [
        {"role": "user", "content": "I'm Dana Whitfield and I'm based in Seattle."},
        {"role": "assistant", "content": "Nice to meet you, Dana Whitfield."},
    ]

    result = resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=messages,
        nlp=nlp,
    )

    assert "dana whitfield" in result["aliases"]
    current = resolve_attributes(store.get_attributes(user_id))
    assert current["name"].value == "Dana Whitfield"
    assert current["location"].value == "Seattle"
    assert current["name"].source == "inferred"


def test_short_form_aliases_via_subset_rule(driver, user_id, nlp):
    store = ProfileStore(driver)
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=[{"role": "user", "content": "I'm Dana Whitfield."}],
        nlp=nlp,
    )
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s2",
        episode_id=None,
        messages=[{"role": "user", "content": "Dana shipped the release."}],
        nlp=nlp,
    )

    assert set(store.get_aliases(user_id)) >= {"dana whitfield", "dana"}


def test_assistant_messages_never_create_aliases(driver, user_id, nlp):
    store = ProfileStore(driver)
    resolve_profile_from_messages(
        store,
        user_id=user_id,
        session_id=f"{user_id}-s1",
        episode_id=None,
        messages=[{"role": "assistant", "content": "I'm Claude, your assistant."}],
        nlp=nlp,
    )
    assert store.get_aliases(user_id) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_resolution.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_profile_from_messages'`

- [ ] **Step 3: Write the implementation**

Append to `app/services/profile_extraction.py`:

```python
import logging

logger = logging.getLogger(__name__)


def resolve_profile_from_messages(
    store: Any,
    *,
    user_id: str,
    session_id: str,
    episode_id: int | None,
    messages: list[dict[str, Any]],
    nlp: Any,
) -> dict[str, Any]:
    """Apply every profile rule for one ingest and write the results.

    Order matters: self-reference names are confirmed first, because the subset
    rule can only expand aliases that are already confirmed.

    An alias conflict is logged and skipped rather than raised — one ambiguous
    name should not fail an entire ingest, and the unaliased entity remains in
    the graph for a human to resolve.
    """
    from app.services.neo4j_store import normalize_entity_name
    from app.services.profile_store import ProfileAliasConflictError

    store.upsert_profile(user_id)
    store.link_session(user_id, session_id)

    texts = user_messages(messages)
    linked: list[str] = []
    recorded: list[str] = []

    # Rule 1: self-reference introductions.
    for text in texts:
        doc = nlp(text)
        for name in extract_self_reference_names(text, doc):
            try:
                linked.append(store.link_alias(user_id, name, source="inferred", confidence=0.9))
            except ProfileAliasConflictError as exc:
                logger.warning("skipping ambiguous alias for %s: %s", user_id, exc)
                continue
            store.set_attribute(
                user_id, "name", name, source="inferred",
                confidence=_CONFIDENCE["name"], evidence=text.strip(),
            )
            recorded.append("name")
            store.upsert_profile(user_id, display_name=name)

        for attribute in extract_attributes(text, doc):
            store.set_attribute(
                user_id,
                attribute.key,
                attribute.value,
                source="inferred",
                confidence=attribute.confidence,
                evidence=attribute.evidence,
            )
            recorded.append(attribute.key)

    # Rule 2: short forms of names already confirmed for this profile.
    confirmed = store.get_aliases(user_id)
    mentioned = [
        normalize_entity_name(ent.text)
        for text in texts
        for ent in nlp(text).ents
        if ent.label_ == "PERSON"
    ]
    for candidate in subset_alias_candidates(confirmed, mentioned):
        try:
            linked.append(store.link_alias(user_id, candidate, source="inferred", confidence=0.7))
        except ProfileAliasConflictError as exc:
            logger.warning("skipping ambiguous subset alias for %s: %s", user_id, exc)

    if episode_id is not None:
        store.promote_episode_facts(user_id, episode_id)

    return {"aliases": sorted(set(linked)), "attributes": sorted(set(recorded))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_resolution.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/profile_extraction.py tests/test_profile_resolution.py
git commit -m "feat: profile resolution orchestrator"
```

---

### Task 8: Thread user_id through the API and worker

**Files:**
- Modify: `app/api/models.py:17-24`, `app/api/services.py:52-62`, `app/api/routes.py:59-90`, `app/workers/tasks.py`
- Test: `tests/test_profile_api.py` (new)

**Interfaces:**
- Consumes: `resolve_profile_from_messages` (Task 7), `ProfileStore` (Tasks 1–6)
- Produces: `MemoryIngestRequest.user_id: str | None`; `process_conversation(self, session_id, messages, metadata=None, user_id=None)`

- [ ] **Step 1: Write the failing test**

```python
"""user_id threading: optional field, profile built on ingest."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.main import app
from app.services.profile_store import ProfileStore

pytestmark = pytest.mark.integration

AUTH = {"X-API-Key": "dummy-api-key-123"}


@pytest.fixture(scope="module", autouse=True)
def _eager():
    from app.workers.celery_app import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


def test_ingest_without_user_id_still_works(client):
    """The field is optional; omitting it must not change behaviour."""
    response = client.post(
        "/memory/ingest",
        headers=AUTH,
        json={
            "session_id": f"nouser-{uuid.uuid4().hex[:8]}",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 202


def test_ingest_with_user_id_builds_profile(client, driver):
    user_id = f"api-{uuid.uuid4().hex[:10]}"
    session_id = f"{user_id}-s1"
    try:
        response = client.post(
            "/memory/ingest",
            headers=AUTH,
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {"role": "user", "content": "I'm Dana Whitfield and I'm based in Seattle."},
                    {"role": "assistant", "content": "Good to meet you."},
                ],
            },
        )
        assert response.status_code == 202

        store = ProfileStore(driver)
        assert store.get_profile(user_id) is not None
        assert "dana whitfield" in store.get_aliases(user_id)
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (p:UserProfile {user_id: $uid})
                OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
                OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
                DETACH DELETE p, a, e
                """,
                uid=user_id,
            )
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_api.py -v`
Expected: FAIL — `test_ingest_with_user_id_builds_profile` fails because `get_profile` returns `None` (the field is silently ignored by Pydantic).

- [ ] **Step 3: Add the model field**

In `app/api/models.py`, in `MemoryIngestRequest`, add after `session_id`:

```python
    #: Optional canonical identity. When absent, no profile work occurs.
    user_id: str | None = Field(default=None, min_length=1)
```

- [ ] **Step 4: Pass it through the enqueue helper**

In `app/api/services.py`, replace `enqueue_conversation_task` with:

```python
def enqueue_conversation_task(
    session_id: str,
    messages: list[dict[str, str]],
    metadata: dict[str, object] | None,
    user_id: str | None = None,
):
    """Queue the Celery background job."""
    return process_conversation.delay(session_id, messages, metadata, user_id)
```

In `app/api/routes.py`, update the call inside `ingest_memory`:

```python
        task = api_services.enqueue_conversation_task(
            payload.session_id,
            messages,
            payload.metadata,
            payload.user_id,
        )
```

- [ ] **Step 5: Wire the worker**

In `app/workers/tasks.py`, change the task signature:

```python
def process_conversation(
    self: Task,
    session_id: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
```

Add this helper above `process_conversation`:

```python
def _write_profile(
    neo_driver: Any,
    *,
    user_id: str,
    session_id: str,
    episode_id: int | None,
    messages: list[dict[str, Any]],
    nlp: Any,
) -> None:
    """Resolve the canonical profile for this ingest.

    Kept separate from `_write_l3` because the profile rules need role-aware
    access to the message list, while `_write_l3` only receives flattened text.
    """
    from app.services.profile_extraction import resolve_profile_from_messages
    from app.services.profile_store import ProfileStore

    resolve_profile_from_messages(
        ProfileStore(neo_driver),
        user_id=user_id,
        session_id=session_id,
        episode_id=episode_id,
        messages=messages,
        nlp=nlp,
    )
```

Then call it in **both** places where `_write_l3` is called. In the dedupe/retry branch, after the existing `_write_l3(...)` call and inside the same `try`:

```python
            if user_id:
                _write_profile(
                    neo_driver,
                    user_id=user_id,
                    session_id=session_id,
                    episode_id=existing_episode_id,
                    messages=messages,
                    nlp=nlp,
                )
```

And in the main path, after the second `_write_l3(...)` call, inside the same `try`:

```python
        if user_id:
            _write_profile(
                neo_driver,
                user_id=user_id,
                session_id=session_id,
                episode_id=episode_id,
                messages=messages,
                nlp=nlp,
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_api.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Verify nothing regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all previously passing tests still pass

- [ ] **Step 8: Commit**

```bash
git add app/api/models.py app/api/services.py app/api/routes.py app/workers/tasks.py tests/test_profile_api.py
git commit -m "feat: thread optional user_id through ingest to profile resolution"
```

---

### Task 9: Profile read and write endpoints

**Files:**
- Modify: `app/api/models.py`, `app/api/routes.py`
- Test: `tests/test_profile_api.py`

**Interfaces:**
- Consumes: `ProfileStore` (Tasks 1–6)
- Produces: `GET /profile/{user_id}`, `PATCH /profile/{user_id}`, `POST /profile/{user_id}/alias`; models `ProfileAttributeValue`, `ProfileResponse`, `ProfileUpdateRequest`, `ProfileAliasRequest`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profile_api.py`:

```python
def test_profile_endpoints_roundtrip(client, driver):
    user_id = f"ep-{uuid.uuid4().hex[:10]}"
    try:
        assert client.get(f"/profile/{user_id}", headers=AUTH).status_code == 404

        patch = client.patch(
            f"/profile/{user_id}",
            headers=AUTH,
            json={"attributes": {"title": "Principal Engineer", "location": "Seattle"}},
        )
        assert patch.status_code == 200

        alias = client.post(
            f"/profile/{user_id}/alias",
            headers=AUTH,
            json={"entity_name": "Dana Whitfield"},
        )
        assert alias.status_code == 200
        assert alias.json()["aliases"] == ["dana whitfield"]

        body = client.get(f"/profile/{user_id}", headers=AUTH).json()
        assert body["user_id"] == user_id
        assert body["attributes"]["title"]["value"] == "Principal Engineer"
        assert body["attributes"]["title"]["source"] == "explicit"
        assert body["aliases"] == ["dana whitfield"]
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (p:UserProfile {user_id: $uid})
                OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
                OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
                DETACH DELETE p, a, e
                """,
                uid=user_id,
            )


def test_explicit_patch_overrides_inference(client, driver):
    user_id = f"ov-{uuid.uuid4().hex[:10]}"
    session_id = f"{user_id}-s1"
    try:
        client.post(
            "/memory/ingest",
            headers=AUTH,
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [{"role": "user", "content": "I'm based in Boston."}],
            },
        )
        client.patch(
            f"/profile/{user_id}",
            headers=AUTH,
            json={"attributes": {"location": "Seattle"}},
        )
        body = client.get(f"/profile/{user_id}", headers=AUTH).json()
        assert body["attributes"]["location"]["value"] == "Seattle"
        assert body["attributes"]["location"]["source"] == "explicit"
    finally:
        with driver.session() as s:
            s.run(
                """
                MATCH (p:UserProfile {user_id: $uid})
                OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
                OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
                DETACH DELETE p, a, e
                """,
                uid=user_id,
            )
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_id)


def test_alias_conflict_returns_409(client, driver):
    first = f"c1-{uuid.uuid4().hex[:10]}"
    second = f"c2-{uuid.uuid4().hex[:10]}"
    try:
        client.patch(f"/profile/{first}", headers=AUTH, json={"attributes": {}})
        client.patch(f"/profile/{second}", headers=AUTH, json={"attributes": {}})
        client.post(f"/profile/{first}/alias", headers=AUTH, json={"entity_name": "Dana Whitfield"})

        clash = client.post(
            f"/profile/{second}/alias", headers=AUTH, json={"entity_name": "Dana Whitfield"}
        )
        assert clash.status_code == 409
    finally:
        with driver.session() as s:
            for uid in (first, second):
                s.run(
                    """
                    MATCH (p:UserProfile {user_id: $uid})
                    OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
                    DETACH DELETE p, e
                    """,
                    uid=uid,
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_api.py -v -k "endpoints or override or conflict"`
Expected: FAIL — `GET /profile/...` returns 404 from the router (route not registered), and `PATCH` returns 405.

- [ ] **Step 3: Add the models**

Append to `app/api/models.py`:

```python
class ProfileAttributeValue(BaseModel):
    """One resolved attribute value with its provenance."""

    value: str
    source: Literal["explicit", "inferred"]
    confidence: float
    observed_at: str
    evidence: str | None = None


class ProfileResponse(BaseModel):
    """Resolved canonical profile."""

    user_id: str
    display_name: str | None = None
    attributes: dict[str, ProfileAttributeValue] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


class ProfileUpdateRequest(BaseModel):
    """Explicitly set profile attributes. Overrides any inferred value."""

    attributes: dict[str, str] = Field(default_factory=dict)
    display_name: str | None = None


class ProfileAliasRequest(BaseModel):
    """Manually register an alias for a profile."""

    entity_name: str = Field(min_length=1)
```

- [ ] **Step 4: Add the routes**

In `app/api/routes.py`, add imports:

```python
from app.api.models import (
    ProfileAliasRequest,
    ProfileAttributeValue,
    ProfileResponse,
    ProfileUpdateRequest,
)
from app.services.profile_store import (
    ATTRIBUTE_KEYS,
    ProfileAliasConflictError,
    ProfileStore,
    resolve_attributes,
)
```

Then append:

```python
def _profile_store() -> ProfileStore:
    return ProfileStore(api_services.get_neo4j_driver())


def _profile_response(store: ProfileStore, user_id: str) -> ProfileResponse:
    row = store.get_profile(user_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile for user_id {user_id!r}",
        )
    current = resolve_attributes(store.get_attributes(user_id))
    return ProfileResponse(
        user_id=row.user_id,
        display_name=row.display_name,
        attributes={
            key: ProfileAttributeValue(
                value=attr.value,
                source=attr.source,
                confidence=attr.confidence,
                observed_at=attr.observed_at,
                evidence=attr.evidence,
            )
            for key, attr in current.items()
        },
        aliases=store.get_aliases(user_id),
        decisions=store.get_profile_decisions(user_id),
        preferences=store.get_profile_preferences(user_id),
    )


@router.get("/profile/{user_id}", response_model=ProfileResponse)
def get_profile(user_id: str, _api_key: str = Depends(require_api_key)) -> ProfileResponse:
    """Return the resolved canonical profile."""
    return _profile_response(_profile_store(), user_id)


@router.patch("/profile/{user_id}", response_model=ProfileResponse)
def update_profile(
    user_id: str,
    payload: ProfileUpdateRequest,
    _api_key: str = Depends(require_api_key),
) -> ProfileResponse:
    """Set attributes explicitly. Explicit values always beat inferred ones."""
    unknown = sorted(set(payload.attributes) - ATTRIBUTE_KEYS)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown attribute keys: {unknown}; expected {sorted(ATTRIBUTE_KEYS)}",
        )

    store = _profile_store()
    store.upsert_profile(user_id, display_name=payload.display_name)
    for key, value in payload.attributes.items():
        store.set_attribute(user_id, key, value, source="explicit", confidence=1.0)
    return _profile_response(store, user_id)


@router.post("/profile/{user_id}/alias", response_model=ProfileResponse)
def add_profile_alias(
    user_id: str,
    payload: ProfileAliasRequest,
    _api_key: str = Depends(require_api_key),
) -> ProfileResponse:
    """Register an alias manually. Conflicts are reported, never resolved by guessing."""
    store = _profile_store()
    store.upsert_profile(user_id)
    try:
        store.link_alias(user_id, payload.entity_name, source="explicit", confidence=1.0)
    except ProfileAliasConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _profile_response(store, user_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_profile_api.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add app/api/models.py app/api/routes.py tests/test_profile_api.py
git commit -m "feat: profile read, update, and alias endpoints"
```

---

### Task 10: Retrieval resolves through the profile

**Files:**
- Modify: `app/services/retrieval.py:75-115` (`_format_graph_facts`), `app/services/retrieval.py:150-252` (`retrieve_context`)
- Test: `tests/test_retrieval_profile.py` (new)

**Interfaces:**
- Consumes: `ProfileStore`, `resolve_attributes` (Tasks 1–6)
- Produces: `retrieve_context(session_id, query, max_tokens=None, user_id=None)`; new source types `profile_identity` and `profile_decision` / `profile_preference`, all tier `L3`

- [ ] **Step 1: Write the failing test**

```python
"""Retrieval resolves identity through the canonical profile."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.profile_store import ProfileStore
from app.services.retrieval import retrieve_context

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


@pytest.fixture
def seeded(driver):
    """A profile with identity and a decision, plus an unrelated empty session."""
    from app.services.neo4j_store import Neo4jStore

    uid = f"rp-{uuid.uuid4().hex[:10]}"
    session_a = f"{uid}-a"
    session_b = f"{uid}-b"
    episode_id = -920001

    graph = Neo4jStore(driver)
    store = ProfileStore(driver)
    graph.upsert_session(session_a)
    graph.upsert_episode(session_a, episode_id, "Chose Rust for motion control.")
    graph.record_decisions_and_preferences(
        episode_id, ["use Rust for motion control"], ["async standups"]
    )
    store.upsert_profile(uid, display_name="Dana Whitfield")
    store.set_attribute(uid, "title", "Staff Engineer", source="explicit", confidence=1.0)
    store.link_alias(uid, "Dana Whitfield", source="explicit", confidence=1.0)
    store.link_session(uid, session_a)
    store.promote_episode_facts(uid, episode_id)

    yield uid, session_b

    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )
        s.run(
            """
            MATCH (ep:Episode {id: $eid})
            OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
            DETACH DELETE ep, dp
            """,
            eid=episode_id,
        )
        for sid in (session_a, session_b):
            s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=sid)


def test_profile_facts_reach_a_different_session(seeded):
    """The payoff: session B knows who the user is and what they decided in A."""
    uid, session_b = seeded

    result = retrieve_context(session_b, "What do you know about me?", 1500, user_id=uid)
    context = result["context"].lower()

    assert "dana whitfield" in context
    assert "staff engineer" in context
    assert "use rust for motion control" in context

    types = {s["type"] for s in result["sources"]}
    assert "profile_identity" in types
    assert {s["tier"] for s in result["sources"] if s["type"] == "profile_identity"} == {"L3"}


def test_retrieval_without_user_id_is_unchanged(seeded):
    """Omitting user_id must not surface profile facts."""
    _uid, session_b = seeded

    result = retrieve_context(session_b, "What do you know about me?", 1500)
    assert "staff engineer" not in result["context"].lower()
    assert all(s["type"] != "profile_identity" for s in result["sources"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_retrieval_profile.py -v`
Expected: FAIL with `TypeError: retrieve_context() got an unexpected keyword argument 'user_id'`

- [ ] **Step 3: Add the profile section builder**

In `app/services/retrieval.py`, add after `_format_graph_facts`:

```python
def _format_profile_facts(user_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Identity, attributes, and profile-scoped decisions/preferences.

    These are not session-scoped, which is what lets a fresh session know who it
    is talking to.
    """
    from app.services.profile_store import ProfileStore, resolve_attributes

    store = ProfileStore(api_services.get_neo4j_driver())
    row = store.get_profile(user_id)
    if row is None:
        return [], []

    lines: list[str] = []
    sources: list[dict[str, Any]] = []

    name = row.display_name or user_id
    lines.append(f"User: {name}")
    sources.append(_source("profile_identity", user_id=user_id, display_name=row.display_name))

    for key, attr in sorted(resolve_attributes(store.get_attributes(user_id)).items()):
        if key == "name":
            continue
        lines.append(f"User {key}: {attr.value}")
        sources.append(
            _source("profile_identity", user_id=user_id, key=key, source=attr.source)
        )

    for decision in store.get_profile_decisions(user_id)[: settings.retrieval_max_graph_facts]:
        lines.append(f"User decision: {decision}")
        sources.append(_source("profile_decision", user_id=user_id, text=decision))

    for preference in store.get_profile_preferences(user_id)[: settings.retrieval_max_graph_facts]:
        lines.append(f"User preference: {preference}")
        sources.append(_source("profile_preference", user_id=user_id, text=preference))

    return lines, sources
```

- [ ] **Step 4: Map the new source types to L3**

In `app/services/retrieval.py`, replace `_source_tier` with:

```python
def _source_tier(source_type: str) -> str:
    if source_type == "recent_message":
        return "L1"
    if source_type == "episode":
        return "L2"
    return "L3"
```

> This already returns `L3` for anything unrecognised, so `profile_identity`,
> `profile_decision`, and `profile_preference` are handled with no change. Confirm
> the function reads exactly as above and move on.

- [ ] **Step 5: Wire it into retrieve_context**

Change the signature:

```python
def retrieve_context(
    session_id: str,
    query: str,
    max_tokens: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
```

Then, immediately before the `context, sources, truncated = _merge_sections(` call, add:

```python
    profile_lines: list[str] = []
    profile_sources: list[dict[str, Any]] = []
    if user_id:
        try:
            profile_lines, profile_sources = _format_profile_facts(user_id)
        except Exception:
            overall_status = "degraded"
            warnings.append("Profile retrieval unavailable; returning partial context")
```

And add the profile section to the `_merge_sections` list, placed after recent
conversation so identity survives truncation ahead of older episodes:

```python
    context, sources, truncated = _merge_sections(
        [
            ("Recent Conversation", recent_lines, recent_sources),
            ("User Profile", profile_lines, profile_sources),
            ("Relevant Past Episodes", episode_lines, episode_sources),
            ("Graph Facts", graph_lines, graph_sources),
        ],
        max_tokens=token_budget,
    )
```

- [ ] **Step 6: Pass user_id from the route**

In `app/api/models.py`, add to `MemoryRetrieveRequest`:

```python
    user_id: str | None = Field(default=None, min_length=1)
```

In `app/api/routes.py`, update the `retrieve_context` call in `retrieve_memory`:

```python
        result = retrieve_context(
            payload.session_id,
            payload.query,
            payload.max_tokens,
            payload.user_id,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_retrieval_profile.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Verify nothing regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all tests green

- [ ] **Step 9: Commit**

```bash
git add app/services/retrieval.py app/api/models.py app/api/routes.py tests/test_retrieval_profile.py
git commit -m "feat: surface canonical profile facts in retrieval"
```

---

### Task 11: Agentic scenarios for identity collapse and cross-session recall

**Files:**
- Modify: `tests/agentic/scenarios.py`, `tests/agentic/conftest.py`, `tests/agentic/test_memory_pipeline.py`, `tests/agentic/graph_probe.py`

**Interfaces:**
- Consumes: everything above
- Produces: `profile_user_id` fixture; `probe.profile_aliases(driver, user_id)`; scenario `IDENTITY`

- [ ] **Step 1: Write the failing test**

Add to `tests/agentic/graph_probe.py`:

```python
def profile_aliases(driver: Any, user_id: str) -> set[str]:
    """Normalized alias names attached to a profile."""
    q = """
    MATCH (:UserProfile {user_id: $uid})-[:HAS_ALIAS]->(e:Entity)
    RETURN e.name AS name
    """
    with driver.session() as s:
        return {r["name"] for r in s.run(q, uid=user_id)}
```

Add to `tests/agentic/conftest.py`:

```python
@pytest.fixture
def profile_user_id(neo4j_driver):
    """A unique profile id, fully removed afterwards."""
    uid = f"agentic-u-{uuid.uuid4().hex[:10]}"
    yield uid
    with neo4j_driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(e:Entity)
            DETACH DELETE p, a, e
            """,
            uid=uid,
        )
```

Also extend the `ingest` fixture in `tests/agentic/conftest.py` to accept a user id:

```python
@pytest.fixture
def ingest(api, auth):
    """Post messages to the real /memory/ingest endpoint."""

    def _ingest(sid: str, messages: list[dict[str, str]], metadata=None, user_id=None):
        body: dict[str, object] = {
            "session_id": sid,
            "messages": messages,
            "metadata": metadata,
        }
        if user_id is not None:
            body["user_id"] = user_id
        response = api.post("/memory/ingest", headers=auth, json=body)
        assert response.status_code == 202, response.text
        return response.json()

    return _ingest
```

Add to `tests/agentic/test_memory_pipeline.py`:

```python
def test_identity_collapses_to_one_profile(
    agent, ingest, session_id, profile_user_id, neo4j_driver
):
    """'Dana Whitfield' then 'Dana' resolve to one identity, not two entities."""
    from tests.agentic.ollama_agent import Turn

    intro = Turn(
        intent="Introduce yourself by full name to your assistant.",
        anchors=["Dana Whitfield"],
        fallback="Hi, I'm Dana Whitfield.",
    )
    followup = Turn(
        intent="Refer to yourself by first name only while reporting progress.",
        anchors=["Dana"],
        fallback="Dana shipped the release today.",
    )

    agent.transcript.clear()
    for turn in (intro, followup):
        ingest(session_id, agent.exchange(turn), None, profile_user_id)

    aliases = probe.profile_aliases(neo4j_driver, profile_user_id)
    assert "dana whitfield" in aliases, f"self-reference alias missing: {aliases}"
    assert "dana" in aliases, f"subset alias missing: {aliases}"


def test_profile_facts_survive_into_a_new_session(
    agent, ingest, retrieve, api, auth, session_id, profile_user_id
):
    """The audit's blocker: a second session recalls the first session's facts."""
    from tests.agentic.ollama_agent import Turn

    agent.transcript.clear()
    ingest(
        session_id,
        agent.exchange(
            Turn(
                intent="Introduce yourself and state the language your team chose.",
                anchors=["Dana Whitfield", "decided to use Rust"],
                fallback="I'm Dana Whitfield and we decided to use Rust for the backend.",
            )
        ),
        None,
        profile_user_id,
    )

    fresh_session = f"{session_id}-second"
    response = api.post(
        "/memory/retrieve",
        headers=auth,
        json={
            "session_id": fresh_session,
            "user_id": profile_user_id,
            "query": "Who am I and what did we choose?",
            "max_tokens": 1500,
        },
    )
    assert response.status_code == 200, response.text
    context = response.json()["context"].lower()
    assert "dana whitfield" in context, f"identity lost across sessions:\n{context}"
    assert "rust" in context, f"decision lost across sessions:\n{context}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/agentic -v -k "identity or new_session"`
Expected: FAIL with `fixture 'profile_user_id' not found` before the fixture is added; after adding it, the assertions fail until Tasks 1–10 are complete.

- [ ] **Step 3: Run the full agentic suite**

Run: `.venv/bin/python -m pytest tests/agentic -q`
Expected: PASS (13 passed)

- [ ] **Step 4: Check stability**

LLM-driven tests must not flake. Run:

```bash
for i in 1 2 3 4 5; do .venv/bin/python -m pytest tests/agentic -q 2>&1 | tail -1; done
```

Expected: 5 clean runs. If any run fails, diagnose before continuing — do not
retry until it passes by chance. Anchors are contiguous strings for exactly this
reason; if the generated text satisfies the anchors but the assertion fails, the
bug is in the implementation, not the model.

- [ ] **Step 5: Run everything**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all tests green

- [ ] **Step 6: Commit**

```bash
git add tests/agentic/
git commit -m "test: agentic coverage for identity collapse and cross-session recall"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Schema (`UserProfile`, `ProfileAttribute`) + constraints | 1, 2 |
| Attributes as nodes; precedence resolution | 2 |
| Attribute extraction table (title/role/location/gender) | 4 |
| Gender from stated evidence only | 4 |
| Alias Rule 1 (self-reference) | 3, 7 |
| Alias Rule 2 (subset) | 5, 7 |
| Explicit alias registration | 9 |
| `ProfileAliasConflictError` | 5, 9 |
| `PARTICIPATED_IN` / profile-scoped `DECIDED` / `PREFERS` | 6 |
| Related entities by traversal, not materialized | 6 (no edges created — by omission, as specified) |
| Optional `user_id`, additive behaviour | 8 |
| API endpoints | 9 |
| Retrieval resolves through profile | 10 |
| Testing: unit + agentic scenarios | 3, 4, 5, 11 |
| Migration: purely additive | No task needed — every label and field is new and optional |

**Type consistency:** `ProfileStore` method names are used identically across Tasks 1–10 (`upsert_profile`, `get_profile`, `set_attribute`, `get_attributes`, `link_alias`, `get_aliases`, `unlink_alias`, `link_session`, `promote_episode_facts`, `get_profile_decisions`, `get_profile_preferences`). `resolve_attributes` returns `dict[str, AttributeRow]` in Task 2 and is consumed as such in Tasks 7, 9, 10. `ATTRIBUTE_SOURCES` is defined in Task 2 and reused for validation in Task 5.

**Known ordering constraint:** Task 5 uses `ATTRIBUTE_SOURCES`, defined in Task 2. Task 7 depends on Tasks 1–6. Tasks must run in order.
