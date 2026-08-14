"""Pure extraction rules for the canonical user profile.

Every function here takes text (and optionally a spaCy `Doc`) and returns plain
data. No database, no network, no settings lookups — which is what makes the
rules cheap to test exhaustively and safe to tune.

Only `role == "user"` messages are ever considered: the profile describes the
person speaking, not the assistant and not third parties they mention.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

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


#: spaCy labels that confirm a span really is a place.
_PLACE_LABELS = frozenset({"GPE", "LOC", "FAC"})
#: spaCy labels that disqualify a span from being a job title.
_NOT_TITLE_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"})

_TITLE_PATTERNS = [
    re.compile(
        r"(?i)\b(?:i'?m|i am)\s+an?\s+([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|\s+at\s|\s+for\s|$)"
    ),
    re.compile(
        r"(?i)\b(?:i work as|my title is)\s+(?:an?\s+)?([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|$)"
    ),
]

_ROLE_PATTERNS = [
    re.compile(r"(?i)\bmy role is\s+(?:an?\s+)?([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|$)"),
    re.compile(r"(?i)\bi'?m the\s+([a-z][a-z\s/-]{2,40}?)(?=[.,;!?\n]|\s+on\s|\s+for\s|$)"),
]

_LOCATION_PATTERNS = [
    re.compile(
        r"(?i)\b(?:i'?m based in|i live in|i'?m located in|i'?m in)\s+([^.,;!?\n]{2,60})"
    ),
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
        overlapping = _spans_overlapping(
            doc, location_match.start(1), location_match.end(1)
        )
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
                linked.append(
                    store.link_alias(user_id, name, source="inferred", confidence=0.9)
                )
            except ProfileAliasConflictError as exc:
                logger.warning("skipping ambiguous alias for %s: %s", user_id, exc)
                continue
            store.set_attribute(
                user_id,
                "name",
                name,
                source="inferred",
                confidence=_CONFIDENCE["name"],
                evidence=text.strip(),
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
            linked.append(
                store.link_alias(user_id, candidate, source="inferred", confidence=0.7)
            )
        except ProfileAliasConflictError as exc:
            logger.warning("skipping ambiguous subset alias for %s: %s", user_id, exc)

    if episode_id is not None:
        store.promote_episode_facts(user_id, episode_id)

    return {"aliases": sorted(set(linked)), "attributes": sorted(set(recorded))}
