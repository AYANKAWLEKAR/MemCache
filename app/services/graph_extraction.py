"""spaCy NER, co-occurrence edges, and regex-based decision/preference hints for L3."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import spacy.tokens


from app.services.summarization import format_conversation_for_prompt


def conversation_text(messages: list[dict[str, Any]]) -> str:
    """Flatten messages to one string for NER / regex."""
    return format_conversation_for_prompt(messages)


def _gap_between_spans(
    a: "spacy.tokens.Span",
    b: "spacy.tokens.Span",
) -> int:
    """Minimum token gap between two non-overlapping spans; 0 if overlapping."""
    if a.end <= b.start:
        return b.start - a.end
    if b.end <= a.start:
        return a.start - b.end
    return 0


def entity_cooccurrence_pairs(
    doc: "spacy.tokens.Doc",
    *,
    window_tokens: int = 10,
    allowed_labels: "frozenset[str] | None" = None,
) -> list[tuple[str, str]]:
    """Pairs of entity *surface* strings whose spans are within ``window_tokens`` (or overlap).

    Filtered by the same label allowlist as `ner_entity_texts`, so co-occurrence
    edges connect real entities rather than linking them to dates and numbers.
    """
    labels = ENTITY_LABELS if allowed_labels is None else allowed_labels
    ents = [e for e in doc.ents if e.text.strip() and e.label_ in labels]
    pairs: list[tuple[str, str]] = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            g = _gap_between_spans(ents[i], ents[j])
            if g <= window_tokens:
                pairs.append((ents[i].text.strip(), ents[j].text.strip()))
    return pairs


# spaCy labels worth persisting as graph Entities: things a persona is *about*.
# Deliberately excludes DATE, TIME, CARDINAL, ORDINAL, PERCENT, MONEY, QUANTITY —
# those are properties of a statement, not durable entities, and without this
# filter the graph fills with "today" / "later today" / "3" nodes that crowd out
# the people, orgs and products the memory is supposed to be tracking.
ENTITY_LABELS = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "FAC",
        "NORP",
        "LANGUAGE",
        "LAW",
    }
)


def ner_entity_texts(
    doc: "spacy.tokens.Doc",
    *,
    allowed_labels: frozenset[str] = ENTITY_LABELS,
) -> list[str]:
    """Entity texts from NER (order preserved, rough dedupe by normalized text).

    Only labels in `allowed_labels` are kept; see `ENTITY_LABELS` for why.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in allowed_labels:
            continue
        t = ent.text.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# Decision verbs, with the subject optional. The earlier list required an explicit
# subject for most forms ("we chose"), so a bare past-tense report — "chose
# ClickHouse for the database" — produced no Decision node at all.
_DECISION_VERBS = (
    r"(?:decided\s+(?:to|on)|chose(?:\s+to)?|picked|selected|settled\s+on|"
    r"went\s+with|going\s+with|opted\s+for|adopted|standardi[sz]ed\s+on|"
    r"will\s+use|agreed\s+to\s+use|committed\s+to)"
)

_DECISION_PATTERNS = [
    re.compile(rf"(?i)\b(?:we|i|they|the team)?\s*\b{_DECISION_VERBS}\s+([^.;\n]+)"),
    re.compile(r"(?i)(?:we will|we'll)\s+([^.;\n]+)"),
]

_PREFERENCE_PATTERNS = [
    re.compile(
        r"(?i)(?:prefer|prefers|preference for|loves?|likes?|favors?)\s+(?:to\s+)?([^.;\n]+)",
    ),
]


# A conjunction followed by a fresh subject pronoun starts a new clause, so the
# decision/preference span ends there. Without this, "we decided to use Python
# for the backend and I prefer dark mode" captures the preference clause too.
_CLAUSE_BOUNDARY = re.compile(
    r"(?i)\s*\b(?:and|but|though|although|however|while|whereas|so)\s+"
    r"(?:i|we|they|he|she|you|it)\b",
)


def _trim_clause(span: str) -> str:
    """Cut a captured span at the first new-clause boundary."""
    match = _CLAUSE_BOUNDARY.search(span)
    if match:
        span = span[: match.start()]
    return span.strip().rstrip(",;:")


def _dedupe_subsumed(spans: list[str]) -> list[str]:
    """Drop spans wholly contained in a longer span.

    Overlapping patterns capture the same statement at different offsets
    ("we will" vs "will use"), which would otherwise create near-duplicate
    Decision/Preference nodes for one utterance.
    """
    unique = set(spans)
    kept = [
        s
        for s in unique
        if not any(other != s and s.lower() in other.lower() for other in unique)
    ]
    return sorted(kept)


def extract_decisions_preferences_regex(text: str) -> tuple[list[str], list[str]]:
    """Lightweight regex extraction for Decision / Preference nodes."""
    decisions: list[str] = []
    preferences: list[str] = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text):
            s = _trim_clause(m.group(1))
            if len(s) > 2:
                decisions.append(s)
    for pat in _PREFERENCE_PATTERNS:
        for m in pat.finditer(text):
            s = _trim_clause(m.group(1))
            if len(s) > 2:
                preferences.append(s)
    return _dedupe_subsumed(decisions), _dedupe_subsumed(preferences)
