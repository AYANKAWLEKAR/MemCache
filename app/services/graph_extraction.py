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
) -> list[tuple[str, str]]:
    """Pairs of entity *surface* strings whose spans are within ``window_tokens`` (or overlap)."""
    ents = [e for e in doc.ents if e.text.strip()]
    pairs: list[tuple[str, str]] = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            g = _gap_between_spans(ents[i], ents[j])
            if g <= window_tokens:
                pairs.append((ents[i].text.strip(), ents[j].text.strip()))
    return pairs


def ner_entity_texts(doc: "spacy.tokens.Doc") -> list[str]:
    """Entity texts from NER (order preserved, rough dedupe by normalized text)."""
    seen: set[str] = set()
    out: list[str] = []
    for ent in doc.ents:
        t = ent.text.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


_DECISION_PATTERNS = [
    re.compile(
        r"(?i)(?:decided to|we decided to|chose to|we chose|going with|will use|"
        r"agreed to use|committed to)\s+([^.;\n]+)",
    ),
    re.compile(r"(?i)(?:we will|we'll)\s+([^.;\n]+)"),
]

_PREFERENCE_PATTERNS = [
    re.compile(
        r"(?i)(?:prefer|prefers|preference for|loves?|likes?|favors?)\s+(?:to\s+)?([^.;\n]+)",
    ),
]


def extract_decisions_preferences_regex(text: str) -> tuple[list[str], list[str]]:
    """Lightweight regex extraction for Decision / Preference nodes."""
    decisions: list[str] = []
    preferences: list[str] = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(1).strip()
            if len(s) > 2:
                decisions.append(s)
    for pat in _PREFERENCE_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(1).strip()
            if len(s) > 2:
                preferences.append(s)
    return sorted(set(decisions)), sorted(set(preferences))
