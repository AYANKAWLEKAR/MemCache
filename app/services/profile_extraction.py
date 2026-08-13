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
