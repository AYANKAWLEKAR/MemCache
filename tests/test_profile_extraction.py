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
