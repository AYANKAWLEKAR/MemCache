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
