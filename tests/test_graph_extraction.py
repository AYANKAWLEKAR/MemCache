"""Regex and pure helpers for graph extraction."""

from app.services.graph_extraction import extract_decisions_preferences_regex


def test_extract_decisions_preferences_regex():
    text = (
        "We decided to use PostgreSQL for storage. "
        "I prefer dark mode and love Python."
    )
    dec, pref = extract_decisions_preferences_regex(text)
    assert any("PostgreSQL" in d for d in dec)
    assert any("dark mode" in p.lower() for p in pref)
