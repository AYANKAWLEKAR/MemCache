"""Shared pytest fixtures and synthetic conversation data."""

import pytest

SYNTHETIC_SESSION_1 = [
    {
        "role": "user",
        "content": (
            "I'm John from Acme Corp. I prefer dark mode and decided to use Python "
            "for the project."
        ),
    },
    {
        "role": "assistant",
        "content": "Noted. John from Acme Corp prefers dark mode and chose Python.",
    },
]

SYNTHETIC_SESSION_2 = [
    {
        "role": "user",
        "content": (
            "We chose PostgreSQL for the database. Sarah at TechCo agreed on "
            "Monday deadlines."
        ),
    },
]


@pytest.fixture
def fake_redis():
    """In-memory Redis for fast unit tests."""
    import fakeredis

    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()
