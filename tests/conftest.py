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

# L3 synthetic graph seeds (entity names, related pairs, decisions/preferences)
SYNTHETIC_L3_ENTITIES_ACME = [
    "John",
    "Acme Corp",
    "Python",
]
SYNTHETIC_L3_RELATED_PAIRS = [
    ("John", "Acme Corp"),
    ("Acme Corp", "Python"),
]
SYNTHETIC_L3_DECISIONS = ["Use Python for the backend service."]
SYNTHETIC_L3_PREFERENCES = ["Dark mode UI."]

# L2 synthetic episode summaries (paired with synthetic embeddings in tests)
SYNTHETIC_EPISODE_SUMMARIES = {
    "session_alpha": [
        "User John from Acme prefers dark mode and chose Python.",
        "Team discussed database choice and agreed on PostgreSQL.",
        "Unrelated episode about hiking and weather.",
    ],
    "session_beta": [
        "Sarah at TechCo aligned on Monday deadlines.",
    ],
}


def unit_embedding_384(
    *,
    primary_axis: int = 0,
    secondary_axis: int | None = None,
    secondary_weight: float = 0.0,
) -> list[float]:
    """Deterministic 384-d unit vector for tests (no ML deps)."""
    dim = 384
    v = [0.0] * dim
    v[primary_axis % dim] = 1.0
    if secondary_axis is not None and secondary_weight != 0.0:
        v[secondary_axis % dim] = secondary_weight
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


@pytest.fixture
def fake_redis():
    """In-memory Redis for fast unit tests."""
    import fakeredis

    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()
