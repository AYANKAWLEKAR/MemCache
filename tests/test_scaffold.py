"""Tests that verify project scaffolding and configuration load correctly."""

import pytest

from app.config import settings


def test_settings_load():
    """Configuration loads with defaults or from env."""
    assert settings.redis_url is not None
    assert settings.postgres_url is not None
    assert settings.neo4j_uri is not None


def test_api_keys_parsing():
    """API keys are parsed into a set."""
    keys = settings.get_valid_api_keys()
    assert isinstance(keys, set)
    assert len(keys) >= 1
    assert "dummy-api-key-123" in keys or len(keys) > 0


def test_embedding_dimension():
    """Embedding dimension matches all-MiniLM-L6-v2."""
    assert settings.embedding_dimension == 384
