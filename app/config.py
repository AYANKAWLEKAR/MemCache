"""Configuration via environment variables. Replace placeholder values with real credentials."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Uses env vars with sensible defaults for local dev."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API authentication - comma-separated valid keys
    api_keys: str = "dummy-api-key-123"

    # L1 Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl_seconds: int = 86400  # 24 hours
    redis_max_messages_per_session: int = 200

    # L2 PostgreSQL
    postgres_url: str = "postgresql://memcache:memcache@localhost:5432/memcache"

    # L3 Neo4j
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "dummy-neo4j-password"

    # Summarization (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    # Optional: sent as Authorization Bearer for hosted/custom gateways; local Ollama ignores it
    ollama_api_key: str = "dummy-ollama-api-key"

    # Celery (defaults share Redis host; use different DB index if you want isolation)
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # spaCy NER
    spacy_model: str = "en_core_web_sm"

    # Retrieval defaults
    retrieval_max_recent_messages: int = 10
    retrieval_max_episodes: int = 5
    retrieval_max_graph_facts: int = 10
    retrieval_default_max_tokens: int = 2000
    retrieval_similarity_threshold: float = 0.7

    def get_valid_api_keys(self) -> set[str]:
        """Return set of valid API keys for auth."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
