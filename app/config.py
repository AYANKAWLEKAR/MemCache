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
    ollama_model: str = "qwen2.5:3b"
    # Optional: sent as Authorization Bearer for hosted/custom gateways; local Ollama ignores it
    ollama_api_key: str = "dummy-ollama-api-key"

    # Demo agent
    demo_memcache_base_url: str = "http://localhost:8000"
    demo_memcache_api_key: str = "dummy-api-key-123"
    demo_ollama_base_url: str | None = None
    demo_ollama_model: str | None = None
    demo_ollama_api_key: str | None = None

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
    # Cosine-similarity floor for L2 episode recall.
    #
    # Calibrated against the default embedding model (all-MiniLM-L6-v2), whose
    # query-to-summary similarities are far lower than the 0.7 that was here
    # before: measured 0.237-0.673 for genuinely relevant pairs and -0.085-0.200
    # for irrelevant ones. A 0.7 floor sits above the *maximum* achievable score,
    # so every episode was filtered out and L2 recall never returned anything.
    # 0.25 clears the observed distractor ceiling with margin while keeping
    # relevant episodes. Retune if you change `embedding_model` — the scale is
    # model-specific, not universal.
    retrieval_similarity_threshold: float = 0.25
    # Cross-session recall: episodes lose half their ranking weight every N days,
    # so recent context wins ties without discarding strong older matches. Applied
    # to ordering only — never to the threshold above.
    retrieval_recency_half_life_days: float = 30.0
    # Candidates fetched per requested episode before recency reranking. Reranking
    # happens in Python because ordering by a computed score in SQL would defeat
    # the IVFFlat index; this factor is the accuracy/cost dial for that trade.
    retrieval_overfetch_factor: int = 4

    # Task inference (L3 Task nodes). Candidate cap bounds the adjudication
    # prompt: with an unbounded open-task list the prompt outgrows the model and
    # judgement quality collapses, so the cap is enforced in code, not requested
    # politely of the model.
    task_candidate_limit: int = 20

    # L4 workbench (tool calls). Outputs truncate aggressively; errors keep a
    # much larger cap because a stack trace is the highest-value payload in the
    # tier and is usually small enough to keep whole.
    workbench_output_max_bytes: int = 8192
    workbench_error_max_bytes: int = 32768
    workbench_max_failures_in_context: int = 5

    # Proactive retrieval (activation spreading over the weighted graph).
    # floor/decay are CALIBRATED, not chosen — see the spec §5 and
    # scripts/calibrate_activation.py. Measured on the live graph (2026-08-14,
    # seed 'clickhouse', 13 edges): Episode 0.164, Task 0.131, ToolCall 0.118
    # (three hops: entity->episode->INVOKED), UserProfile 0.094 — every kind the
    # feature must reach lands at 0.09–0.20, a clean band above the 0.05 floor,
    # while a second count-1 RELATED_TO hop (~0.008) dies. Structural edges
    # carry their full prior (see activation.COUNTED_EDGES); without that, the
    # ToolCall measured 0.027 and was lost. Radius bounds only the *fetch*.
    proactive_activation_floor: float = 0.05
    proactive_decay_per_hop: float = 0.8
    proactive_task_seed: float = 0.6
    proactive_fetch_radius: int = 4
    proactive_max_items: int = 8

    def get_valid_api_keys(self) -> set[str]:
        """Return set of valid API keys for auth."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
