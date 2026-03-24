-- Build IVFFlat index for cosine distance (`<=>`). Run only when episodes has enough rows
-- (lists must be <= row count). Adjust lists to sqrt(n) or n/1000 per pgvector guidance.

-- Example for >= 100 rows:
CREATE INDEX IF NOT EXISTS idx_episodes_embedding_ivfflat
    ON episodes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
