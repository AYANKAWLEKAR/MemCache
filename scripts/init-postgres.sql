-- L2 schema: episodic summaries + pgvector (PRD). Extension required for `vector` type.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    embedding vector(384) NOT NULL,  -- must match app.config Settings.embedding_dimension
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    metadata JSONB
);

-- Session filter for retrieval (PRD: B-tree on session_id).
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes (session_id);

-- IVFFlat cosine index: pgvector requires row count >= lists at build time, so it is not
-- created here on an empty database. After loading data, run:
--   psql ... -f scripts/create-ivfflat-index.sql
-- or call `ensure_ivfflat_index(engine)` from Python (see tests and STEP_3_PLAN.md).
