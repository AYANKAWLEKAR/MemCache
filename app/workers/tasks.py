"""Celery tasks: episodic processing (summarize, embed, L2, L3)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Task
from celery.signals import worker_process_init
from neo4j.exceptions import Neo4jError
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.graph_extraction import (
    conversation_text,
    entity_cooccurrence_pairs,
    extract_decisions_preferences_regex,
    ner_entity_texts,
)
from app.services.neo4j_store import Neo4jStore
from app.services.postgres_store import PostgresStore
from app.services.summarization import summarize_conversation_ollama
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_worker_engine = None
_worker_neo4j = None
_worker_nlp = None
_worker_embedder = None


def _ensure_worker_resources() -> None:
    """Open DB connections and load ML models once per process (worker or eager apply)."""
    global _worker_engine, _worker_neo4j, _worker_nlp, _worker_embedder
    if _worker_engine is not None:
        return
    import spacy
    from sentence_transformers import SentenceTransformer

    _worker_engine = create_engine_from_settings()
    ensure_l2_schema(_worker_engine)
    _worker_neo4j = create_driver_from_settings()
    ensure_constraints(_worker_neo4j)
    _worker_embedder = SentenceTransformer(settings.embedding_model)
    _worker_nlp = spacy.load(settings.spacy_model)
    logger.info(
        "worker resources ready: embedding=%s spacy=%s",
        settings.embedding_model,
        settings.spacy_model,
    )


@worker_process_init.connect
def _init_worker_process(**_kwargs: Any) -> None:
    """Preload models when running under ``celery worker`` (forked pool)."""
    _ensure_worker_resources()


def _resources() -> tuple[Any, Any, Any, Any]:
    _ensure_worker_resources()
    assert _worker_engine is not None and _worker_neo4j is not None
    assert _worker_nlp is not None and _worker_embedder is not None
    return _worker_engine, _worker_neo4j, _worker_nlp, _worker_embedder


def _embed_summary(model: Any, text: str) -> list[float]:
    v = model.encode(text, normalize_embeddings=True)
    return [float(x) for x in v.tolist()]


def _write_l3(
    neo_driver: Any,
    *,
    session_id: str,
    episode_id: int,
    summary: str,
    flat_text: str,
    nlp: Any,
) -> None:
    store = Neo4jStore(neo_driver)
    store.upsert_session(session_id)
    store.upsert_episode(session_id, episode_id, summary)

    doc = nlp(flat_text)
    entities = ner_entity_texts(doc)
    if entities:
        store.merge_entities(entities, episode_id=episode_id)

    pairs = entity_cooccurrence_pairs(doc, window_tokens=10)
    if pairs:
        store.create_relationships(pairs)

    decisions, preferences = extract_decisions_preferences_regex(flat_text)
    if decisions or preferences:
        store.record_decisions_and_preferences(episode_id, decisions, preferences)


@celery_app.task(
    bind=True,
    name="memcache.process_conversation",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def process_conversation(
    self: Task,
    session_id: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize (Ollama), embed, insert L2, then Session/Episode/entities/L3 edges."""
    meta = dict(metadata or {})
    task_id = getattr(self.request, "id", None) or ""
    engine, neo_driver, nlp, embedder = _resources()

    if not session_id or not isinstance(messages, list):
        return {"status": "error", "reason": "invalid_arguments"}

    flat_text = conversation_text(messages)

    existing_episode_id: int | None = None
    summary: str | None = None

    try:
        with session_scope(engine) as session:
            store = PostgresStore(session)
            if task_id:
                existing_episode_id = store.find_episode_id_by_celery_task(
                    session_id,
                    task_id,
                )
            if existing_episode_id is not None:
                row = store.get_episode_by_id(existing_episode_id)
                summary = row.summary if row is not None else None
    except OperationalError as e:
        logger.exception("Postgres read failed")
        raise self.retry(exc=e) from e

    if existing_episode_id is not None and summary:
        try:
            _write_l3(
                neo_driver,
                session_id=session_id,
                episode_id=existing_episode_id,
                summary=summary,
                flat_text=flat_text,
                nlp=nlp,
            )
        except (Neo4jError, OSError) as e:
            logger.exception("Neo4j write failed (retry path)")
            raise self.retry(exc=e) from e
        return {
            "status": "ok",
            "session_id": session_id,
            "episode_id": existing_episode_id,
            "deduped": True,
        }

    summary = summarize_conversation_ollama(messages, settings=settings)
    if not summary:
        logger.warning(
            "Skipping episode for session_id=%s: summarization unavailable or empty",
            session_id,
        )
        return {"status": "skipped", "reason": "summarization_failed"}

    now = datetime.now(timezone.utc)
    episode_meta: dict[str, Any] = {**meta, "celery_task_id": task_id} if task_id else meta

    try:
        embedding = _embed_summary(embedder, summary)
    except Exception as e:
        logger.exception("Embedding failed")
        raise self.retry(exc=e) from e

    episode_id: int
    try:
        with session_scope(engine) as session:
            store = PostgresStore(session)
            episode_id = store.insert_episode(
                session_id=session_id,
                summary=summary,
                embedding=embedding,
                start_time=now,
                end_time=now,
                metadata=episode_meta or None,
            )
    except OperationalError as e:
        logger.exception("Postgres insert failed")
        raise self.retry(exc=e) from e

    try:
        _write_l3(
            neo_driver,
            session_id=session_id,
            episode_id=episode_id,
            summary=summary,
            flat_text=flat_text,
            nlp=nlp,
        )
    except (Neo4jError, OSError) as e:
        logger.exception("Neo4j write failed after L2 insert; retry will reconcile graph")
        raise self.retry(exc=e) from e

    return {
        "status": "ok",
        "session_id": session_id,
        "episode_id": episode_id,
        "deduped": False,
    }
