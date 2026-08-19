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
from app.services.task_hierarchy import adjudicate_placement, shortlist_candidates
from app.services.task_inference import adjudicate_task
from app.services.task_store import TaskHierarchyError, TaskStore
from app.services.workbench_store import claim_tool_calls, ensure_l4_schema
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
    ensure_l4_schema(_worker_engine)
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


def _write_task(
    neo_driver: Any,
    *,
    user_id: str,
    episode_id: int,
    summary: str,
) -> str | None:
    """Infer and link the task this episode advances; returns its id, if any.

    Best-effort by contract: any failure here — Ollama down, malformed verdict,
    graph hiccup — logs and returns None. An ingest never fails because task
    inference failed; L1–L3 are already durable by the time this runs. The
    returned id lets the L4 claim stamp tool calls with the task they served.
    """
    try:
        store = TaskStore(neo_driver)
        open_tasks = store.list_open_tasks(user_id, limit=settings.task_candidate_limit)
        verdict = adjudicate_task(summary, [(t.id, t.title) for t in open_tasks])
        if verdict is None or verdict.goal is None:
            return None
        if verdict.matches_task_id is not None:
            store.link_episode(verdict.matches_task_id, episode_id=episode_id)
            if verdict.task_complete:
                store.close_task(verdict.matches_task_id)
            return verdict.matches_task_id
        task_id = store.create_task(user_id, verdict.goal)
        store.link_episode(task_id, episode_id=episode_id)
        return task_id
    except Exception:
        logger.exception("task inference failed; continuing without task attachment")
        return None


def _title_similarity(embedder: Any):
    """Cosine over MiniLM embeddings of two short titles. The embedder is the
    worker's already-loaded model; the shortlist takes a callable so tests can
    inject a table."""
    import numpy as np

    def sim(a: str, b: str) -> float:
        va, vb = embedder.encode([a, b], normalize_embeddings=True)
        return float(np.dot(va, vb))

    return sim


def _write_hierarchy(
    neo_driver: Any,
    *,
    user_id: str,
    task_id: str,
    embedder: Any,
) -> None:
    """Place `task_id` in the user's goal tree, if the evidence and the model
    agree. Best-effort by contract — any failure logs and leaves the tree as
    it was. Runs only for a root subject; a parented Task is never re-placed.
    """
    if not settings.task_placement_enabled:
        return
    try:
        store = TaskStore(neo_driver)
        if store.get_parent(task_id) is not None:
            return
        subject = store.task_evidence(task_id)
        if subject is None:
            return
        pool = store.list_placement_candidates(
            user_id, subject_id=task_id, limit=settings.task_candidate_limit
        )
        short = shortlist_candidates(
            subject,
            pool,
            similarity=_title_similarity(embedder),
            limit=settings.task_placement_candidates,
            min_score=settings.task_placement_min_score,
        )
        if not short:
            return
        verdict = adjudicate_placement(subject.title, [(c.id, c.title) for c in short])
        if verdict is None:
            return
        by_id = {c.id: c for c in short}
        if verdict.relation == "child_of":
            store.set_parent(task_id, verdict.task_id)
        else:  # parent_of: adopt exactly one root
            target = by_id.get(verdict.task_id)
            if target is None or not target.is_root:
                logger.info("placement wanted to adopt non-root %s; skipping", verdict.task_id)
                return
            store.set_parent(verdict.task_id, task_id)
        logger.info("hierarchy: %s %s %s", task_id, verdict.relation, verdict.task_id)
    except TaskHierarchyError as exc:
        logger.warning("placement rejected by hierarchy invariant: %s", exc)
    except Exception:
        logger.exception("hierarchy placement failed; continuing without it")


def _claim_workbench(
    engine: Any,
    neo_driver: Any,
    *,
    session_id: str,
    episode_id: int,
    task_id: str | None,
) -> None:
    """Attach the session's unlinked tool calls to this episode, then mirror
    identity+outcome into the graph as ToolCall nodes.

    Best-effort: L4 is additive, so a failure here logs and the ingest stands.
    """
    try:
        claimed = claim_tool_calls(
            engine, session_id=session_id, episode_id=episode_id, task_id=task_id
        )
        if claimed:
            Neo4jStore(neo_driver).link_tool_calls(
                episode_id,
                [(c.id, c.tool_name, c.status, c.created_at.isoformat()) for c in claimed],
            )
    except Exception:
        logger.exception("workbench claim failed; tool calls stay unlinked for now")


def _write_profile(
    neo_driver: Any,
    *,
    user_id: str,
    session_id: str,
    episode_id: int | None,
    messages: list[dict[str, Any]],
    nlp: Any,
) -> None:
    """Resolve the canonical profile for this ingest.

    Kept separate from `_write_l3` because the profile rules need role-aware
    access to the message list, while `_write_l3` only receives flattened text.
    """
    from app.services.profile_extraction import resolve_profile_from_messages
    from app.services.profile_store import ProfileStore

    resolve_profile_from_messages(
        ProfileStore(neo_driver),
        user_id=user_id,
        session_id=session_id,
        episode_id=episode_id,
        messages=messages,
        nlp=nlp,
    )


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
    user_id: str | None = None,
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
        resolved_task_id: str | None = None
        try:
            _write_l3(
                neo_driver,
                session_id=session_id,
                episode_id=existing_episode_id,
                summary=summary,
                flat_text=flat_text,
                nlp=nlp,
            )
            if user_id:
                _write_profile(
                    neo_driver,
                    user_id=user_id,
                    session_id=session_id,
                    episode_id=existing_episode_id,
                    messages=messages,
                    nlp=nlp,
                )
                resolved_task_id = _write_task(
                    neo_driver,
                    user_id=user_id,
                    episode_id=existing_episode_id,
                    summary=summary,
                )
                if resolved_task_id:
                    _write_hierarchy(
                        neo_driver, user_id=user_id, task_id=resolved_task_id, embedder=embedder
                    )
            _claim_workbench(
                engine,
                neo_driver,
                session_id=session_id,
                episode_id=existing_episode_id,
                task_id=resolved_task_id,
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
                user_id=user_id,
            )
    except OperationalError as e:
        logger.exception("Postgres insert failed")
        raise self.retry(exc=e) from e

    main_task_id: str | None = None
    try:
        _write_l3(
            neo_driver,
            session_id=session_id,
            episode_id=episode_id,
            summary=summary,
            flat_text=flat_text,
            nlp=nlp,
        )
        if user_id:
            _write_profile(
                neo_driver,
                user_id=user_id,
                session_id=session_id,
                episode_id=episode_id,
                messages=messages,
                nlp=nlp,
            )
            main_task_id = _write_task(
                neo_driver,
                user_id=user_id,
                episode_id=episode_id,
                summary=summary,
            )
            if main_task_id:
                _write_hierarchy(
                    neo_driver, user_id=user_id, task_id=main_task_id, embedder=embedder
                )
        _claim_workbench(
            engine,
            neo_driver,
            session_id=session_id,
            episode_id=episode_id,
            task_id=main_task_id,
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
