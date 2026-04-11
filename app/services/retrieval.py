"""Hybrid retrieval service: Redis recent context + Postgres episodes + Neo4j facts."""

from __future__ import annotations

from typing import Any

import tiktoken

from app.api import services as api_services
from app.config import settings
from app.db.postgres import session_scope
from app.services.neo4j_store import GraphEntityRow, Neo4jStore
from app.services.postgres_store import EpisodeSearchResult, PostgresStore


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot proceed at all."""


def _source_tier(source_type: str) -> str:
    if source_type == "recent_message":
        return "L1"
    if source_type == "episode":
        return "L2"
    return "L3"


def _source(source_type: str, **details: Any) -> dict[str, Any]:
    return {
        "type": source_type,
        "tier": _source_tier(source_type),
        "details": details,
    }


def _embed_query(text: str) -> list[float]:
    model = api_services.get_query_embedder()
    vector = model.encode(text, normalize_embeddings=True)
    return [float(x) for x in vector.tolist()]


def _format_recent_messages(messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        lines.append(f"{message['role']}: {message['content']}")
        sources.append(_source("recent_message", session_id=message.get("session_id"), index=index))
    return lines, sources


def _episode_similarity(hit: EpisodeSearchResult) -> float:
    return 1.0 - hit.distance


def _format_episode_hits(hits: list[EpisodeSearchResult]) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for hit in hits:
        similarity = _episode_similarity(hit)
        lines.append(f"Episode {hit.id}: {hit.summary}")
        sources.append(
            _source(
                "episode",
                episode_id=hit.id,
                session_id=hit.session_id,
                distance=hit.distance,
                similarity=similarity,
            )
        )
    return lines, sources


def _focus_entities(query: str, entities: list[GraphEntityRow]) -> list[GraphEntityRow]:
    query_lower = query.lower()
    matched = [
        entity
        for entity in entities
        if entity.name in query_lower
        or ((entity.display_name or "").lower() in query_lower and (entity.display_name or "").strip())
    ]
    return matched or entities[:]


def _format_graph_facts(
    query: str,
    entities: list[GraphEntityRow],
    decisions_preferences: dict[str, list[str]],
    graph_store: Neo4jStore,
) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    sources: list[dict[str, Any]] = []

    for entity in entities[: settings.retrieval_max_graph_facts]:
        display = entity.display_name or entity.name
        lines.append(f"Session entity: {display}")
        sources.append(_source("graph_entity", name=entity.name))

    for entity in _focus_entities(query, entities):
        related = graph_store.query_related_entities(entity.name, max_hops=2)
        if not related:
            continue
        lines.append(f"Related to {entity.display_name or entity.name}: {', '.join(related[:5])}")
        sources.append(
            _source("graph_related", name=entity.name, related=related[:5])
        )
        if len(lines) >= settings.retrieval_max_graph_facts:
            break

    for decision in decisions_preferences.get("decisions", []):
        if len(lines) >= settings.retrieval_max_graph_facts:
            break
        lines.append(f"Decision: {decision}")
        sources.append(_source("decision", text=decision))

    for preference in decisions_preferences.get("preferences", []):
        if len(lines) >= settings.retrieval_max_graph_facts:
            break
        lines.append(f"Preference: {preference}")
        sources.append(_source("preference", text=preference))

    return lines[: settings.retrieval_max_graph_facts], sources[: settings.retrieval_max_graph_facts]


def _count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Offline fallback for local dev / CI when the encoding bundle is unavailable.
        return len(text.split())


def _merge_sections(
    sections: list[tuple[str, list[str], list[dict[str, Any]]]],
    *,
    max_tokens: int,
) -> tuple[str, list[dict[str, Any]], bool]:
    chosen_sections: list[str] = []
    chosen_sources: list[dict[str, Any]] = []
    truncated = False

    for title, lines, section_sources in sections:
        if not lines:
            continue
        kept_lines: list[str] = []
        for index, line in enumerate(lines):
            candidate_section = f"{title}:\n" + "\n".join(kept_lines + [line])
            candidate_document = "\n\n".join(chosen_sections + [candidate_section])
            if _count_tokens(candidate_document) > max_tokens:
                truncated = True
                break
            kept_lines.append(line)

        if not kept_lines:
            truncated = True
            continue

        chosen_sections.append(f"{title}:\n" + "\n".join(kept_lines))
        chosen_sources.extend(section_sources[: len(kept_lines)])

    return "\n\n".join(chosen_sections), chosen_sources, truncated


def retrieve_context(
    session_id: str,
    query: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Retrieve hybrid memory context for a session and query."""
    warnings: list[str] = []
    overall_status = "ok"
    token_budget = max_tokens if max_tokens and max_tokens > 0 else settings.retrieval_default_max_tokens

    try:
        redis_messages = api_services.get_redis_store().get_recent_messages(
            session_id,
            limit=settings.retrieval_max_recent_messages,
        )
    except Exception as exc:
        raise RetrievalError("Redis retrieval failed") from exc

    recent_messages = [
        {
            **message,
            "session_id": session_id,
        }
        for message in redis_messages
    ]
    recent_lines, recent_sources = _format_recent_messages(recent_messages)

    episode_lines: list[str] = []
    episode_sources: list[dict[str, Any]] = []
    try:
        query_embedding = _embed_query(query)
        engine = api_services.get_postgres_engine()
        with session_scope(engine) as session:
            postgres_store = PostgresStore(session)
            hits = postgres_store.search_episodes(
                query_embedding,
                session_id,
                limit=settings.retrieval_max_episodes,
            )
        filtered_hits = [
            hit
            for hit in hits
            if _episode_similarity(hit) >= settings.retrieval_similarity_threshold
        ]
        episode_lines, episode_sources = _format_episode_hits(filtered_hits)
    except Exception:
        overall_status = "degraded"
        warnings.append("PostgreSQL retrieval unavailable; returning partial context")

    graph_lines: list[str] = []
    graph_sources: list[dict[str, Any]] = []
    try:
        graph_store = Neo4jStore(api_services.get_neo4j_driver())
        entities = graph_store.query_session_entities(session_id)
        decisions_preferences = graph_store.query_decisions_preferences(session_id)
        graph_lines, graph_sources = _format_graph_facts(
            query,
            entities,
            decisions_preferences,
            graph_store,
        )
    except Exception:
        overall_status = "degraded"
        warnings.append("Neo4j retrieval unavailable; returning partial context")

    context, sources, truncated = _merge_sections(
        [
            ("Recent Conversation", recent_lines, recent_sources),
            ("Relevant Past Episodes", episode_lines, episode_sources),
            ("Graph Facts", graph_lines, graph_sources),
        ],
        max_tokens=token_budget,
    )

    if truncated:
        warnings.append("Context truncated to fit max_tokens")

    if not context.strip() and recent_lines:
        context = "Recent Conversation:\n" + recent_lines[0]
        sources = recent_sources[:1]
        overall_status = "degraded"
        warnings.append("Context reduced to minimal recent message due to token budget")

    return {
        "context": context,
        "sources": sources,
        "status": overall_status,
        "warnings": warnings,
    }
