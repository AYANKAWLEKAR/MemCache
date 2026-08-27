"""Hybrid retrieval service: Redis recent context + Postgres episodes + Neo4j facts."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import tiktoken

from app.api import services as api_services
from app.config import settings
from app.db.postgres import session_scope
from app.services.activation import spread_activation
from app.services.neo4j_store import GraphEntityRow, Neo4jStore
from app.services.postgres_store import EpisodeSearchResult, PostgresStore
from app.services.proactive import assemble_activated, build_seeds, explain_path

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot proceed at all."""


def _source_tier(source_type: str) -> str:
    if source_type == "recent_message":
        return "L1"
    if source_type == "episode":
        return "L2"
    if source_type == "tool_failure":
        return "L4"
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


def recency_decay(
    end_time: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float,
) -> float:
    """Exponential decay factor in (0, 1] for an episode's age.

    Halves every `half_life_days`. Applied to similarity for *ordering* only —
    never to the similarity threshold, or an old-but-relevant episode would be
    filtered out rather than merely ranked lower.
    """
    reference = now or datetime.now(UTC)
    if end_time.tzinfo is None:
        # Postgres can return naive datetimes depending on driver/column type.
        end_time = end_time.replace(tzinfo=UTC)

    age_days = (reference - end_time).total_seconds() / 86400.0
    if age_days <= 0:
        # Clock skew must never boost a hit above its raw similarity.
        return 1.0
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def rerank_by_recency(
    hits: list[Any],
    *,
    now: datetime | None = None,
    half_life_days: float,
    limit: int,
) -> list[tuple[Any, float]]:
    """Order candidates by `similarity * decay(age)` and truncate to `limit`.

    Reranking happens here rather than in SQL because ordering by a computed
    expression would defeat the IVFFlat index, forcing a full scan of the user's
    entire history — exactly the thing that gets slow as history grows.
    """
    reference = now or datetime.now(UTC)
    scored = [
        (
            hit,
            _episode_similarity(hit)
            * recency_decay(hit.end_time, now=reference, half_life_days=half_life_days),
        )
        for hit in hits
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: max(0, limit)]


def _format_episode_hits(
    ranked: list[tuple[EpisodeSearchResult, float]],
    *,
    now: datetime | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render ranked episodes.

    Cross-session episodes are not labelled in the prose, so provenance carries
    `session_id`, `user_id`, and `age_days` — that is how a caller tells an
    episode from another conversation apart from one in this one.
    """
    reference = now or datetime.now(UTC)
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for hit, decayed_score in ranked:
        similarity = _episode_similarity(hit)
        end_time = hit.end_time
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)
        age_days = max(0.0, (reference - end_time).total_seconds() / 86400.0)

        lines.append(f"Episode {hit.id}: {hit.summary}")
        sources.append(
            _source(
                "episode",
                episode_id=hit.id,
                session_id=hit.session_id,
                user_id=hit.user_id,
                distance=hit.distance,
                similarity=similarity,
                age_days=round(age_days, 3),
                decayed_score=round(decayed_score, 6),
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


def _format_profile_facts(user_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Identity, attributes, and profile-scoped decisions/preferences.

    These are not session-scoped, which is what lets a fresh session know who it
    is talking to.
    """
    from app.services.profile_store import ProfileStore, resolve_attributes

    store = ProfileStore(api_services.get_neo4j_driver())
    row = store.get_profile(user_id)
    if row is None:
        return [], []

    lines: list[str] = []
    sources: list[dict[str, Any]] = []

    name = row.display_name or user_id
    lines.append(f"User: {name}")
    sources.append(
        _source("profile_identity", user_id=user_id, display_name=row.display_name)
    )

    for key, attr in sorted(resolve_attributes(store.get_attributes(user_id)).items()):
        if key == "name":
            continue
        lines.append(f"User {key}: {attr.value}")
        sources.append(
            _source("profile_identity", user_id=user_id, key=key, source=attr.source)
        )

    # The active task — the leaf being worked — rendered with its ancestor path
    # so the agent knows the larger objective it is serving. One line, one
    # source (the section merger pairs lines and sources 1:1).
    from app.services.task_store import TaskStore

    task_store = TaskStore(api_services.get_neo4j_driver())
    active = task_store.active_task(user_id)
    if active is not None:
        ancestors = task_store.get_ancestors(active.id)
        line = f"Current task: {active.title}"
        if ancestors:
            chain = " ▸ ".join(
                a.title + (" (done)" if a.status == "done" else "") for a in ancestors
            )
            line += f" (under: {chain})"
        lines.append(line)
        sources.append(
            _source(
                "task",
                task_id=active.id,
                title=active.title,
                status=active.status,
                lineage=[active.id] + [a.id for a in ancestors],
                depth=len(ancestors),
            )
        )

    for decision in store.get_profile_decisions(user_id)[
        : settings.retrieval_max_graph_facts
    ]:
        lines.append(f"User decision: {decision}")
        sources.append(_source("profile_decision", user_id=user_id, text=decision))

    for preference in store.get_profile_preferences(user_id)[
        : settings.retrieval_max_graph_facts
    ]:
        lines.append(f"User preference: {preference}")
        sources.append(_source("profile_preference", user_id=user_id, text=preference))

    return lines, sources


def _format_known_failures(user_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Recent failed tool calls the agent must already know about.

    Lineage-ranked when an active task exists (the leaf's failures, then its
    ancestors', then the user's others), else user-scoped. Only the first
    line of each error enters the context — the full payload stays in L4,
    reachable via the tool_call id in provenance.
    """
    from app.services.task_store import TaskStore
    from app.services.workbench_store import failed_calls

    engine = api_services.ensure_workbench_ready()
    task_store = TaskStore(api_services.get_neo4j_driver())
    active = task_store.active_task(user_id)
    lineage = task_store.get_lineage_ids(active.id) if active else []

    rows = failed_calls(
        engine,
        user_id=user_id,
        task_ids=lineage or None,
        limit=settings.workbench_max_failures_in_context,
    )

    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for row in rows:
        error_head = (row.error or "").strip().splitlines()[0] if row.error else "(no error text)"
        lines.append(f"Failed action: {row.tool_name} — {error_head}")
        sources.append(
            _source(
                "tool_failure",
                tool_call_id=row.id,
                tool_name=row.tool_name,
                task_id=row.task_id,
                session_id=row.session_id,
                error_head=error_head,
            )
        )
    return lines, sources


def _format_proactive_context(
    *,
    user_id: str,
    session_id: str,
    query: str,
    recent_messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Light the graph from what the conversation *surfaced* and render what glows.

    Seeds: entities in the recent L1 messages + the query (live, 1.0), plus
    entities the active Task touches (inherited, lower). Aliases collapse to
    the profile. Activation spreads over one pulled neighborhood; nodes above
    the floor are hydrated from the tier that owns them and rendered ranked by
    activation. Every source carries `activation` and `path` — the edge chain
    that lit it — so any line is explainable.
    """
    from app.services.graph_extraction import ner_entity_texts
    from app.services.neo4j_store import Neo4jStore, normalize_entity_name
    from app.services.profile_store import ProfileStore
    from app.services.task_store import TaskStore
    from app.services.workbench_store import get_tool_call

    driver = api_services.get_neo4j_driver()
    graph = Neo4jStore(driver)
    nlp = api_services.get_query_nlp()

    # 1. Live seeds: what was actually said, recently, plus the query.
    live_text = "\n".join(str(m.get("content", "")) for m in recent_messages) + "\n" + query
    live = [normalize_entity_name(t) for t in ner_entity_texts(nlp(live_text))]
    live = [n for n in live if n]

    # 2. Inherited seeds: what the active task already touches (topical, leaf
    #    only), plus the lineage Task nodes themselves (structural, decaying up
    #    the tree — see spec §4c).
    from app.services.proactive import lineage_task_seeds

    task_store = TaskStore(driver)
    task_entities: list[str] = []
    lineage: list[str] = []
    active_task = task_store.active_task(user_id)
    if active_task is not None:
        lineage = task_store.get_lineage_ids(active_task.id)
        with driver.session() as sess:
            task_entities = [
                r["name"]
                for r in sess.run(
                    "MATCH (:Task {id: $tid})<-[:ADVANCES]-(:Episode)-[:MENTIONS]->(e:Entity) "
                    "RETURN DISTINCT e.name AS name",
                    tid=active_task.id,
                )
            ]
    task_nodes = lineage_task_seeds(
        lineage,
        base=settings.proactive_task_node_seed,
        decay=settings.proactive_task_depth_decay,
    )

    # 3. Alias collapse: names that are this user's aliases seed the profile.
    alias_to_profile = {a: user_id for a in ProfileStore(driver).get_aliases(user_id)}
    seeds = build_seeds(
        live_entities=live,
        task_entities=task_entities,
        alias_to_profile=alias_to_profile,
        task_seed=settings.proactive_task_seed,
        task_nodes=task_nodes,
    )
    if not seeds:
        return [], []

    # 4. One round-trip neighborhood (entities + lineage tasks), spread in memory.
    entity_seed_names = [nid.split(":", 1)[1] for nid in seeds if nid.startswith("Entity:")]
    task_seed_ids = [nid.split(":", 1)[1] for nid in seeds if nid.startswith("Task:")]
    neighborhood = graph.fetch_neighborhood(
        entity_seed_names, task_ids=task_seed_ids, radius=settings.proactive_fetch_radius
    )
    result = spread_activation(
        neighborhood,
        seeds=seeds,
        floor=settings.proactive_activation_floor,
        decay=settings.proactive_decay_per_hop,
    )
    nodes = assemble_activated(neighborhood, result.scores, seeds=seeds)

    # 5. Hydrate + render, ranked by activation, seeds themselves skipped
    #    (they are already in the conversation — restating them is noise).
    lines: list[str] = []
    sources: list[dict[str, Any]] = []
    engine = api_services.get_postgres_engine()
    for node in nodes:
        if node.is_seed or len(lines) >= settings.proactive_max_items:
            continue
        path = explain_path(result.parents, target=node.node_id, seeds=seeds)
        path_str = " -> ".join(
            f"{src.split(':',1)[1]} -{rel}({cnt})-> {dst.split(':',1)[1]}"
            for src, rel, cnt, dst in path
        )
        common = {
            "activation": round(node.activation, 4),
            "path": [list(p) for p in path],
            "is_seed": False,
        }
        if node.label == "Episode":
            try:
                eid = int(node.key)
            except ValueError:
                continue
            # Read scalars INSIDE the session: session_scope commits on exit and
            # the ORM expires the instance, so touching attributes afterwards
            # raises DetachedInstanceError. (Found live; it was being swallowed.)
            with session_scope(engine) as sess:
                row = PostgresStore(sess).get_episode_by_id(eid)
                if row is None:
                    continue
                summary, ep_session = row.summary, row.session_id
            lines.append(f"Related episode {eid}: {summary}")
            sources.append(_source("proactive_episode", episode_id=eid,
                                   session_id=ep_session, via=path_str, **common))
        elif node.label == "ToolCall":
            try:
                tcid = int(node.key)
            except ValueError:
                continue
            call = get_tool_call(engine, tcid)
            if call is None:
                continue
            head = (call.error or call.output or "").strip().splitlines()[:1]
            detail = head[0] if head else ""
            verb = "Failed action" if call.status == "error" else "Prior action"
            lines.append(f"{verb} ({call.tool_name}): {detail}".rstrip(": "))
            sources.append(_source(
                "proactive_tool_failure" if call.status == "error" else "proactive_tool_call",
                tool_call_id=tcid, tool_name=call.tool_name, status=call.status,
                via=path_str, **common))
        elif node.label == "Entity":
            lines.append(f"Related entity: {node.key}")
            sources.append(_source("proactive_entity", name=node.key, via=path_str, **common))
        elif node.label == "Task":
            t = task_store.get_task(node.key)
            if t is None:
                continue
            lines.append(f"Related task: {t.title} ({t.status})")
            sources.append(_source("proactive_task", task_id=t.id, via=path_str, **common))
        elif node.label in {"Decision", "Preference"}:
            with driver.session() as sess:
                rec = sess.run(
                    f"MATCH (n:{node.label} {{id: $id}}) RETURN n.text AS text", id=node.key
                ).single()
            if rec and rec["text"]:
                lines.append(f"Related {node.label.lower()}: {rec['text']}")
                sources.append(_source(f"proactive_{node.label.lower()}", text=rec["text"],
                                       via=path_str, **common))
        # UserProfile / Session nodes: structural, nothing to render.
    return lines, sources


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
    user_id: str | None = None,
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
            # Over-fetch by raw distance so recency reranking has candidates to
            # promote. Without this, an old-but-similar episode occupies the slot
            # a recent one should win.
            candidate_limit = settings.retrieval_max_episodes * max(
                1, settings.retrieval_overfetch_factor
            )
            hits = postgres_store.search_episodes(
                query_embedding,
                session_id,
                limit=candidate_limit,
                user_id=user_id,
            )
        # Threshold applies to *raw* similarity. Filtering on the decayed score
        # would silently drop old-but-relevant episodes instead of ranking them
        # lower, which is the opposite of what decay is for.
        filtered_hits = [
            hit
            for hit in hits
            if _episode_similarity(hit) >= settings.retrieval_similarity_threshold
        ]
        ranked = rerank_by_recency(
            filtered_hits,
            half_life_days=settings.retrieval_recency_half_life_days,
            limit=settings.retrieval_max_episodes,
        )
        episode_lines, episode_sources = _format_episode_hits(ranked)
    except Exception:
        overall_status = "degraded"
        warnings.append("PostgreSQL retrieval unavailable; returning partial context")

    graph_lines: list[str] = []
    graph_sources: list[dict[str, Any]] = []
    try:
        if user_id:
            # Proactive: light the graph from what the conversation surfaced.
            graph_lines, graph_sources = _format_proactive_context(
                user_id=user_id,
                session_id=session_id,
                query=query,
                recent_messages=recent_messages,
            )
        else:
            # No identity to scope by: fall back to session-locked graph facts.
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
        # Degrade, never fail — but never silently either. A swallowed
        # DetachedInstanceError hid a real bug here once.
        logger.exception("graph/proactive retrieval failed; degrading")
        overall_status = "degraded"
        warnings.append("Neo4j retrieval unavailable; returning partial context")

    profile_lines: list[str] = []
    profile_sources: list[dict[str, Any]] = []
    failure_lines: list[str] = []
    failure_sources: list[dict[str, Any]] = []
    if user_id:
        try:
            profile_lines, profile_sources = _format_profile_facts(user_id)
        except Exception:
            overall_status = "degraded"
            warnings.append("Profile retrieval unavailable; returning partial context")
        try:
            failure_lines, failure_sources = _format_known_failures(user_id)
        except Exception:
            overall_status = "degraded"
            warnings.append("Workbench retrieval unavailable; returning partial context")

    # Profile sits directly after recent conversation so identity survives
    # truncation ahead of older episodes.
    context, sources, truncated = _merge_sections(
        [
            ("Recent Conversation", recent_lines, recent_sources),
            ("User Profile", profile_lines, profile_sources),
            # Failures sit ahead of episodes: an agent about to act needs "do
            # not repeat this" to survive truncation before older narrative.
            ("Known Failures", failure_lines, failure_sources),
            ("Relevant Past Episodes", episode_lines, episode_sources),
            ("Proactive Context" if user_id else "Graph Facts", graph_lines, graph_sources),
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
