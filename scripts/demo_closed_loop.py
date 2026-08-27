"""Closed-loop demo: memory changes an agent's behavior, on camera.

Act 1 — session A: the user states a goal; a migration tool call fails and is
recorded in the L4 workbench; the conversation is ingested.

Act 2 — session B (a brand-new conversation): MemCache context is retrieved,
and the same model is asked the same operational question twice — once WITH the
retrieved memory, once WITHOUT. The with-memory agent avoids the failed
workflow; the without-memory agent walks straight back into it.

The tool failure itself is scripted (this demo is about memory, not alembic),
and the script says so. Everything else is real: real API, real worker path
(Celery eager), real Ollama summarization/adjudication, real Postgres/Neo4j/
Redis underneath. Run with the docker-compose stack and Ollama up:

    .venv/bin/python scripts/demo_closed_loop.py [--keep]
"""

from __future__ import annotations

import argparse
import sys
import uuid

import httpx

sys.path.insert(0, ".")  # run from the repo root

from app.workers.celery_app import celery_app  # noqa: E402

celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

AUTH = {"X-API-Key": next(iter(settings.get_valid_api_keys()))}

# The question deliberately contains NO hint of what went wrong on Monday.
# Anything the with-memory agent knows about the failure can only have come
# from MemCache — that is the point of the control.
QUESTION = (
    "You are resuming work on the telemetry schema migration. "
    "What is your very first action, and why? Answer in one short sentence."
)


def ask_model(question: str, memory_context: str | None) -> str:
    """One agent turn: the question, optionally grounded in retrieved memory."""
    context_block = (
        f"Context retrieved from your memory system:\n{memory_context}\n\n"
        if memory_context
        else ""
    )
    prompt = (
        f"{context_block}{question}"
        if context_block
        else f"You have no memory of previous sessions.\n\n{question}"
    )
    resp = httpx.post(
        settings.ollama_base_url.rstrip("/") + "/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="skip cleanup")
    args = parser.parse_args()

    uid = f"demo-{uuid.uuid4().hex[:6]}"
    session_a, session_b = f"{uid}-monday", f"{uid}-thursday"

    with TestClient(app) as client:
        banner("ACT 1 — Monday, session A: work happens, and some of it fails")
        print(f"  user_id={uid}  session={session_a}")
        print("  $ alembic upgrade 0042        (scripted failure, recorded for real)")
        fail = client.post(
            "/workbench/tool-call",
            headers=AUTH,
            json={
                "session_id": session_a,
                "user_id": uid,
                "tool_name": "alembic",
                "status": "error",
                "args": {"command": "upgrade", "revision": "0042"},
                "error": "DuplicateColumn: column user_id already exists on episodes",
                "duration_ms": 412,
            },
        )
        fail.raise_for_status()
        print(f"  -> recorded to L4 workbench (id={fail.json()['id']})")

        print("  user tells the assistant what they're doing; MemCache ingests it")
        client.post(
            "/memory/ingest",
            headers=AUTH,
            json={
                "session_id": session_a,
                "user_id": uid,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "I'm trying to migrate the telemetry schema to "
                            "ClickHouse, but the alembic migration just failed."
                        ),
                    },
                    {"role": "assistant", "content": "Noted — the migration errored."},
                ],
            },
        ).raise_for_status()
        print("  -> episode embedded (L2), entities+task in the graph (L3), failure claimed (L4)")

        banner("ACT 2 — Thursday, session B: a NEW conversation, same user")
        retrieved = client.post(
            "/memory/retrieve",
            headers=AUTH,
            json={
                "session_id": session_b,
                "user_id": uid,
                "query": "Picking the telemetry migration back up. What should I know?",
                "max_tokens": 900,
            },
        ).json()
        print("  retrieved context:")
        for line in retrieved["context"].splitlines():
            print(f"    | {line}")

        banner("THE BEHAVIOR DELTA — same model, same question")
        print(f"  Q: {QUESTION}\n")
        with_memory = ask_model(QUESTION, retrieved["context"])
        without_memory = ask_model(QUESTION, None)
        print(f"  WITH memory   : {with_memory}")
        print(f"  WITHOUT memory: {without_memory}")

    if not args.keep:
        _cleanup(uid, session_a, session_b)
        print("\n  (demo data cleaned up — rerun any time)")


def _cleanup(uid: str, *sessions: str) -> None:
    import redis

    from app.db.neo4j import create_driver_from_settings
    from app.db.postgres import create_engine_from_settings

    r = redis.from_url(settings.redis_url, decode_responses=True)
    for sid in sessions:
        r.delete(f"session:{sid}")
    r.close()

    engine = create_engine_from_settings()
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (uid,))
        conn.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    engine.dispose()

    driver = create_driver_from_settings()
    with driver.session() as s:
        s.run(
            """
            MATCH (p:UserProfile {user_id: $uid})
            OPTIONAL MATCH (p)-[:PURSUES]->(t:Task)
            OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(a:ProfileAttribute)
            OPTIONAL MATCH (p)-[:HAS_ALIAS]->(al:Entity)
            DETACH DELETE p, t, a, al
            """,
            uid=uid,
        )
        for sid in sessions:
            s.run(
                """
                MATCH (se:Session {id: $sid})
                OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode)
                OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp)
                OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall)
                DETACH DELETE se, ep, dp, tc
                """,
                sid=sid,
            )
    driver.close()


if __name__ == "__main__":
    main()
