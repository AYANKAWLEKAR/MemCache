"""Unit tests for the terminal demo agent and provenance rendering."""

from __future__ import annotations

from app.demo_agent import DemoAgent, build_agent_prompt, render_provenance_log


def test_render_provenance_log_groups_sources_by_tier():
    log = render_provenance_log(
        status="degraded",
        warnings=["Neo4j retrieval unavailable; returning partial context"],
        sources=[
            {
                "type": "recent_message",
                "tier": "L1",
                "details": {"session_id": "sess-1", "index": 0},
            },
            {
                "type": "episode",
                "tier": "L2",
                "details": {
                    "episode_id": 7,
                    "session_id": "sess-1",
                    "similarity": 0.91,
                    "distance": 0.09,
                },
            },
            {
                "type": "decision",
                "tier": "L3",
                "details": {"text": "Use Python for the backend service."},
            },
        ],
    )

    assert "retrieval_status: degraded" in log
    assert "Neo4j retrieval unavailable" in log
    assert "L1:" in log and "recent_message session=sess-1 index=0" in log
    assert "L2:" in log and "episode id=7 session=sess-1 similarity=0.910 distance=0.090" in log
    assert "L3:" in log and "decision text=Use Python for the backend service." in log


def test_build_agent_prompt_mentions_missing_memory_instructions():
    prompt = build_agent_prompt(user_message="What did John decide?", memory_context="")

    assert "If memory is missing or insufficient, say so plainly" in prompt
    assert "No memory context was retrieved." in prompt
    assert "User message: What did John decide?" in prompt


def test_chat_turn_retrieves_generates_and_ingests_in_order(monkeypatch):
    agent = DemoAgent(
        memcache_base_url="http://memcache.test",
        memcache_api_key="demo-key",
        ollama_base_url="http://ollama.test",
        ollama_model="llama3",
        ollama_api_key=None,
        session_id="sess-123",
    )
    events: list[tuple[str, object]] = []

    def fake_retrieve(user_message, max_tokens=None):
        events.append(("retrieve", user_message, max_tokens))
        return {
            "context": "Recent Conversation:\nuser: John likes dark mode.",
            "sources": [
                {
                    "type": "recent_message",
                    "tier": "L1",
                    "details": {"session_id": "sess-123", "index": 0},
                }
            ],
            "status": "degraded",
            "warnings": ["PostgreSQL retrieval unavailable; returning partial context"],
        }

    def fake_generate(user_message, memory_context):
        events.append(("generate", user_message, memory_context))
        return "John likes dark mode."

    def fake_ingest(user_message, answer):
        events.append(("ingest", user_message, answer))
        return {"status": "accepted", "task_id": "task-1", "session_id": "sess-123"}

    monkeypatch.setattr(agent, "retrieve_memory", fake_retrieve)
    monkeypatch.setattr(agent, "generate_answer", fake_generate)
    monkeypatch.setattr(agent, "ingest_turn", fake_ingest)

    result = agent.chat_turn("What does John prefer?", max_tokens=200)

    assert result["answer"] == "John likes dark mode."
    assert "retrieval_status: degraded" in result["provenance"]
    assert "PostgreSQL retrieval unavailable" in result["provenance"]
    assert "L1:" in result["provenance"]
    assert result["ingest"]["task_id"] == "task-1"
    assert events == [
        ("retrieve", "What does John prefer?", 200),
        ("generate", "What does John prefer?", "Recent Conversation:\nuser: John likes dark mode."),
        ("ingest", "What does John prefer?", "John likes dark mode."),
    ]
