"""Celery ``process_conversation`` wiring (mocked resources)."""

from unittest.mock import MagicMock, patch

import pytest

from app.workers import tasks as worker_tasks


@pytest.fixture
def reset_worker_globals():
    """Isolate tests that touch module-level worker state."""
    worker_tasks._worker_engine = None
    worker_tasks._worker_neo4j = None
    worker_tasks._worker_nlp = None
    worker_tasks._worker_embedder = None
    yield
    worker_tasks._worker_engine = None
    worker_tasks._worker_neo4j = None
    worker_tasks._worker_nlp = None
    worker_tasks._worker_embedder = None


def test_process_conversation_skips_when_summarization_fails(reset_worker_globals):
    mock_engine = MagicMock()
    mock_neo = MagicMock()
    mock_nlp = MagicMock()
    mock_embedder = MagicMock()
    store_inst = MagicMock()
    store_inst.find_episode_id_by_celery_task.return_value = None

    with patch.object(worker_tasks, "_resources", return_value=(mock_engine, mock_neo, mock_nlp, mock_embedder)):
        with patch.object(worker_tasks, "session_scope") as ss:
            ss.return_value.__enter__.return_value = MagicMock()
            ss.return_value.__exit__.return_value = None
            with patch("app.workers.tasks.PostgresStore", return_value=store_inst):
                with patch.object(worker_tasks, "summarize_conversation_ollama", return_value=None):
                    from app.workers.tasks import process_conversation

                    r = process_conversation.apply(
                        args=("sess-1", [{"role": "user", "content": "x"}], {}),
                    ).get()
    assert r["status"] == "skipped"
    assert r["reason"] == "summarization_failed"


def test_process_conversation_inserts_and_writes_l3(reset_worker_globals):
    mock_engine = MagicMock()
    mock_neo = MagicMock()
    mock_nlp = MagicMock()
    mock_embedder = MagicMock()

    class _Vec:
        def tolist(self):
            return [0.1] * 384

    mock_embedder.encode.return_value = _Vec()

    doc = MagicMock()
    mock_nlp.return_value = doc
    doc.ents = []

    store_inst = MagicMock()
    store_inst.find_episode_id_by_celery_task.return_value = None
    store_inst.insert_episode.return_value = 42

    with patch.object(worker_tasks, "_resources", return_value=(mock_engine, mock_neo, mock_nlp, mock_embedder)):
        with patch.object(worker_tasks, "session_scope") as ss:
            ss.return_value.__enter__.return_value = MagicMock()
            ss.return_value.__exit__.return_value = None
            with patch("app.workers.tasks.PostgresStore", return_value=store_inst):
                with patch.object(worker_tasks, "summarize_conversation_ollama", return_value="A summary."):
                    with patch.object(worker_tasks, "_write_l3") as w3:
                        from app.workers.tasks import process_conversation

                        r = process_conversation.apply(
                            args=("sess-2", [{"role": "user", "content": "hello"}], {"src": "test"}),
                        ).get()

    assert r["status"] == "ok"
    assert r["episode_id"] == 42
    w3.assert_called_once()
