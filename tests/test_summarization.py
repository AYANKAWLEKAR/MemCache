"""Ollama summarization client (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import httpx

from app.config import Settings
from app.services.summarization import summarize_conversation_ollama


def test_summarize_conversation_ollama_success():
    cfg = Settings(
        ollama_base_url="http://ollama:11434",
        ollama_model="mistral",
        ollama_api_key="dummy-ollama-api-key",
    )
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Short summary.", "done": True}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.services.summarization.httpx.Client") as client_cls:
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.post.return_value = mock_resp
        client_cls.return_value = inst

        out = summarize_conversation_ollama(
            [{"role": "user", "content": "Hello"}],
            settings=cfg,
        )

    assert out == "Short summary."
    call_kw = inst.post.call_args
    assert "Authorization" in call_kw[1]["headers"]
    assert call_kw[1]["headers"]["Authorization"] == "Bearer dummy-ollama-api-key"


def test_summarize_conversation_ollama_http_error_returns_none():
    cfg = Settings(ollama_api_key="dummy-ollama-api-key")
    with patch("app.services.summarization.httpx.Client") as client_cls:
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.post.side_effect = httpx.ConnectError("no server", request=MagicMock())
        client_cls.return_value = inst

        assert (
            summarize_conversation_ollama(
                [{"role": "user", "content": "Hi"}],
                settings=cfg,
            )
            is None
        )
