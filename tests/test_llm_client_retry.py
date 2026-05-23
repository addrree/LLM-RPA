import pytest
import requests

from app.utils.llm_client import LLMClient, LLMClientError


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def test_ollama_chat_retries_once_on_500_and_records_retry_diagnostics():
    client = LLMClient(backend="ollama", planner_model="qwen3-vl:4b", verifier_model="qwen3-vl:4b", timeout_sec=1)
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(500, {}, text="internal error")
        return _FakeResponse(200, {"message": {"content": '{"ok": true}'}})

    client.session.post = fake_post

    content = client._ollama_chat(model="qwen3-vl:4b", system_prompt="s", user_prompt="u", image_path=None)
    assert content == '{"ok": true}'
    assert calls["count"] == 2
    assert client.last_chat_diagnostics["transport_retry_used"] is True
    assert client.last_chat_diagnostics["transport_retry_reason"] == "http_500"


def test_ollama_chat_retries_once_on_timeout_and_records_retry_diagnostics():
    client = LLMClient(backend="ollama", planner_model="qwen3-vl:4b", verifier_model="qwen3-vl:4b", timeout_sec=1)
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.Timeout("timeout")
        return _FakeResponse(200, {"message": {"content": '{"ok": true}'}})

    client.session.post = fake_post

    content = client._ollama_chat(model="qwen3-vl:4b", system_prompt="s", user_prompt="u", image_path=None)
    assert content == '{"ok": true}'
    assert calls["count"] == 2
    assert client.last_chat_diagnostics["transport_retry_used"] is True
    assert client.last_chat_diagnostics["transport_retry_reason"] == "timeout"


def test_ollama_chat_uses_thinking_json_block_when_content_empty():
    client = LLMClient(backend="ollama", planner_model="qwen3-vl:4b", verifier_model="qwen3-vl:4b", timeout_sec=1)
    client.session.post = lambda *args, **kwargs: _FakeResponse(200, {"message": {"content": "", "thinking": 'reason\n{"ok": true}\n'}, "response": ""})
    content = client._ollama_chat(model="qwen3-vl:4b", system_prompt="s", user_prompt="u", image_path=None)
    assert content == '{"ok": true}'
    assert client.last_chat_diagnostics["content_source"] == "message.thinking_json_block"


def test_ollama_chat_raises_clear_error_for_thinking_without_json():
    client = LLMClient(backend="ollama", planner_model="qwen3-vl:4b", verifier_model="qwen3-vl:4b", timeout_sec=1)
    client.session.post = lambda *args, **kwargs: _FakeResponse(200, {"message": {"content": "", "thinking": "plain reasoning prose"}, "response": ""})
    with pytest.raises(LLMClientError, match="reasoning/thinking but no JSON content"):
        client._ollama_chat(model="qwen3-vl:4b", system_prompt="s", user_prompt="u", image_path=None)
    assert client.last_chat_diagnostics["content_source"] == "empty_content_with_thinking"


def test_ollama_chat_rejects_non_json_object_like_thinking_block():
    client = LLMClient(backend="ollama", planner_model="qwen3-vl:4b", verifier_model="qwen3-vl:4b", timeout_sec=1)
    client.session.post = lambda *args, **kwargs: _FakeResponse(
        200,
        {"message": {"content": "", "thinking": "{x: 135, y: 132, width: 5.5625, height: 11}"}, "response": ""},
    )
    with pytest.raises(LLMClientError, match="reasoning/thinking but no JSON content"):
        client._ollama_chat(model="qwen3-vl:4b", system_prompt="s", user_prompt="u", image_path=None)
    assert client.last_chat_diagnostics["content_source"] == "empty_content_with_thinking"
