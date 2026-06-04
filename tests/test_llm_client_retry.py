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


def test_ollama_chat_retries_once_on_timeout_and_records_retry_diagnostics(monkeypatch):
    monkeypatch.setenv("OLLAMA_RETRY_ON_TIMEOUT", "1")
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


def test_planner_disables_thinking_so_json_is_returned_in_content():
    client = LLMClient(backend="ollama", planner_model="qwen3-next:80b", verifier_model="gpt-oss:20b", timeout_sec=1)
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _FakeResponse(
            200,
            {
                "message": {"content": '{"steps": []}', "thinking": ""},
                "done": True,
                "done_reason": "stop",
                "eval_count": 12,
            },
        )

    client.session.post = fake_post
    artifact = client.generate_planner_artifact(system_prompt="json only", user_prompt="goal", stage="initial_planner")

    assert captured["think"] is False
    assert captured["format"] == "json"
    assert artifact.parsed_response == {"steps": []}
    assert client.last_chat_diagnostics["think_requested"] is False
    assert client.last_chat_diagnostics["done_reason"] == "stop"
