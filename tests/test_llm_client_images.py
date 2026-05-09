import pytest

from app.utils.llm_client import LLMClient, LLMClientError


class _FakeResponse:
    status_code = 200
    text = ""

    @property
    def ok(self):
        return True

    def json(self):
        return {"message": {"content": '{"goal":"g","start_url":"https://example.com","allowed_domains":["example.com"],"constraints":{"max_steps":1,"max_replans":1,"max_verification_retries":1,"timeout_sec":1},"expected_result":{"description":"d","required_fields":[]},"steps":[{"step_id":1,"action":"finish","args":{}}]}'}}


class _ErrorResponse:
    status_code = 403
    text = "model requires a subscription"

    @property
    def ok(self):
        return False


def test_generate_planner_artifact_sends_images_base64_payload():
    client = LLMClient(backend="ollama", planner_model="m", verifier_model="m", timeout_sec=1)
    captured = {}

    def fake_post(url, json, headers=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse()

    client.session.post = fake_post
    client.generate_planner_artifact("system", "user", images_base64=["abc"])
    assert captured["payload"]["messages"][1]["images"] == ["abc"]


def test_generate_planner_artifact_omits_images_when_not_provided():
    client = LLMClient(backend="ollama", planner_model="m", verifier_model="m", timeout_sec=1)
    captured = {}

    def fake_post(url, json, headers=None, timeout=None):
        captured["payload"] = json
        return _FakeResponse()

    client.session.post = fake_post
    client.generate_planner_artifact("system", "user")
    assert "images" not in captured["payload"]["messages"][1]


def test_ollama_cloud_subscription_403_mentions_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    client = LLMClient(backend="ollama_cloud", planner_model="qwen3.5:397b-cloud", verifier_model="m", timeout_sec=1)
    client.session.post = lambda *a, **k: _ErrorResponse()
    with pytest.raises(LLMClientError) as exc:
        client.generate_planner_artifact("system", "user")
    message = str(exc.value)
    assert "requires subscription/access" in message
    assert "qwen3.5:397b-cloud" in message
