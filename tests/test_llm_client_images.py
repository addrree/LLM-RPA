import pytest
import requests

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


class _ImageUnsupportedResponse:
    status_code = 400
    text = '{"error":"this model does not support image input"}'

    @property
    def ok(self):
        return False


class _VerifierResponse:
    status_code = 200
    text = ""

    @property
    def ok(self):
        return True

    def json(self):
        return {
            "message": {
                "content": '{"task_completed":true,"confidence":0.9,"verdict":"accept","issues":[],"summary":"ok"}'
            }
        }


def test_ollama_timeout_does_not_retry_by_default():
    client = LLMClient(backend="ollama", planner_model="m", verifier_model="m", timeout_sec=1)
    calls = 0

    def fake_post(url, json, headers=None, timeout=None):
        nonlocal calls
        calls += 1
        raise requests.Timeout("slow")

    client.session.post = fake_post
    with pytest.raises(LLMClientError):
        client.generate_planner_artifact("system", "user")
    assert calls == 1


def test_ollama_timeout_retry_is_env_opt_in(monkeypatch):
    monkeypatch.setenv("OLLAMA_RETRY_ON_TIMEOUT", "1")
    client = LLMClient(backend="ollama", planner_model="m", verifier_model="m", timeout_sec=1)
    calls = 0

    def fake_post(url, json, headers=None, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.Timeout("slow")
        return _FakeResponse()

    client.session.post = fake_post
    client.generate_planner_artifact("system", "user")
    assert calls == 2


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


def test_generate_verifier_artifact_retries_without_image_when_model_rejects_images():
    client = LLMClient(backend="ollama", planner_model="m", verifier_model="text-only", timeout_sec=1)
    payloads = []

    def fake_post(url, json, headers=None, timeout=None):
        payloads.append(json)
        if len(payloads) == 1:
            return _ImageUnsupportedResponse()
        return _VerifierResponse()

    client.session.post = fake_post
    artifact = client.generate_verifier_artifact("system", "user", images_base64=["abc"])

    assert payloads[0]["messages"][1]["images"] == ["abc"]
    assert "images" not in payloads[1]["messages"][1]
    assert artifact.parsed_response["verdict"] == "accept"
    assert artifact.generation.fallback_used is True
    assert client.last_chat_diagnostics["image_input_omitted_after_model_reject"] is True


def test_generate_verifier_artifact_uses_vision_model_for_image_inputs():
    client = LLMClient(
        backend="ollama",
        planner_model="planner",
        verifier_model="text-only",
        vision_model="vision",
        timeout_sec=1,
    )
    payloads = []

    def fake_post(url, json, headers=None, timeout=None):
        payloads.append(json)
        return _VerifierResponse()

    client.session.post = fake_post
    artifact = client.generate_verifier_artifact("system", "user", images_base64=["abc"])

    assert payloads[0]["model"] == "vision"
    assert payloads[0]["messages"][1]["images"] == ["abc"]
    assert artifact.generation.model == "vision"
    assert artifact.generation.fallback_used is False
    assert client.last_chat_diagnostics["vision_model_used"] is True


def test_generate_verifier_artifact_uses_text_model_without_images():
    client = LLMClient(
        backend="ollama",
        planner_model="planner",
        verifier_model="text-only",
        vision_model="vision",
        timeout_sec=1,
    )
    payloads = []

    def fake_post(url, json, headers=None, timeout=None):
        payloads.append(json)
        return _VerifierResponse()

    client.session.post = fake_post
    artifact = client.generate_verifier_artifact("system", "user")

    assert payloads[0]["model"] == "text-only"
    assert "images" not in payloads[0]["messages"][1]
    assert artifact.generation.model == "text-only"
    assert client.last_chat_diagnostics["vision_model_used"] is False
