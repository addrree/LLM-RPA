import json

import pytest

from app.browsergym_integration.agent_adapter import BrowserGymAgentAdapter
from app.schemas.task_spec import TaskSpec


_PLAN = {
    "goal": "g",
    "start_url": "https://example.com",
    "allowed_domains": ["example.com"],
    "constraints": {"max_steps": 1, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 1},
    "expected_result": {"description": "d", "required_fields": []},
    "steps": [{"step_id": 1, "action": "finish", "args": {"answer": "done"}}],
}


class _Planner:
    def __init__(self):
        self.calls = []

    def build_plan(self, user_goal, **kwargs):
        self.calls.append({"user_goal": user_goal, "kwargs": kwargs})
        return TaskSpec.model_validate(_PLAN)


class _Validator:
    def validate(self, plan):
        return None


def test_use_vision_false_does_not_extract_or_send_images(monkeypatch):
    def fail_extract(*args, **kwargs):
        raise AssertionError("extractor should not be called")

    monkeypatch.setattr("app.browsergym_integration.agent_adapter.extract_browsergym_image_base64", fail_extract)
    planner = _Planner()
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), use_vision=False)
    decision = adapter.act("goal", {"url": "https://example.com", "text": "hello"}, {}, [])
    assert planner.calls[0]["kwargs"] == {}
    assert decision.vision_used is False
    assert decision.vision_image_present is False


def test_use_vision_true_with_image_sends_images_base64_and_keeps_reports_safe(monkeypatch):
    monkeypatch.setattr("app.browsergym_integration.agent_adapter.extract_browsergym_image_base64", lambda obs, info: "abc")
    planner = _Planner()
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), use_vision=True)
    decision = adapter.act("goal", {"url": "https://example.com", "text": "hello"}, {}, [])
    assert planner.calls[0]["kwargs"] == {"images_base64": ["abc"]}
    assert decision.vision_used is True
    assert decision.vision_image_present is True
    serialized = json.dumps({"internal_plan": decision.internal_plan, "selected_step": decision.selected_step})
    assert "abc" not in serialized


def test_use_vision_true_without_image_falls_back_to_text(monkeypatch):
    monkeypatch.setattr("app.browsergym_integration.agent_adapter.extract_browsergym_image_base64", lambda obs, info: None)
    planner = _Planner()
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), use_vision=True)
    decision = adapter.act("goal", {"url": "https://example.com", "text": "hello"}, {}, [])
    assert planner.calls[0]["kwargs"] == {}
    assert decision.vision_used is True
    assert decision.vision_image_present is False
