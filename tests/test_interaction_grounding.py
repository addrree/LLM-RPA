from __future__ import annotations

import pytest

from app.interaction.action_grounder import ActionGrounder
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator, PlanValidationError


def _candidates():
    return [
        {"candidate_id": "u", "kind": "textbox", "selector": "#u", "placeholder": "Username", "enabled": True},
        {"candidate_id": "p", "kind": "textbox", "selector": "#p", "placeholder": "Password", "enabled": True},
        {"candidate_id": "submit", "kind": "button", "selector": "#login", "text": "Login", "enabled": True},
        {"candidate_id": "name", "kind": "textbox", "selector": "#name", "placeholder": "Name", "enabled": True},
    ]


def test_action_grounder_maps_enter_text_to_fill():
    result = ActionGrounder().ground({"intent": "enter_text", "target": "Name", "value": "Alice"}, _candidates())
    assert result.actions[0].action == "fill"
    assert result.actions[0].args["selector"] == "#name"
    assert result.grounding_strategy == "fill_candidate"


def test_action_grounder_maps_generic_multi_field_sequence():
    result = ActionGrounder().ground(
        {
            "intent": "fill_fields",
            "fields": {"Username": "alice", "Password": "secret"},
            "completion_target": "Login",
        },
        _candidates(),
    )
    assert [action.action for action in result.actions] == ["fill", "fill", "click"]
    assert result.actions[0].args["selector"] == "#u"
    assert result.actions[1].args["selector"] == "#p"
    assert result.actions[2].args["selector"] == "#login"
    assert result.grounding_strategy == "multi_field_sequence"


def test_action_grounder_rejects_unknown_submit_fallback():
    with pytest.raises(ValueError, match="refusing Submit fallback|unknown target|Unable to ground click"):
        ActionGrounder().ground({"intent": "click", "target": "Does not exist"}, _candidates())


def test_task_spec_validates_new_actions():
    plan = TaskSpec.model_validate(
        {
            "goal": "fill form",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 10, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "done", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "fill", "args": {"target": "Name", "text": "Alice"}},
                {"step_id": 2, "action": "select_option", "args": {"target": "City", "option_text": "Paris"}},
                {"step_id": 3, "action": "check", "args": {"label": "Accept"}},
                {"step_id": 4, "action": "choose_date", "args": {"target": "Date", "date": "2026-05-17"}},
                {"step_id": 5, "action": "select_autocomplete", "args": {"target": "Search", "query": "App", "suggestion": "Apple"}},
                {"step_id": 6, "action": "finish", "args": {}},
            ],
        }
    )
    PlanValidator().validate(plan)


def test_task_spec_rejects_invalid_fill():
    plan = TaskSpec.model_validate(
        {
            "goal": "bad",
            "start_url": "https://example.com",
            "allowed_domains": [],
            "constraints": {"max_steps": 10, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "done", "required_fields": []},
            "steps": [{"step_id": 1, "action": "fill", "args": {"target": "Name"}}, {"step_id": 2, "action": "finish", "args": {}}],
        }
    )
    with pytest.raises(PlanValidationError, match="fill requires text/value"):
        PlanValidator().validate(plan)


def test_playwright_handlers_expose_new_methods():
    from app.executor.action_handlers import ActionHandlers

    handlers = ActionHandlers()
    for name in ["fill", "focus", "clear", "press", "hover", "select_option", "check", "uncheck", "select_autocomplete", "choose_date"]:
        assert callable(getattr(handlers, name))
