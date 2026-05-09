import pytest

from app.browsergym_integration.plan_normalizer import (
    normalize_allowed_domains_for_browsergym,
    normalize_minwob_click_targets,
)
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator


def _plan(*, start_url="http://127.0.0.1:8765/miniwob/click-button.html", allowed_domains=None, steps=None):
    return TaskSpec.model_validate(
        {
            "goal": "Complete MiniWoB task",
            "start_url": start_url,
            "allowed_domains": ["127.0.0.1"] if allowed_domains is None else allowed_domains,
            "constraints": {"max_steps": 3, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 5},
            "expected_result": {"description": "done", "required_fields": []},
            "steps": steps
            or [
                {"step_id": 1, "action": "click", "args": {"text": "Submit"}},
                {"step_id": 2, "action": "finish", "args": {"answer": "done"}},
            ],
        }
    )


def test_minwob_click_text_normalization_adds_exact_true():
    plan = _plan(allowed_domains=["127.0.0.1:8765"])

    normalize_minwob_click_targets(plan, env_id="browsergym/miniwob.click-button")

    assert plan.steps[0].args == {"text": "Submit", "exact": True}
    PlanValidator().validate(plan)


def test_non_minwob_click_text_is_not_made_permissive():
    plan = _plan(start_url="https://example.com", allowed_domains=["example.com"])

    normalize_minwob_click_targets(plan, env_id="browsergym/openended")

    assert plan.steps[0].args == {"text": "Submit"}
    with pytest.raises(PlanValidationError, match="click with bare text is too weak"):
        PlanValidator().validate(plan)


def test_allowed_domains_normalization_adds_loopback_hostname_and_netloc():
    plan = _plan(allowed_domains=["127.0.0.1"])

    normalize_allowed_domains_for_browsergym(plan, env_id="browsergym/miniwob.click-button", current_url=None)

    assert "127.0.0.1" in plan.allowed_domains
    assert "127.0.0.1:8765" in plan.allowed_domains
    plan.steps[0].args["exact"] = True
    PlanValidator().validate(plan)


def test_allowed_domains_normalization_adds_localhost_hostname_and_netloc():
    plan = _plan(
        start_url="http://localhost:8765/miniwob/click-button.html",
        allowed_domains=["localhost"],
    )

    normalize_allowed_domains_for_browsergym(plan, env_id="browsergym/miniwob.click-button", current_url=None)

    assert "localhost" in plan.allowed_domains
    assert "localhost:8765" in plan.allowed_domains
    plan.steps[0].args["exact"] = True
    PlanValidator().validate(plan)


def test_allowed_domains_normalization_is_minwob_only():
    plan = _plan(start_url="http://127.0.0.1:8765/app", allowed_domains=["127.0.0.1"])

    normalize_allowed_domains_for_browsergym(plan, env_id="browsergym/openended", current_url=None)

    assert plan.allowed_domains == ["127.0.0.1"]
    plan.steps[0].args["exact"] = True
    with pytest.raises(PlanValidationError, match="start_url domain is not allowed"):
        PlanValidator().validate(plan)
