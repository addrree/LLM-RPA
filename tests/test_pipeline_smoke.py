from app.planner.planner import Planner
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import DummyLLMClient
from app.validator.plan_validator import PlanValidationError, PlanValidator
from app.verifier.llm_verifier import LLMVerifier


def test_dummy_planner_uses_url_from_goal():
    llm = DummyLLMClient()
    planner = Planner(llm)

    plan = planner.build_plan("Open https://www.wikipedia.org and extract h1")

    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org"


def test_planner_validator_verifier_smoke_success():
    llm = DummyLLMClient()

    planner = Planner(llm)
    plan = planner.build_plan("Open https://example.com and extract h1")

    validator = PlanValidator()
    validator.validate(plan)

    result = ExecutionResult(
        status="success",
        extracted_data={"heading": "Example Domain"},
        final_url="https://example.com",
        page_title="Example Domain",
        page_text_excerpt="Example Domain This domain is for use in illustrative examples",
        screenshot_path="artifacts/screenshots/step_3.png",
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, result)

    assert verdict.verdict == "accept"
    assert verdict.task_completed is True


def test_dummy_verifier_rejects_failed_execution():
    llm = DummyLLMClient()

    planner = Planner(llm)
    plan = planner.build_plan("Open https://example.com and extract h1")

    failed_result = ExecutionResult(
        status="failed",
        extracted_data={},
        logs=[StepLog(step_id=1, action="open_url", status="failed", message="DNS error")],
        error_message="DNS error",
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, failed_result)

    assert verdict.verdict == "reject"
    assert verdict.task_completed is False


def test_validator_requires_save_as_for_extract_items():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract products",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Extract products", "required_fields": ["products"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_items",
                    "args": {
                        "container_selector": ".card",
                        "limit": 10,
                        "fields": {"title": ".title", "link": {"selector": "a", "attr": "href"}},
                    },
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    try:
        validator.validate(plan)
        raise AssertionError("Expected validator to reject extract_items without save_as")
    except PlanValidationError as exc:
        assert "save_as" in str(exc)
