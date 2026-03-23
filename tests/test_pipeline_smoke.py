from app.planner.planner import Planner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier
from app.utils.llm_client import DummyLLMClient
from app.schemas.execution import ExecutionResult


def test_planner_validator_verifier_smoke():
    llm = DummyLLMClient()

    planner = Planner(llm)
    plan = planner.build_plan("Open example.com and extract h1")

    validator = PlanValidator()
    validator.validate(plan)

    result = ExecutionResult(
        status="success",
        extracted_data={"heading": "Example Domain"},
        final_url="https://example.com",
        page_title="Example Domain",
        page_text_excerpt="Example Domain This domain is for use in illustrative examples",
        screenshot_path="artifacts/screenshots/step_3.png",
        logs=[],
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, result)

    assert verdict.verdict == "accept"
    assert verdict.task_completed is True
