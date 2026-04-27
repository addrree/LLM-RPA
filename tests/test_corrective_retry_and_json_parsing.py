import asyncio
import json

from app.orchestrator.workflow_manager import WorkflowManager
from app.schemas.execution import ExecutionResult
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient, LLMClientError
from app.validator.plan_validator import PlanValidationError


class _StubPlanner:
    last_artifact = None
    last_initial_artifact = None

    def build_plan(self, user_goal: str) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
                "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


class _StubValidator:
    def validate(self, plan: TaskSpec) -> None:
        assert plan.steps[0].action == "open_url"
        assert plan.steps[0].args.get("url")


class _StrictValidator(_StubValidator):
    def validate(self, plan: TaskSpec) -> None:
        super().validate(plan)
        for step in plan.steps:
            if step.action == "extract_items" and not step.args.get("container_selector"):
                raise PlanValidationError("extract_items missing required args: container_selector")
            if step.action == "click" and step.args.get("selector") == "a":
                raise PlanValidationError("click selector is too broad: 'a'")


class _StubExecutor:
    def __init__(self):
        self.calls = 0

    async def execute(self, plan: TaskSpec, session=None, runtime_state=None) -> ExecutionResult:
        self.calls += 1
        value = "bad" if self.calls == 1 else "good"
        return ExecutionResult(status="success", extracted_data={"value": value}, logs=[])


class _Verdict:
    def __init__(self, verdict: str):
        self.verdict = verdict
        self.confidence = 1.0 if verdict == "accept" else 0.1
        self.task_completed = verdict == "accept"
        self.issues = ["returned wrong structure"] if verdict != "accept" else []
        self.summary = "ok" if verdict == "accept" else "need correction"

    def model_dump(self, mode="json"):
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "task_completed": self.task_completed,
            "issues": self.issues,
            "summary": self.summary,
        }


class _StubVerifier:
    last_artifact = None

    def verify(self, plan, execution):
        return _Verdict("accept" if execution.extracted_data.get("value") == "good" else "reject")


class _StubReplanner:
    def __init__(self):
        self.last_artifact = None

    def build_corrective_plan(self, **kwargs):
        previous = kwargs["previous_plan"]
        return TaskSpec.model_validate(
            {
                **previous.model_dump(mode="json"),
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


class _InvalidCorrectiveReplanner:
    def __init__(self):
        self.last_artifact = None

    def build_corrective_plan(self, **kwargs):
        previous = kwargs["previous_plan"]
        return TaskSpec.model_validate(
            {
                **previous.model_dump(mode="json"),
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {
                        "step_id": 2,
                        "action": "extract_items",
                        "args": {"limit": 10, "fields": {"name": ".name"}},
                        "save_as": "items",
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


class _FlakyCorrectiveReplanner:
    def __init__(self):
        self.last_artifact = None
        self.calls = 0

    def build_corrective_plan(self, **kwargs):
        self.calls += 1
        previous = kwargs["previous_plan"]
        if self.calls == 1:
            return TaskSpec.model_validate(
                {
                    **previous.model_dump(mode="json"),
                    "steps": [
                        {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                        {"step_id": 2, "action": "click", "args": {"selector": "a"}},
                        {"step_id": 3, "action": "finish", "args": {}},
                    ],
                }
            )
        return TaskSpec.model_validate(
            {
                **previous.model_dump(mode="json"),
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


def test_llm_json_parser_reports_stage_and_position():
    try:
        LLMClient._safe_parse_json("```json\n{\n  \"a\": 1,,\n}\n```", stage="replanner")
        raise AssertionError("Expected parse failure")
    except LLMClientError as exc:
        msg = str(exc)
        assert "stage=replanner" in msg
        assert "line=" in msg and "col=" in msg and "pos=" in msg


def test_workflow_corrective_retry_after_verifier_reject_single_stage():
    manager = WorkflowManager(
        planner=_StubPlanner(),
        validator=_StubValidator(),
        executor=_StubExecutor(),
        verifier=_StubVerifier(),
        replanner=_StubReplanner(),
        two_stage_planning=False,
    )

    result = asyncio.run(manager.run("Extract value"))
    assert result["verdict"].verdict == "accept"
    assert result["execution_result"].extracted_data["value"] == "good"


def test_ensure_open_url_for_final_plan_fills_missing_args_url_from_start_url():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 3, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {}},
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )
    fixed = WorkflowManager._ensure_open_url_for_final_plan(plan)
    assert fixed.steps[0].args["url"] == "https://example.com/"


def test_normalize_plan_for_validation_fills_observe_page_save_as():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 3, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "observe_page", "args": {}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    fixed = WorkflowManager._normalize_plan_for_validation(plan)
    assert fixed.steps[1].save_as == "page_snapshot"


def test_identify_offending_step_for_extract_items_without_container_selector():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_items",
                    "args": {"limit": 5, "fields": {"title": ".title"}},
                    "save_as": "rows",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    offending = WorkflowManager._identify_offending_step(
        corrective_plan=plan,
        validation_error="extract_items missing required args: container_selector",
    )
    assert offending is not None
    assert offending["step_id"] == 2
    assert offending["action"] == "extract_items"


def test_workflow_stops_retry_and_persists_artifacts_when_corrective_plan_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr("app.orchestrator.workflow_manager.RAW_LLM_DIR", tmp_path)
    manager = WorkflowManager(
        planner=_StubPlanner(),
        validator=_StrictValidator(),
        executor=_StubExecutor(),
        verifier=_StubVerifier(),
        replanner=_InvalidCorrectiveReplanner(),
        two_stage_planning=False,
    )

    result = asyncio.run(manager.run("Extract list"))
    assert result["corrective_plan_invalid_count"] == 1

    candidate_files = sorted(tmp_path.glob("corrective_plan_candidate_attempt1_*.json"))
    failure_files = sorted(tmp_path.glob("corrective_plan_validation_failed_attempt1_*.json"))
    assert candidate_files
    assert failure_files

    failure_payload = json.loads(failure_files[-1].read_text(encoding="utf-8"))
    assert failure_payload["validation_error"] == "extract_items missing required args: container_selector"
    assert failure_payload["offending_step"]["action"] == "extract_items"


def test_failure_context_marks_anchor_not_found_as_recoverable():
    execution = ExecutionResult(
        status="failed",
        extracted_data={},
        logs=[],
        failure_type="execution_step_failed",
        failed_action="extract_value_near_anchor",
        failed_args={"anchor_text": "Поддержка"},
        error_message="Anchor text not found: Поддержка",
    )
    verdict = _Verdict("reject")
    ctx = WorkflowManager._build_failure_context(execution_result=execution, verdict=verdict)
    assert ctx["failure_type"] == "anchor_not_found"
    assert WorkflowManager._should_retry_corrective(
        failure_type=ctx["failure_type"],
        prior_corrective_attempts=[],
        max_retries=2,
        corrective_attempt_count=0,
    )


def test_failure_context_marks_bad_click_locator_as_recoverable():
    execution = ExecutionResult(
        status="failed",
        extracted_data={},
        logs=[],
        failure_type="browser_operation_failed",
        failed_action="click",
        failed_args={"text": "More"},
        error_message="Timeout 30000ms exceeded while waiting for locator",
    )
    verdict = _Verdict("reject")
    ctx = WorkflowManager._build_failure_context(execution_result=execution, verdict=verdict)
    assert ctx["failure_type"] == "bad_locator_choice"
    assert WorkflowManager._should_retry_corrective(
        failure_type=ctx["failure_type"],
        prior_corrective_attempts=[],
        max_retries=2,
        corrective_attempt_count=0,
    )


def test_failure_context_marks_broad_pattern_no_fallback_as_recoverable():
    execution = ExecutionResult(
        status="failed",
        extracted_data={},
        logs=[],
        failure_type="execution_step_failed",
        failed_action="extract_structured_items",
        failed_args={"pattern": "(.+)", "fields": {"value": 1}},
        error_message="broad_pattern_rejected_no_structured_fallback: Regex pattern is too broad",
    )
    verdict = _Verdict("reject")
    ctx = WorkflowManager._build_failure_context(execution_result=execution, verdict=verdict)
    assert ctx["failure_type"] == "broad_pattern_rejected_no_structured_fallback"
    assert WorkflowManager._should_retry_corrective(
        failure_type=ctx["failure_type"],
        prior_corrective_attempts=[],
        max_retries=2,
        corrective_attempt_count=0,
    )


def test_workflow_continues_after_invalid_corrective_and_recovers():
    manager = WorkflowManager(
        planner=_StubPlanner(),
        validator=_StrictValidator(),
        executor=_StubExecutor(),
        verifier=_StubVerifier(),
        replanner=_FlakyCorrectiveReplanner(),
        two_stage_planning=False,
    )
    result = asyncio.run(manager.run("Extract value"))
    assert result["verdict"].verdict == "accept"
    assert result["correction_attempt_count"] == 1
    assert result["corrective_plan_invalid_count"] == 1
    assert result["corrective_plan_valid_count"] == 0


def test_corrective_retry_stops_on_repeated_failure_class():
    attempts = [
        {"attempt": 1, "failure_type": "missing_required_field"},
        {"attempt": 2, "failure_type": "missing_required_field"},
    ]
    allowed = WorkflowManager._should_retry_corrective(
        failure_type="missing_required_field",
        prior_corrective_attempts=attempts,
        max_retries=3,
        corrective_attempt_count=2,
    )
    assert allowed is False


def test_multi_step_comparison_is_augmented_deterministically():
    execution = ExecutionResult(
        status="success",
        extracted_data={"section_a_data": {"x": 1}, "section_b_data": {"x": 2}},
        logs=[],
    )
    WorkflowManager._augment_multi_step_comparison(execution)
    assert "structured_comparison" in execution.extracted_data
    assert execution.extracted_data["section_a_data"] == {"x": 1}
    assert execution.extracted_data["section_b_data"] == {"x": 2}
    assert execution.extracted_data["combined_result"]["comparison"]["exact_match"] is False
    assert execution.extracted_data["structured_comparison"]["status"] == "different"


def test_corrective_retry_is_disabled_for_browser_failures():
    allowed = WorkflowManager._should_retry_corrective(
        failure_type="browser_operation_failed",
        prior_corrective_attempts=[],
        max_retries=3,
        corrective_attempt_count=0,
    )
    assert allowed is False


def test_effective_max_retries_keeps_corrective_loop_active():
    assert WorkflowManager._effective_max_retries(0) == 1
    assert WorkflowManager._effective_max_retries(1) == 1
    assert WorkflowManager._effective_max_retries(5) == 3
