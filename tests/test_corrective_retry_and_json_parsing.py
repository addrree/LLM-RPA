import asyncio

from app.orchestrator.workflow_manager import WorkflowManager
from app.schemas.execution import ExecutionResult
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient, LLMClientError


class _StubPlanner:
    last_artifact = None
    last_initial_artifact = None

    def build_plan(self, user_goal: str) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
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
