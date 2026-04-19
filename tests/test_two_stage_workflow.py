import asyncio
from datetime import datetime, timezone

from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.replanner import Replanner
from app.schemas.execution import ExecutionResult
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError


def _plan_without_open_url():
    return TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {
                    "step_id": 1,
                    "action": "extract_pattern_from_page_text",
                    "args": {"pattern": r"English\s+([0-9][0-9\s,\.\u00A0\u202F\+]*)"},
                    "save_as": "count",
                },
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )


def test_ensure_open_url_for_final_plan_injects_step():
    normalized = WorkflowManager._ensure_open_url_for_final_plan(_plan_without_open_url())

    assert normalized.steps[0].action == "open_url"
    assert normalized.steps[0].args["url"] == "https://www.wikipedia.org/"
    assert [step.step_id for step in normalized.steps] == [1, 2, 3]


def test_ensure_open_url_for_final_plan_keeps_existing_open_url():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )

    normalized = WorkflowManager._ensure_open_url_for_final_plan(plan)
    assert len(normalized.steps) == 2
    assert normalized.steps[0].action == "open_url"


def test_normalize_final_plan_fills_required_shape_from_context():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://www.wikipedia.org/",
        title="Wikipedia",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Wikipedia",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url"},
                {"action": "extract_value_near_anchor", "args": {"anchor_text": "English"}, "save_as": "count"},
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["count"]},
        },
        user_goal="Extract English count",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
    )
    plan = TaskSpec.model_validate(normalized)

    assert plan.goal == "Extract count"
    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.constraints.max_steps == 5
    assert plan.expected_result.description == "Count"
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org/"
    assert plan.steps[1].args == {"anchor_text": "English"}
    assert [step.step_id for step in plan.steps] == [1, 2, 3]


def test_normalize_final_plan_maps_structured_nested_required_fields_to_save_as():
    snapshot = PageSnapshot(
        url="https://www.wikipedia.org/",
        title="Wikipedia",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Wikipedia",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "goal": "Top languages",
            "start_url": "https://www.wikipedia.org/",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {"description": "Top language blocks", "required_fields": ["language_name", "article_count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {
                    "step_id": 2,
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": r"([A-Za-zА-Яа-яЁё]+)\\s+([0-9][0-9\\s,\\.\\u00A0\\u202F\\+]*)",
                        "limit": 10,
                        "fields": {
                            "language_name": {"group_index": 1},
                            "article_count": {"group_index": 2, "normalize_number": True, "number_type": "int"},
                        },
                    },
                    "save_as": "language_blocks",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        },
        user_goal="Extract top 10 languages",
        previous_plan=None,
        page_snapshot=snapshot,
    )
    plan = TaskSpec.model_validate(normalized)

    assert plan.expected_result.required_fields == ["language_blocks"]


class _FakePlanner:
    def build_initial_plan(self, user_goal: str) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                "expected_result": {"description": "Observe page", "required_fields": ["page_snapshot"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )

    last_artifact = None
    last_initial_artifact = None


class _FakeValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, plan: TaskSpec) -> None:
        self.calls += 1
        if self.calls == 2:
            raise PlanValidationError("extract_value_near_anchor requires non-empty 'value_pattern'")


class _RecordingValidator:
    def __init__(self):
        self.allowed_actions_history: list[set[str] | None] = []

    def validate(self, plan: TaskSpec, allowed_actions: set[str] | None = None) -> None:
        self.allowed_actions_history.append(allowed_actions)


class _FakeExecutor:
    async def _start_session(self):
        return {"id": "session"}

    async def _close_session(self, session):
        return None

    async def execute(self, plan: TaskSpec, session=None, runtime_state=None) -> ExecutionResult:
        if any(step.action == "observe_page" for step in plan.steps):
            return ExecutionResult(
                status="success",
                extracted_data={
                    "page_snapshot": {
                        "url": "https://example.com",
                        "title": "Example",
                        "screenshot_path": "artifacts/screenshots/s.png",
                        "page_text_excerpt": "Example Domain",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                },
                logs=[],
            )
        return ExecutionResult(status="success", extracted_data={"answer": "42"}, logs=[])


class _FakeVerifier:
    last_artifact = None

    class _Verdict:
        verdict = "accept"
        confidence = 1.0
        task_completed = True

        def model_dump(self, mode="json"):
            return {"verdict": "accept", "confidence": 1.0, "task_completed": True}

    def verify(self, plan, execution):
        return self._Verdict()


class _SingleStagePlanner:
    last_artifact = None
    last_initial_artifact = None
    last_action_oov_detected = False

    def build_plan(self, user_goal: str, benchmark_context=None) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
                "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                    {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                    {"step_id": 4, "action": "finish", "args": {}},
                ],
            }
        )


class _FakeReplanner:
    def __init__(self):
        self.calls = 0
        self.last_artifact = None

    def revise_plan(self, user_goal, page_snapshot, previous_plan=None, validation_error=None, invalid_plan=None):
        self.calls += 1
        if self.calls == 1:
            return TaskSpec.model_validate(
                {
                    "goal": user_goal,
                    "start_url": "https://example.com",
                    "allowed_domains": ["example.com"],
                    "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                    "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                    "steps": [
                        {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                        {
                            "step_id": 2,
                            "action": "extract_value_near_anchor",
                            "args": {"anchor_text": "Users"},
                            "save_as": "value",
                        },
                        {"step_id": 3, "action": "finish", "args": {}},
                    ],
                }
            )
        assert validation_error is not None
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {
                        "step_id": 2,
                        "action": "extract_value_near_anchor",
                        "args": {"anchor_text": "Users", "value_pattern": r"(\\d+)"},
                        "save_as": "value",
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


def test_two_stage_workflow_retries_invalid_final_plan_once():
    manager = WorkflowManager(
        planner=_FakePlanner(),
        validator=_FakeValidator(),
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=_FakeReplanner(),
        two_stage_planning=True,
    )

    result = asyncio.run(manager.run("Extract users count"))

    assert result["execution_result"].status == "success"
    assert result["final_plan"] is not None
    assert result["final_plan"].steps[1].args["value_pattern"] == r"(\\d+)"


def test_two_stage_initial_plan_uses_observation_actions_in_benchmark_mode():
    validator = _RecordingValidator()
    manager = WorkflowManager(
        planner=_FakePlanner(),
        validator=validator,
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=_FakeReplanner(),
        two_stage_planning=True,
    )

    result = asyncio.run(
        manager.run(
            "Extract users count",
            benchmark_context={
                "task_family": "single_value_extraction",
                "allowed_actions": ["open_url", "extract_text", "extract_pattern_from_page_text", "finish"],
            },
        )
    )

    assert result["execution_result"].status == "success"
    assert validator.allowed_actions_history[0] == {"open_url", "observe_page", "finish"}
    assert validator.allowed_actions_history[1] == {"open_url", "extract_text", "extract_pattern_from_page_text", "finish"}


def test_single_stage_benchmark_run_does_not_raise_normalization_attribute_error():
    manager = WorkflowManager(
        planner=_SingleStagePlanner(),
        validator=_RecordingValidator(),
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=None,
        two_stage_planning=False,
    )

    result = asyncio.run(
        manager.run(
            "Extract users count",
            benchmark_context={
                "task_family": "single_value_extraction",
                "allowed_actions": ["open_url", "extract_text", "extract_pattern_from_page_text", "finish"],
            },
        )
    )

    assert result["execution_result"].status == "success"
    assert result["plan"] is not None
    assert [step.action for step in result["plan"].steps] == ["open_url", "extract_text", "finish"]
