import pytest

from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.action_vocab import PlannerValidationFailed, normalize_plan_action_aliases
from app.planner.planner import Planner
from app.planner.prompts import INITIAL_PLANNER_SYSTEM_PROMPT
from app.schemas.execution import GenerationMetadata, LLMArtifact
from app.schemas.task_spec import TaskSpec


def test_normalize_initial_plan_to_full_taskspec_shape():
    raw_plan = {
        "steps": [
            {"action": "open_url", "url": "https://www.wikipedia.org/catalog"},
            {"action": "observe_page", "save_as": "page_snapshot"},
        ],
        "expected_result": {"required_fields": ["page_snapshot"]},
    }

    normalized = Planner._normalize_initial_plan(raw_plan, "Collect product cards")
    plan = TaskSpec.model_validate(normalized)

    assert plan.goal == "Collect product cards"
    assert str(plan.start_url) == "https://www.wikipedia.org/catalog"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.expected_result.description == "Collect page snapshot for replanning"
    assert [step.step_id for step in plan.steps] == [1, 2, 3]
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org/catalog"
    assert plan.steps[-1].action == "finish"


def test_normalize_initial_plan_fills_observe_page_save_as_and_required_fields():
    raw_plan = {
        "steps": [
            {"action": "open_url", "url": "https://www.wikipedia.org"},
            {"action": "observe_page"},
            {"action": "finish"},
        ]
    }

    normalized = Planner._normalize_initial_plan(raw_plan, "Observe wikipedia landing page")
    plan = TaskSpec.model_validate(normalized)

    assert plan.steps[0].args["url"] == "https://www.wikipedia.org"
    assert plan.steps[1].action == "observe_page"
    assert plan.steps[1].save_as == "page_snapshot"
    assert [step.step_id for step in plan.steps] == [1, 2, 3]
    assert "page_snapshot" in plan.expected_result.required_fields
    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.constraints.max_steps == 4
    assert plan.constraints.max_replans == 1
    assert plan.constraints.timeout_sec == 30
    assert plan.goal == "Observe wikipedia landing page"


def test_initial_fallback_normalizes_domain_without_scheme_from_model_payload():
    raw_plan = {"start_url": "www.wikipedia.org"}

    fallback = Planner._build_initial_fallback(
        "Open Wikipedia and observe the landing page",
        candidate_payload=raw_plan,
    )
    plan = TaskSpec.model_validate(fallback)

    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org"


def test_normalize_initial_plan_normalizes_open_url_domain_without_scheme():
    raw_plan = {
        "steps": [
            {"action": "open_url", "args": {"url": "wikipedia.org"}},
            {"action": "observe_page"},
            {"action": "finish"},
        ]
    }

    normalized = Planner._normalize_initial_plan(raw_plan, "Observe Wikipedia")
    plan = TaskSpec.model_validate(normalized)

    assert str(plan.start_url) == "https://wikipedia.org/"
    assert plan.steps[0].args["url"] == "https://wikipedia.org"


def test_initial_prompt_requires_canonical_url_inference_without_placeholders():
    prompt = INITIAL_PLANNER_SYSTEM_PROMPT.lower()

    assert "canonical public https homepage url" in prompt
    assert "placeholder" in prompt
    assert "dummy urls" in prompt


def test_initial_shape_requires_url_not_only_action_order():
    payload = {
        "steps": [
            {"step_id": 1, "action": "open_url", "args": {}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "finish", "args": {}},
        ]
    }

    assert Planner._is_valid_initial_shape(payload) is False


def test_initial_shape_rejects_invalid_leading_dot_hostname():
    payload = {
        "start_url": "https://.wikipedia.org",
        "steps": [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://.wikipedia.org"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "finish", "args": {}},
        ],
    }

    assert Planner._is_valid_initial_shape(payload) is False
    assert Planner._extract_first_url(payload) == ""


def test_action_normalization_strips_model_whitespace():
    normalized, action_oov_detected = normalize_plan_action_aliases(
        {
            "steps": [
                {"step_id": 1, "action": " observe_page"},
                {"step_id": 2, "action": " fill_input ", "args": {"target": "Search", "text": "ssd"}},
            ]
        }
    )

    assert action_oov_detected is False
    assert [step["action"] for step in normalized["steps"]] == [
        "observe_page",
        "fill_by_semantic_target",
    ]


def test_workflow_validation_normalizer_repairs_step_ids():
    plan = TaskSpec.model_validate(
        {
            "goal": "Observe Wikipedia",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["www.wikipedia.org"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "Observe page", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 10, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"step_id": 10, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 99, "action": "finish", "args": {}},
            ],
        }
    )

    normalized = WorkflowManager._normalize_plan_for_validation(plan)

    assert [step.step_id for step in normalized.steps] == [1, 2, 3]


def test_initial_planner_retries_generation_error_before_fallback():
    class RetryInitialClient:
        def __init__(self):
            self.calls = 0

        def generate_planner_artifact(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("JSON response must be an object.")
            plan = {
                "goal": "Open Wikipedia",
                "start_url": "https://www.wikipedia.org",
                "allowed_domains": ["www.wikipedia.org"],
                "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
                "expected_result": {"description": "Observe page", "required_fields": ["page_snapshot"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                    {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
            return LLMArtifact(
                raw_response="{}",
                parsed_response=plan,
                generation=GenerationMetadata(backend="test", model="fake", source="llm"),
            )

    client = RetryInitialClient()
    plan = Planner(client).build_initial_plan("Open Wikipedia")

    assert client.calls == 2
    assert str(plan.start_url) == "https://www.wikipedia.org/"


def test_initial_repair_generation_error_returns_controlled_validation_failure():
    class BrokenRepairClient:
        def __init__(self):
            self.stages = []

        def generate_planner_artifact(self, **kwargs):
            stage = kwargs["stage"]
            self.stages.append(stage)
            if stage == "initial_planner_repair":
                raise ConnectionError("remote end closed connection")
            return LLMArtifact(
                raw_response="{}",
                parsed_response={
                    "steps": [
                        {"step_id": 1, "action": "open_url", "args": {}},
                        {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                        {"step_id": 3, "action": "finish", "args": {}},
                    ]
                },
                generation=GenerationMetadata(backend="test", model="fake", source="llm"),
            )

    with pytest.raises(PlannerValidationFailed) as exc_info:
        Planner(BrokenRepairClient()).build_initial_plan("Open Wikipedia and observe the landing page")

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["reason"] == "missing_start_url"
    assert "remote end closed connection" in diagnostics["repair_error"]
