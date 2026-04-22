import pytest
from app.benchmark.contract import required_contract_fields
from app.benchmark.policies import build_benchmark_context
from app.orchestrator.workflow_manager import normalize_benchmark_plan
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator


def _base_plan(required_fields: list[str], steps: list[dict]) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "benchmark task",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 8, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": required_fields},
            "steps": steps,
        }
    )


def test_required_contract_fields_map_by_task_family():
    assert required_contract_fields(task_family="single_value_extraction") == ["value"]
    assert required_contract_fields(task_family="anchored_value_extraction") == ["anchor", "value"]
    assert required_contract_fields(task_family="repeated_structured_items") == ["items"]
    assert required_contract_fields(task_family="navigation_then_extraction") == ["source_page", "target_page", "value"]
    assert required_contract_fields(task_family="multi_step_information_retrieval") == ["source_a", "source_b", "combined_result"]


def test_navigation_contract_normalizer_enforces_required_top_level_shape():
    plan = _base_plan(
        ["value"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "click", "args": {"href_contains": "/pricing"}},
            {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "heading"},
            {"step_id": 4, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="navigation_then_extraction",
        task_family="navigation_then_extraction",
        required_top_level_fields=["source_page", "target_page", "value"],
    )

    normalized = normalize_benchmark_plan(plan, ctx)

    assert normalized.expected_result.required_fields == ["source_page", "target_page", "value"]
    assert [step.save_as for step in normalized.steps if step.action == "observe_page"] == ["source_page", "target_page"]
    assert any(step.action.startswith("extract") and step.save_as == "value" for step in normalized.steps)


def test_multi_step_contract_normalizer_rewrites_to_combined_result_pipeline():
    plan = _base_plan(
        ["structured_comparison"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "extract_value_from_section", "args": {"section_selector": "#a", "pattern": "(.*)"}, "save_as": "a"},
            {"step_id": 3, "action": "extract_value_from_section", "args": {"section_selector": "#b", "pattern": "(.*)"}, "save_as": "b"},
            {"step_id": 4, "action": "compare_structured_values", "args": {}, "save_as": "structured_comparison"},
            {"step_id": 5, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="multi_step_information_retrieval",
        task_family="multi_step_information_retrieval",
        required_top_level_fields=["source_a", "source_b", "combined_result"],
    )

    normalized = normalize_benchmark_plan(plan, ctx)
    compare_step = next(step for step in normalized.steps if step.action == "compare_structured_values")

    assert normalized.expected_result.required_fields == ["source_a", "source_b", "combined_result"]
    assert compare_step.save_as == "combined_result"
    assert compare_step.args["left_key"] == "source_a"
    assert compare_step.args["right_key"] == "source_b"


def test_validator_rejects_plan_that_violates_family_contract_shape():
    plan = _base_plan(
        ["value"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
            {"step_id": 3, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="navigation_then_extraction",
        task_family="navigation_then_extraction",
        required_top_level_fields=["source_page", "target_page", "value"],
    )

    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan, allowed_actions=set(ctx["allowed_actions"]), benchmark_context=ctx)
