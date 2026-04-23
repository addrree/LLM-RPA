from app.benchmark.policies import build_benchmark_context
from app.orchestrator.workflow_manager import normalize_benchmark_plan
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator


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


def test_required_fields_are_always_taken_from_benchmark_context():
    plan = _base_plan(
        ["wrong_field"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
            {"step_id": 3, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="single_value_extraction",
        task_family="single_value_extraction",
        required_top_level_fields=["value"],
    )

    normalized = normalize_benchmark_plan(plan, ctx)

    assert normalized.expected_result.required_fields == ["value"]


def test_extraction_steps_without_save_as_get_required_field_fallback():
    plan = _base_plan(
        ["noise"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "extract_text", "args": {"selector": ".price"}},
            {"step_id": 3, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="navigation_then_extraction",
        task_family="navigation_then_extraction",
        required_top_level_fields=["value"],
    )

    normalized = normalize_benchmark_plan(plan, ctx)
    extraction_step = next(step for step in normalized.steps if step.action == "extract_text")

    assert extraction_step.save_as == "value"


def test_technical_artifacts_do_not_break_benchmark_validation():
    plan = _base_plan(
        ["value"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
            {"step_id": 4, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="single_value_extraction",
        task_family="single_value_extraction",
        required_top_level_fields=["value"],
    )

    PlanValidator().validate(plan, allowed_actions=set(ctx["allowed_actions"] + ["observe_page"]), benchmark_context=ctx)


def test_compare_family_no_longer_relies_on_region_or_section_extractors_in_benchmark_mode():
    plan = _base_plan(
        ["combined_result"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {
                "step_id": 3,
                "action": "extract_structured_items",
                "args": {"pattern": "(A)\\s+(1)", "fields": {"name": 1, "value": 2}, "limit": 1},
                "save_as": "source_a",
            },
            {
                "step_id": 4,
                "action": "extract_structured_items",
                "args": {"pattern": "(B)\\s+(2)", "fields": {"name": 1, "value": 2}, "limit": 1},
                "save_as": "source_b",
            },
            {
                "step_id": 5,
                "action": "compare_structured_values",
                "args": {"left_key": "source_a", "right_key": "source_b"},
                "save_as": "combined_result",
            },
            {"step_id": 6, "action": "finish", "args": {}},
        ],
    )
    ctx = build_benchmark_context(
        category="multi_step_information_retrieval",
        task_family="multi_step_information_retrieval",
        required_top_level_fields=["combined_result"],
    )

    normalized = normalize_benchmark_plan(plan, ctx)

    actions = [step.action for step in normalized.steps]
    assert "extract_value_from_section" not in actions
    assert "extract_structured_items_from_region" not in actions
    PlanValidator().validate(plan=normalized, allowed_actions=set(ctx["allowed_actions"]), benchmark_context=ctx)


def test_navigation_and_anchored_families_preserve_final_value_output():
    navigation_plan = _base_plan(
        ["value"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "click", "args": {"href_contains": "/docs"}},
            {"step_id": 3, "action": "wait_for", "args": {"url_contains": "/docs"}},
            {"step_id": 4, "action": "extract_text", "args": {"selector": "h1"}},
            {"step_id": 5, "action": "finish", "args": {}},
        ],
    )
    navigation_ctx = build_benchmark_context(
        category="navigation_then_extraction",
        task_family="navigation_then_extraction",
        required_top_level_fields=["value"],
    )

    anchored_plan = _base_plan(
        ["value"],
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {
                "step_id": 2,
                "action": "extract_value_near_anchor",
                "args": {"anchor_candidates": ["Contact"], "value_type": "email", "page_language": "en"},
            },
            {"step_id": 3, "action": "finish", "args": {}},
        ],
    )
    anchored_ctx = build_benchmark_context(
        category="anchored_value_extraction",
        task_family="anchored_value_extraction",
        required_top_level_fields=["value"],
    )

    normalized_navigation = normalize_benchmark_plan(navigation_plan, navigation_ctx)
    normalized_anchored = normalize_benchmark_plan(anchored_plan, anchored_ctx)

    nav_extract = next(step for step in normalized_navigation.steps if step.action == "extract_text")
    anchored_extract = next(step for step in normalized_anchored.steps if step.action == "extract_value_near_anchor")

    assert nav_extract.save_as == "value"
    assert anchored_extract.save_as == "value"
    assert "page_language" not in anchored_extract.args
