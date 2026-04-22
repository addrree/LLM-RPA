from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.benchmark.policies import BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY, build_benchmark_context
from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult, BenchmarkSelection
from app.benchmark.scenario_loader import BenchmarkScenario, load_scenario_suite
from app.orchestrator.workflow_manager import normalize_benchmark_plan
from app.planner.action_vocab import normalize_plan_action_aliases
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator


def test_scenario_suite_contains_required_categories():
    suite = load_scenario_suite(Path("benchmarks/scenarios/core_task_suite.json"))
    categories = {scenario.category for scenario in suite.scenarios}
    assert categories == {
        "single_value_extraction",
        "anchored_value_extraction",
        "repeated_structured_items",
        "navigation_then_extraction",
        "multi_step_information_retrieval",
        "negative_or_ambiguous_case",
    }
    assert all(scenario.start_url for scenario in suite.scenarios)
    assert all(isinstance(scenario.preconditions, list) for scenario in suite.scenarios)
    assert all(isinstance(scenario.page_expectations, list) for scenario in suite.scenarios)
    assert all(scenario.task_family for scenario in suite.scenarios)
    assert all(isinstance(scenario.anchor_candidates, list) for scenario in suite.scenarios)


def test_benchmark_selection_filters_by_id_and_category():
    suite = load_scenario_suite(Path("benchmarks/scenarios/core_task_suite.json"))

    filtered = BenchmarkRunner._filter_scenarios(
        suite.scenarios,
        BenchmarkSelection(
            scenario_ids=["repeated_listing_cards", "missing_or_ambiguous_field"],
            categories=["repeated_structured_items"],
        ),
    )

    assert [scenario.scenario_id for scenario in filtered] == ["repeated_listing_cards"]


def test_metrics_are_computed_from_scenario_results():
    results = [
        BenchmarkScenarioResult(
            scenario_id="s1",
            category="single_value_extraction",
            should_succeed=True,
            execution_status="success",
            verifier_verdict="accept",
            runtime_sec=1.2,
            corrective_retry_used=False,
            correction_attempt_count=0,
            corrective_plan_valid_count=0,
            corrective_plan_invalid_count=0,
            initial_plan_valid=True,
            final_plan_valid=True,
            export_success=True,
        ),
        BenchmarkScenarioResult(
            scenario_id="s2",
            category="negative_or_ambiguous_case",
            should_succeed=False,
            execution_status="success",
            verifier_verdict="reject",
            runtime_sec=2.4,
            corrective_retry_used=True,
            correction_attempt_count=1,
            corrective_plan_valid_count=1,
            corrective_plan_invalid_count=0,
            initial_plan_valid=True,
            final_plan_valid=True,
            export_success=False,
        ),
    ]

    metrics = BenchmarkRunner._compute_metrics(results)

    assert metrics.total_scenarios == 2
    assert metrics.positive_execution_success_rate == 1.0
    assert metrics.positive_verifier_accept_rate == 1.0
    assert metrics.negative_expected_reject_rate == 1.0
    assert metrics.plan_validation_pass_rate == 1.0
    assert metrics.correction_recovery_rate == 1.0
    assert metrics.corrective_plan_valid_count == 1
    assert metrics.corrective_plan_invalid_count == 0
    assert metrics.export_success_rate == 0.5
    assert metrics.mean_runtime_sec == 1.8


def test_negative_expected_reject_ignores_technical_failures():
    results = [
        BenchmarkScenarioResult(
            scenario_id="neg_ok",
            category="negative_or_ambiguous_case",
            should_succeed=False,
            execution_status="success",
            verifier_verdict="reject",
            runtime_sec=1.0,
            corrective_retry_used=False,
            correction_attempt_count=0,
            export_success=True,
        ),
        BenchmarkScenarioResult(
            scenario_id="neg_tech",
            category="negative_or_ambiguous_case",
            should_succeed=False,
            execution_status="failed",
            verifier_verdict="reject",
            runtime_sec=1.0,
            corrective_retry_used=False,
            correction_attempt_count=0,
            export_success=False,
        ),
    ]
    metrics = BenchmarkRunner._compute_metrics(results)
    assert metrics.negative_expected_reject_rate == 1.0


def test_negative_outcome_classification_distinguishes_expected_and_unexpected():
    expected = BenchmarkScenarioResult(
        scenario_id="neg_expected",
        category="negative_or_ambiguous_case",
        should_succeed=False,
        execution_status="success",
        verifier_verdict="reject",
        runtime_sec=1.0,
        corrective_retry_used=False,
        correction_attempt_count=0,
        export_success=True,
    )
    unexpected_accept = BenchmarkScenarioResult(
        scenario_id="neg_unexpected_accept",
        category="negative_or_ambiguous_case",
        should_succeed=False,
        execution_status="success",
        verifier_verdict="accept",
        runtime_sec=1.0,
        corrective_retry_used=False,
        correction_attempt_count=0,
        export_success=True,
    )
    technical = BenchmarkScenarioResult(
        scenario_id="neg_technical",
        category="negative_or_ambiguous_case",
        should_succeed=False,
        execution_status="failed",
        verifier_verdict="reject",
        runtime_sec=1.0,
        corrective_retry_used=False,
        correction_attempt_count=0,
        technical_failure=True,
        export_success=False,
    )

    assert BenchmarkRunner._classify_negative_outcome(expected) == "expected_reject"
    assert BenchmarkRunner._classify_negative_outcome(unexpected_accept) == "unexpected_accept"
    assert BenchmarkRunner._classify_negative_outcome(technical) == "technical_failure"


def test_action_alias_normalization_marks_legacy_click_alias_as_oov():
    payload = {
        "steps": [
            {"action": "click_element", "args": {"selector": "#go"}},
            {"action": "extract_value_near_anchor", "args": {"anchor_text": "Users", "value_type": "number"}},
            {"action": "finish", "args": {}},
        ]
    }
    normalized, oov_detected = normalize_plan_action_aliases(payload)
    assert oov_detected is True
    assert normalized["steps"][0]["action"] == "click_element"
    assert normalized["steps"][1]["action"] == "extract_value_near_anchor"


def test_negative_outcome_classification_respects_technical_failure_flag():
    technical = BenchmarkScenarioResult(
        scenario_id="neg_browser",
        category="negative_or_ambiguous_case",
        should_succeed=False,
        execution_status="success",
        verifier_verdict="reject",
        runtime_sec=1.0,
        corrective_retry_used=False,
        correction_attempt_count=0,
        technical_failure=True,
        export_success=True,
    )
    assert BenchmarkRunner._classify_negative_outcome(technical) == "technical_failure"


def test_failure_stage_inference_for_negative_semantic_reject():
    stage = BenchmarkRunner._infer_failure_stage(
        should_succeed=False,
        execution_status="success",
        verifier_verdict="reject",
        initial_plan_valid=True,
        final_plan_valid=True,
        export_success=True,
    )
    assert stage is None


def test_plan_validation_pass_rate_respects_validation_failures():
    results = [
        BenchmarkScenarioResult(
            scenario_id="ok",
            category="single_value_extraction",
            should_succeed=True,
            execution_status="success",
            verifier_verdict="accept",
            runtime_sec=0.5,
            corrective_retry_used=False,
            correction_attempt_count=0,
            corrective_plan_valid_count=0,
            corrective_plan_invalid_count=0,
            initial_plan_valid=True,
            final_plan_valid=True,
            failure_stage=None,
            export_success=True,
        ),
        BenchmarkScenarioResult(
            scenario_id="bad_validation",
            category="single_value_extraction",
            should_succeed=True,
            execution_status="failed",
            verifier_verdict="error",
            runtime_sec=0.4,
            corrective_retry_used=False,
            correction_attempt_count=0,
            corrective_plan_valid_count=0,
            corrective_plan_invalid_count=1,
            initial_plan_valid=None,
            final_plan_valid=None,
            failure_stage="validation",
            export_success=False,
        ),
    ]

    metrics = BenchmarkRunner._compute_metrics(results)
    assert metrics.plan_validation_pass_rate == 0.5


def test_smoke_suite_contains_three_core_categories():
    suite = load_scenario_suite(Path("benchmarks/scenarios/smoke_generalized_suite.json"))
    categories = {scenario.category for scenario in suite.scenarios}
    assert categories == {
        "single_value_extraction",
        "anchored_value_extraction",
        "repeated_structured_items",
    }


def test_grounded_goal_uses_auto_language_detection_when_language_unknown():
    scenario = BenchmarkScenario.model_validate(
        {
            "scenario_id": "s",
            "goal": "Open site and extract value",
            "start_url": "https://example.com",
            "category": "single_value_extraction",
            "description": "d",
            "expected_output_type": "scalar",
            "page_language": "auto",
        }
    )
    goal = BenchmarkRunner._build_grounded_goal(scenario)
    assert "Page language hint:" not in goal
    assert "Page language is unknown before navigation" in goal


def test_benchmark_allowed_actions_are_category_specific():
    assert BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["single_value_extraction"] == [
        "open_url",
        "extract_text",
        "extract_pattern_from_page_text",
        "finish",
    ]
    assert "click" not in BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["single_value_extraction"]
    assert "compare_structured_values" in BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["multi_step_information_retrieval"]


def test_plan_validator_rejects_actions_outside_benchmark_policy():
    plan = TaskSpec.model_validate(
        {
            "goal": "benchmark",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"text": "Docs", "exact": True}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    ctx = build_benchmark_context(category="single_value_extraction", task_family="single_value_extraction")
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_single_value_rewrites_brittle_literal_pattern_to_h1():
    plan = TaskSpec.model_validate(
        {
            "goal": "extract header",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_pattern_from_page_text",
                    "args": {"pattern": r"Welcome to Python\\.org"},
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="single_value_extraction", task_family="single_value_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    extraction = normalized.steps[1]
    assert extraction.action == "extract_text"
    assert extraction.args["selector"] == "h1"


def test_normalize_benchmark_plan_single_value_forces_save_as_value_even_for_alias():
    plan = TaskSpec.model_validate(
        {
            "goal": "extract header",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "heading"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="single_value_extraction", task_family="single_value_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    extraction = normalized.steps[1]
    assert extraction.action == "extract_text"
    assert extraction.save_as == "value"


def test_normalize_benchmark_plan_adds_guardrail_for_navigation_bare_text_click():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"text": "Pricing"}},
                {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_adds_guardrail_for_regex_only_multi_step_compare():
    plan = TaskSpec.model_validate(
        {
            "goal": "compare two sections",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["structured_comparison"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_pattern_from_page_text",
                    "args": {"pattern": r"(\\d{4}-\\d{2}-\\d{2}.+)"},
                    "save_as": "section_a_data",
                },
                {
                    "step_id": 3,
                    "action": "compare_structured_values",
                    "args": {"left_key": "section_a_data", "right_key": "section_b_data"},
                    "save_as": "structured_comparison",
                },
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(
        category="multi_step_information_retrieval",
        task_family="multi_step_information_retrieval",
    )
    normalized = normalize_benchmark_plan(plan, ctx)
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_anchored_uses_scenario_anchor_candidates_as_source_of_truth():
    plan = TaskSpec.model_validate(
        {
            "goal": "extract support email",
            "start_url": "https://pypi.org/help/",
            "allowed_domains": ["pypi.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://pypi.org/help/"}},
                {
                    "step_id": 2,
                    "action": "extract_value_near_anchor",
                    "args": {
                        "anchor_text": "Контактная информация",
                        "anchor_candidates": ["Контакт", "Поддержка"],
                        "page_language": "ru",
                        "value_type": "email",
                    },
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(
        category="anchored_value_extraction",
        task_family="anchored_value_extraction",
        scenario_anchor_candidates=["Email", "Contact", "Support"],
        scenario_anchor_matching_mode="auto",
        scenario_page_language="en",
    )
    normalized = normalize_benchmark_plan(plan, ctx)
    anchored_step = next(step for step in normalized.steps if step.action == "extract_value_near_anchor")
    assert anchored_step.args["anchor_candidates"] == ["Email", "Contact", "Support"]
    assert "anchor_text" not in anchored_step.args
    assert "page_language" not in anchored_step.args
    assert normalized.expected_result.required_fields == ["value"]


def test_normalize_benchmark_plan_adds_guardrail_for_navigation_weak_wait_for_text():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://www.python.org",
            "allowed_domains": ["www.python.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.python.org"}},
                {"step_id": 2, "action": "wait_for", "args": {"text": "Python"}},
                {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_promotes_navigation_wait_to_url_contains_from_href_contains_click():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"href_contains": "/pricing"}},
                {"step_id": 3, "action": "wait_for", "args": {"text": "Pricing"}},
                {"step_id": 4, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 5, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    wait_step = next(step for step in normalized.steps if step.action == "wait_for")
    assert wait_step.args["url_contains"] == "/pricing"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_promotes_navigation_wait_to_main_selector_for_role_name_click():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"role": "link", "name": "Pricing"}},
                {"step_id": 3, "action": "wait_for", "args": {"text": "Pricing"}},
                {"step_id": 4, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 5, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    wait_step = next(step for step in normalized.steps if step.action == "wait_for")
    assert wait_step.args["selector"] == "main h1, article h1, [role='main'] h1, main, article, [role='main']"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_rejects_overconstrained_navigation_text_click_without_snapshot_evidence():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "click",
                    "args": {"scope_selector": "main", "text": "Tutorial", "exact": True},
                },
                {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://example.com",
        title="Example",
        screenshot_path="artifacts/screenshots/x.png",
        page_text_excerpt="Welcome to docs",
        visible_headings=["Home"],
        visible_labels=[],
        visible_buttons=[],
        visible_inputs=[],
        timestamp=datetime.now(timezone.utc),
        page_text="Pricing and docs overview",
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx, page_snapshot=snapshot)
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))
