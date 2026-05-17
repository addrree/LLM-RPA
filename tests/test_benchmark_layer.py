from datetime import datetime, timezone
from pathlib import Path
import asyncio

import pytest

from app.benchmark.policies import BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY, build_benchmark_context
from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult, BenchmarkSelection
from app.benchmark.scenario_loader import BenchmarkScenario, load_scenario_suite
from app.observer.page_observer import PageObserver
from app.orchestrator.workflow_manager import normalize_benchmark_plan
from app.planner.action_vocab import normalize_plan_action_aliases
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from tests.fakes import DummyLLMClient
from app.validator.plan_validator import PlanValidationError, PlanValidator
from app.verifier.llm_verifier import LLMVerifier


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


def test_grounded_goal_excludes_benchmark_metadata_labels():
    scenario = BenchmarkScenario.model_validate(
        {
            "scenario_id": "navigate_then_extract_docs_python",
            "goal": "Open the tutorial and extract heading",
            "start_url": "https://docs.python.org/3/",
            "category": "navigation_then_extraction",
            "task_family": "navigation_then_extraction",
            "description": "d",
            "expected_output_type": "scalar",
            "should_succeed": True,
            "notes": "internal note should stay hidden from planner prompt",
            "required_top_level_fields": ["value"],
        }
    )
    goal = BenchmarkRunner._build_grounded_goal(scenario, allowed_actions=["open_url", "click", "extract_text", "finish"])
    banned_tokens = [
        "Scenario ID",
        "Category",
        "Notes",
        "Required fields",
        "Should succeed",
        "Benchmark",
        "required_top_level_fields",
    ]
    assert all(token not in goal for token in banned_tokens)


def test_extended_suite_replaces_gnu_navigation_with_iana_protocols():
    suite = load_scenario_suite(Path("benchmarks/scenarios/extended_generalized_task_suite.json"))
    scenario_ids = {scenario.scenario_id for scenario in suite.scenarios}
    assert "navigate_then_extract_gnu" not in scenario_ids
    assert "navigate_then_extract_iana_protocols" in scenario_ids


def test_extended_suite_has_no_example_dot_com_urls():
    suite = load_scenario_suite(Path("benchmarks/scenarios/extended_generalized_task_suite.json"))
    chunks: list[str] = []
    for scenario in suite.scenarios:
        chunks.extend(
            [
                scenario.start_url,
                scenario.goal,
                scenario.target_page_hint,
                scenario.description,
                scenario.notes,
            ]
        )
    all_text = " ".join(chunks)
    assert "example.com" not in all_text.lower()


def test_negative_scenario_does_not_accept_unrelated_numeric_match():
    suite = load_scenario_suite(Path("benchmarks/scenarios/core_task_suite_v3.json"))
    scenario = next(item for item in suite.scenarios if item.scenario_id == "negative_ambiguous_on_wikipedia")
    assert scenario.should_succeed is False
    assert scenario.required_top_level_fields == []
    goal = scenario.goal.lower()
    assert "domain registration fee" in goal
    assert "billing contact email" in goal
    assert "article" not in goal


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
    assert "Infer anchors/headings/value patterns from observe_page" in goal


def test_benchmark_context_keeps_evaluator_metadata_private_from_prompt_contract():
    ctx = build_benchmark_context(
        category="repeated_structured_items",
        task_family="repeated_structured_items",
        scenario_id="repeated_listing_iana_rootdb",
        expected_min_items=3,
        expected_item_fields=["name", "detail"],
        required_top_level_fields=["items"],
    )
    assert "scenario_anchor_candidates" not in ctx
    assert "scenario_page_language" not in ctx
    assert ctx["required_top_level_fields"] == ["items"]
    assert ctx["evaluator_metadata"]["scenario_id"] == "repeated_listing_iana_rootdb"


def test_llm_visible_context_excludes_forbidden_answer_hint_fields():
    scenario = BenchmarkScenario.model_validate(
        {
            "scenario_id": "navigate_then_extract_iana_protocols",
            "goal": "Open https://www.iana.org and extract heading after navigation",
            "start_url": "https://www.iana.org",
            "category": "navigation_then_extraction",
            "description": "d",
            "expected_output_type": "object",
            "required_top_level_fields": ["value"],
        }
    )
    goal = BenchmarkRunner._build_grounded_goal(scenario, allowed_actions=["open_url", "observe_page", "finish"])
    forbidden_tokens = [
        "expected_answer",
        "expected_value",
        "expected_pattern",
        "expected_heading",
        "expected_anchor",
        "anchor_candidates",
        "target_candidates",
        "preselected regex",
        "preselected section names",
    ]
    lowered = goal.lower()
    assert all(token.lower() not in lowered for token in forbidden_tokens)


def test_benchmark_allowed_actions_are_category_specific():
    assert BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["single_value_extraction"] == [
        "open_url",
        "extract_text",
        "extract_pattern_from_page_text",
        "finish",
    ]
    assert "click" not in BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["single_value_extraction"]
    assert "compare_structured_values" in BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY["multi_step_information_retrieval"]


def test_negative_open_url_then_finish_is_rejected_by_verifier_policy():
    verifier = LLMVerifier(DummyLLMClient())
    plan = TaskSpec.model_validate(
        {
            "goal": "Open page and report missing/ambiguous field.",
            "start_url": "https://www.iana.org/about",
            "allowed_domains": ["www.iana.org"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "Negative case", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.iana.org/about"}},
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )
    result = ExecutionResult(
        status="success",
        extracted_data={},
        logs=[
            StepLog(step_id=1, action="open_url", status="success"),
            StepLog(step_id=2, action="finish", status="success"),
        ],
    )
    verdict = verifier.verify(
        plan=plan,
        result=result,
        benchmark_context={"task_family": "negative_or_ambiguous_case"},
    )
    assert verdict.verdict == "reject"
    assert any("open_url -> finish" in issue for issue in verdict.issues)


def test_negative_without_probe_attempt_is_rejected_by_verifier_policy():
    verifier = LLMVerifier(DummyLLMClient())
    plan = TaskSpec.model_validate(
        {
            "goal": "Open page and report missing/ambiguous field.",
            "start_url": "https://www.iana.org/domains/reserved",
            "allowed_domains": ["www.iana.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "Negative case", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.iana.org/domains/reserved"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    result = ExecutionResult(
        status="success",
        extracted_data={"status": "not_found"},
        logs=[
            StepLog(step_id=1, action="open_url", status="success"),
            StepLog(step_id=2, action="observe_page", status="success"),
            StepLog(step_id=3, action="finish", status="success"),
        ],
    )
    verdict = verifier.verify(
        plan=plan,
        result=result,
        benchmark_context={"task_family": "negative_or_ambiguous_case"},
    )
    assert verdict.verdict == "reject"
    assert any("probe/extraction" in issue.lower() for issue in verdict.issues)


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


def test_normalize_benchmark_plan_keeps_extract_pattern_in_minimal_mode():
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
    assert extraction.action == "extract_pattern_from_page_text"


def test_normalize_benchmark_plan_sets_single_value_final_save_as_to_value():
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
    click_step = next(step for step in normalized.steps if step.action == "click")
    assert click_step.args.get("exact") is True
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_rewrites_navigation_plain_selector_click_to_text_contract():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"selector": "Tutorial"}},
                {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    click_step = next(step for step in normalized.steps if step.action == "click")
    assert "selector" not in click_step.args
    assert click_step.args["text"] == "Tutorial"
    assert click_step.args["exact"] is True


def test_normalize_benchmark_plan_recovers_empty_navigation_click_from_next_wait_for_text():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {}},
                {"step_id": 3, "action": "wait_for", "args": {"text": "Tutorial"}},
                {"step_id": 4, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 5, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    click_step = next(step for step in normalized.steps if step.action == "click")
    wait_step = next(step for step in normalized.steps if step.action == "wait_for")
    assert click_step.args["text"] == "Tutorial"
    assert click_step.args["exact"] is True
    assert wait_step.args["selector"] == "h1"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_recovers_empty_navigation_click_from_later_slug_hint():
    plan = TaskSpec.model_validate(
        {
            "goal": "navigate and extract",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {}},
                {"step_id": 3, "action": "open_url", "args": {"url": "https://example.com/tutorial/getting-started"}},
                {"step_id": 4, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 5, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    click_step = next(step for step in normalized.steps if step.action == "click")
    assert click_step.args["href_contains"] == "/tutorial/getting-started"
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_recovers_empty_navigation_click_from_goal_label():
    plan = TaskSpec.model_validate(
        {
            "goal": 'Open "Pricing" and extract the header',
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {}},
                {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )
    ctx = build_benchmark_context(category="navigation_then_extraction", task_family="navigation_then_extraction")
    normalized = normalize_benchmark_plan(plan, ctx)
    click_step = next(step for step in normalized.steps if step.action == "click")
    assert click_step.args["text"] == "Pricing"
    assert click_step.args["exact"] is True
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_replaces_placeholder_compare_headings_from_snapshot():
    plan = TaskSpec.model_validate(
        {
            "goal": "compare two sections",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 30},
            "expected_result": {"description": "x", "required_fields": ["combined_result"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "extract_section_lines", "args": {"heading_text": "Section A"}, "save_as": "source_a"},
                {"step_id": 3, "action": "extract_section_lines", "args": {"heading_text": "Section B"}, "save_as": "source_b"},
                {"step_id": 4, "action": "compare_structured_values", "args": {"left_key": "source_a", "right_key": "source_b"}},
                {"step_id": 5, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://example.com",
        title="Example",
        screenshot_path="artifacts/screenshots/x.png",
        page_text_excerpt="Example page",
        visible_headings=["Section A", "Pricing", "Documentation", "Pricing"],
        visible_labels=[],
        visible_buttons=[],
        visible_inputs=[],
        timestamp=datetime.now(timezone.utc),
        page_text="Pricing\nDocumentation\nMore details paragraph.",
    )
    ctx = build_benchmark_context(
        category="multi_step_information_retrieval",
        task_family="multi_step_information_retrieval",
    )
    normalized = normalize_benchmark_plan(plan, ctx, page_snapshot=snapshot)
    source_a = next(step for step in normalized.steps if step.save_as == "source_a")
    source_b = next(step for step in normalized.steps if step.save_as == "source_b")
    assert source_a.args["heading_text"] == "Pricing"
    assert source_b.args["heading_text"] == "Documentation"


def test_normalize_benchmark_plan_allows_regex_only_multi_step_compare_without_family_guardrail():
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
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_anchored_only_drops_page_language():
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
        required_top_level_fields=["value"],
    )
    normalized = normalize_benchmark_plan(plan, ctx)
    anchored_step = next(step for step in normalized.steps if step.action == "extract_value_near_anchor")
    assert anchored_step.args["anchor_candidates"] == ["Контакт", "Поддержка"]
    assert anchored_step.args["anchor_text"] == "Контактная информация"
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
    wait_step = next(step for step in normalized.steps if step.action == "wait_for")
    assert wait_step.args["selector"] == "h1"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_keeps_navigation_wait_shape():
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
    assert wait_step.args["selector"] == "h1"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_keeps_navigation_wait_text_for_role_name_click():
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
    assert wait_step.args["selector"] == "h1"
    assert "text" not in wait_step.args
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_normalize_benchmark_plan_allows_overconstrained_navigation_click_without_guardrail():
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
    PlanValidator().validate(normalized, allowed_actions=set(ctx["allowed_actions"]))


def test_observe_page_headings_have_preview_and_line_count():
    class _BodyLocator:
        async def inner_text(self):
            return (
                "Overview\n"
                "Overview line 1\n"
                "Overview line 2\n"
                "Introduction\n"
                "Details\n"
                "Detail line 1\n"
                "Detail line 2\n"
            )

    class _Page:
        url = "https://example.org"

        def locator(self, selector):
            if selector == "body":
                return _BodyLocator()
            raise AssertionError(selector)

        async def screenshot(self, **_kwargs):
            return None

        async def title(self):
            return "Example"

        async def evaluate(self, _script, _args):
            return [
                {"text": "Overview", "level": "h2", "index": 0, "visible": True},
                {"text": "Introduction", "level": "h2", "index": 1, "visible": True},
                {"text": "Details", "level": "h2", "index": 2, "visible": True},
            ]

    observer = PageObserver()

    async def _empty_list(*_args, **_kwargs):
        return []

    observer._collect_texts = _empty_list  # type: ignore[method-assign]
    observer._collect_inputs = _empty_list  # type: ignore[method-assign]
    snapshot = asyncio.run(observer.observe_page(page=_Page(), screenshot_path="artifacts/screenshots/t.png"))
    headings = snapshot.headings
    intro = next(item for item in headings if item.text == "Introduction")
    details = next(item for item in headings if item.text == "Details")
    assert intro.line_count_after == 0
    assert details.line_count_after == 2
    assert details.preview_after[:2] == ["Detail line 1", "Detail line 2"]


def test_observe_page_marks_nav_headings_as_non_content():
    class _BodyLocator:
        async def inner_text(self):
            return "About\nMain Section\nMain line 1\n"

    class _Page:
        url = "https://example.org"

        def locator(self, selector):
            if selector == "body":
                return _BodyLocator()
            raise AssertionError(selector)

        async def screenshot(self, **_kwargs):
            return None

        async def title(self):
            return "Example"

        async def evaluate(self, _script, _args):
            return [
                {"text": "About", "level": "h2", "index": 0, "visible": True, "region": "nav", "dom_path": "html>body>nav>h2"},
                {"text": "Main Section", "level": "h2", "index": 1, "visible": True, "region": "main", "dom_path": "html>body>main>h2"},
            ]

    observer = PageObserver()

    async def _empty_list(*_args, **_kwargs):
        return []

    observer._collect_texts = _empty_list  # type: ignore[method-assign]
    observer._collect_inputs = _empty_list  # type: ignore[method-assign]
    snapshot = asyncio.run(observer.observe_page(page=_Page(), screenshot_path="artifacts/screenshots/t2.png"))
    nav_heading = next(item for item in snapshot.headings if item.text == "About")
    main_heading = next(item for item in snapshot.headings if item.text == "Main Section")
    assert nav_heading.is_content_heading is False
    assert main_heading.is_content_heading is True
