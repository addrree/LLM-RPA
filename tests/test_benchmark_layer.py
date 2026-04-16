from pathlib import Path

from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult, BenchmarkSelection
from app.benchmark.scenario_loader import BenchmarkScenario, load_scenario_suite
from app.planner.action_vocab import normalize_plan_action_aliases


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


def test_action_alias_normalization_reports_oov_without_rewriting_actions():
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
