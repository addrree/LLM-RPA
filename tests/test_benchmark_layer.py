from pathlib import Path

from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult, BenchmarkSelection
from app.benchmark.scenario_loader import load_scenario_suite


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
            export_success=True,
        ),
        BenchmarkScenarioResult(
            scenario_id="s2",
            category="negative_or_ambiguous_case",
            should_succeed=False,
            execution_status="failed",
            verifier_verdict="reject",
            runtime_sec=2.4,
            corrective_retry_used=True,
            export_success=False,
        ),
    ]

    metrics = BenchmarkRunner._compute_metrics(results)

    assert metrics.total_scenarios == 2
    assert metrics.execution_success_rate == 0.5
    assert metrics.verifier_accept_rate == 0.5
    assert metrics.correction_retry_usage_rate == 0.5
    assert metrics.export_success_rate == 0.5
    assert metrics.mean_runtime_sec == 1.8
