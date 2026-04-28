from pathlib import Path

from app.benchmark.multi_run_summary import summarize_reports
from app.benchmark.runner import BenchmarkMetrics, BenchmarkRunReport, BenchmarkScenarioResult
from app.benchmark.scenario_loader import load_scenario_suite


def _build_report(*, run_suffix: str, rate: float) -> BenchmarkRunReport:
    scenario = BenchmarkScenarioResult(
        scenario_id=f"s_{run_suffix}",
        category="single_value_extraction",
        should_succeed=True,
        execution_status="success",
        verifier_verdict="accept",
        runtime_sec=1.0,
        corrective_retry_used=True,
        correction_attempt_count=1,
        export_success=True,
    )
    metrics = BenchmarkMetrics(
        total_scenarios=1,
        positive_execution_success_rate=rate,
        positive_verifier_accept_rate=rate,
        negative_expected_reject_rate=0.0,
        plan_validation_pass_rate=rate,
        correction_recovery_rate=rate,
        corrective_plan_valid_count=1,
        corrective_plan_invalid_count=0,
        export_success_rate=rate,
        mean_runtime_sec=1.0,
    )
    return BenchmarkRunReport(
        suite_id="core_generalized_task_suite_v2",
        generated_at="2026-04-24T00:00:00+00:00",
        metrics=metrics,
        scenarios=[scenario],
    )


def test_extended_suite_contains_all_families_and_multiple_sites():
    suite = load_scenario_suite(Path("benchmarks/scenarios/extended_generalized_task_suite.json"))
    categories = {scenario.category for scenario in suite.scenarios}
    assert categories == {
        "single_value_extraction",
        "anchored_value_extraction",
        "repeated_structured_items",
        "navigation_then_extraction",
        "multi_step_information_retrieval",
        "negative_or_ambiguous_case",
    }
    assert len(suite.scenarios) == 12
    hosts = {scenario.start_url.split("/")[2] for scenario in suite.scenarios}
    assert len(hosts) >= 5


def test_multi_run_summary_aggregates_mean_std_and_recovery_frequency():
    reports = [_build_report(run_suffix="a", rate=1.0), _build_report(run_suffix="b", rate=0.5)]
    summary = summarize_reports(reports)

    assert summary.number_of_runs == 2
    assert summary.key_metrics["positive_execution_success_rate"].mean == 0.75
    assert summary.key_metrics["positive_execution_success_rate"].std == 0.25
    assert summary.correction_summary.attempted == 2
    assert summary.correction_summary.recovered == 2
    assert summary.correction_summary.recovery_frequency_overall == 1.0
