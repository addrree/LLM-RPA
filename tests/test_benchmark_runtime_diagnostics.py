from pathlib import Path

from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult
from app.benchmark.scenario_loader import load_scenario_suite


def test_metrics_include_runtime_breakdown_means():
    results = [
        BenchmarkScenarioResult(
            scenario_id="s1",
            category="single_value_extraction",
            should_succeed=True,
            execution_status="success",
            verifier_verdict="accept",
            runtime_sec=10.0,
            planning_time_sec=1.0,
            execution_time_sec=6.0,
            verification_time_sec=2.0,
            correction_time_sec=1.0,
            corrective_retry_used=False,
            correction_attempt_count=0,
            export_success=True,
        ),
        BenchmarkScenarioResult(
            scenario_id="s2",
            category="single_value_extraction",
            should_succeed=True,
            execution_status="success",
            verifier_verdict="accept",
            runtime_sec=20.0,
            planning_time_sec=2.0,
            execution_time_sec=11.0,
            verification_time_sec=5.0,
            correction_time_sec=2.0,
            corrective_retry_used=True,
            correction_attempt_count=1,
            export_success=True,
        ),
    ]

    metrics = BenchmarkRunner._compute_metrics(results)
    assert metrics.mean_runtime_sec == 15.0
    assert metrics.mean_planning_time_sec == 1.5
    assert metrics.mean_execution_time_sec == 8.5
    assert metrics.mean_verification_time_sec == 3.5
    assert metrics.mean_correction_time_sec == 1.5


def test_v3_suite_avoids_answer_leaking_fields():
    suite = load_scenario_suite(Path("benchmarks/scenarios/core_task_suite_v3.json"))
    assert len(suite.scenarios) >= 6
    forbidden = {"expected_answer", "expected_answer_values", "anchor_candidates", "page_language"}

    for scenario in suite.scenarios:
        payload = scenario.model_dump(mode="json")
        for key in forbidden:
            if key in {"anchor_candidates", "page_language"}:
                # present in schema but intentionally empty in v3 for generalized setup
                assert not payload.get(key)
            else:
                assert key not in payload
