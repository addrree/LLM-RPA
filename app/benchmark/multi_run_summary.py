from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from pydantic import BaseModel, Field

from app.benchmark.runner import BenchmarkRunReport, BenchmarkScenarioResult
from app.config import BENCHMARKS_DIR

UTC = timezone.utc


class MultiRunMetricStats(BaseModel):
    mean: float
    std: float


class MultiRunFailureSummary(BaseModel):
    provider_failures: int
    semantic_failures: int
    other_failures: int


class MultiRunCorrectionSummary(BaseModel):
    attempted: int
    recovered: int
    recovery_rate_on_attempts: float
    recovery_frequency_overall: float


class MultiRunBenchmarkSummary(BaseModel):
    suite_id: str
    generated_at: str
    number_of_runs: int
    scenarios_per_run: int
    key_metrics: dict[str, MultiRunMetricStats]
    failure_summary: MultiRunFailureSummary
    correction_summary: MultiRunCorrectionSummary
    run_metric_snapshots: list[dict[str, float]] = Field(default_factory=list)
    failure_bucket_counts: dict[str, int] = Field(default_factory=dict)


def summarize_reports(reports: list[BenchmarkRunReport]) -> MultiRunBenchmarkSummary:
    if not reports:
        raise ValueError("Cannot summarize zero benchmark reports.")

    suite_id = reports[0].suite_id
    if any(report.suite_id != suite_id for report in reports):
        raise ValueError("All reports in multi-run summary must have the same suite_id.")

    metric_names = [
        "positive_execution_success_rate",
        "positive_verifier_accept_rate",
        "negative_expected_reject_rate",
        "plan_validation_pass_rate",
        "correction_recovery_rate",
        "export_success_rate",
        "mean_runtime_sec",
        "mean_planning_time_sec",
        "mean_execution_time_sec",
        "mean_verification_time_sec",
        "mean_correction_time_sec",
    ]

    key_metrics: dict[str, MultiRunMetricStats] = {}
    snapshots: list[dict[str, float]] = []
    all_scenarios: list[BenchmarkScenarioResult] = []

    for report in reports:
        metric_snapshot = report.metrics.model_dump(mode="json")
        snapshots.append({name: float(metric_snapshot[name]) for name in metric_names})
        all_scenarios.extend(report.scenarios)

    for name in metric_names:
        values = [float(getattr(report.metrics, name)) for report in reports]
        key_metrics[name] = MultiRunMetricStats(
            mean=round(mean(values), 6),
            std=round(pstdev(values), 6),
        )

    attempted = [item for item in all_scenarios if item.correction_attempt_count > 0]
    recovered = [item for item in attempted if _is_expected_outcome(item)]
    correction_summary = MultiRunCorrectionSummary(
        attempted=len(attempted),
        recovered=len(recovered),
        recovery_rate_on_attempts=_safe_ratio(len(recovered), len(attempted)),
        recovery_frequency_overall=_safe_ratio(len(recovered), len(all_scenarios)),
    )

    failure_buckets = Counter(item.failure_bucket or "none" for item in all_scenarios)
    provider_failures = sum(1 for item in all_scenarios if _is_provider_failure(item))
    semantic_failures = sum(
        1
        for item in all_scenarios
        if item.failure_bucket == "semantic_failure" or item.negative_outcome in {"unexpected_accept", "unexpected_uncertain"}
    )
    other_failures = max(len(all_scenarios) - provider_failures - semantic_failures, 0)

    return MultiRunBenchmarkSummary(
        suite_id=suite_id,
        generated_at=datetime.now(UTC).isoformat(),
        number_of_runs=len(reports),
        scenarios_per_run=reports[0].metrics.total_scenarios,
        key_metrics=key_metrics,
        failure_summary=MultiRunFailureSummary(
            provider_failures=provider_failures,
            semantic_failures=semantic_failures,
            other_failures=other_failures,
        ),
        correction_summary=correction_summary,
        run_metric_snapshots=snapshots,
        failure_bucket_counts=dict(failure_buckets),
    )


def write_multi_run_summary(summary: MultiRunBenchmarkSummary) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path = BENCHMARKS_DIR / f"benchmark_multi_run_summary_{summary.suite_id}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def load_reports(report_paths: list[Path]) -> list[BenchmarkRunReport]:
    reports: list[BenchmarkRunReport] = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append(BenchmarkRunReport.model_validate(payload))
    return reports


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_provider_failure(item: BenchmarkScenarioResult) -> bool:
    error = (item.error_message or "").lower()
    if "ollama" in error or "llm" in error or "model" in error:
        return True
    return item.failure_stage in {"planning", "verification"} and item.verifier_verdict == "error"


def _is_expected_outcome(item: BenchmarkScenarioResult) -> bool:
    if item.should_succeed:
        return item.execution_status == "success" and item.verifier_verdict == "accept"
    return item.negative_outcome == "expected_reject"
