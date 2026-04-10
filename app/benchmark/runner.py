from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from app.benchmark.scenario_loader import BenchmarkScenario, ScenarioSuite
from app.config import BENCHMARKS_DIR
from app.orchestrator.persistence import export_results, save_artifacts

UTC = timezone.utc


class BenchmarkScenarioResult(BaseModel):
    scenario_id: str
    category: str
    should_succeed: bool
    execution_status: str
    verifier_verdict: str
    runtime_sec: float
    corrective_retry_used: bool
    export_success: bool
    final_url: str | None = None
    error_message: str | None = None
    notes: str = ""


class BenchmarkMetrics(BaseModel):
    total_scenarios: int
    execution_success_rate: float
    verifier_accept_rate: float
    correction_retry_usage_rate: float
    export_success_rate: float
    mean_runtime_sec: float


class BenchmarkRunReport(BaseModel):
    suite_id: str
    generated_at: str
    metrics: BenchmarkMetrics
    scenarios: list[BenchmarkScenarioResult]


@dataclass
class BenchmarkSelection:
    scenario_ids: list[str] | None = None
    categories: list[str] | None = None


class BenchmarkRunner:
    def __init__(
        self,
        workflow_factory: Callable[[], object],
        export_formats: list[str] | None = None,
    ):
        self.workflow_factory = workflow_factory
        self.export_formats = export_formats or ["json"]

    async def run_suite(self, suite: ScenarioSuite, selection: BenchmarkSelection | None = None) -> BenchmarkRunReport:
        filtered = self._filter_scenarios(suite.scenarios, selection)
        results: list[BenchmarkScenarioResult] = []

        for scenario in filtered:
            results.append(await self._run_one(scenario))

        metrics = self._compute_metrics(results)
        return BenchmarkRunReport(
            suite_id=suite.suite_id,
            generated_at=datetime.now(UTC).isoformat(),
            metrics=metrics,
            scenarios=results,
        )

    async def _run_one(self, scenario: BenchmarkScenario) -> BenchmarkScenarioResult:
        started = perf_counter()
        workflow = self.workflow_factory()
        run_id = datetime.now(UTC).strftime(f"benchmark_{scenario.scenario_id}_%Y%m%d_%H%M%S")

        execution_status = "failed"
        verifier_verdict = "error"
        corrective_retry_used = False
        export_success = False
        final_url = None
        error_message = None

        try:
            result = await workflow.run(scenario.goal)
            execution = result["execution_result"]
            verdict = result["verdict"]
            execution_status = execution.status
            verifier_verdict = verdict.verdict
            corrective_retry_used = bool(result.get("corrective_retry_used", False))
            final_url = execution.final_url
            error_message = execution.error_message

            save_artifacts(result, run_id=run_id)
            export_results(result, run_id=run_id, export_formats=self.export_formats)
            export_success = True
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)

        runtime_sec = round(perf_counter() - started, 3)
        return BenchmarkScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            should_succeed=scenario.should_succeed,
            execution_status=execution_status,
            verifier_verdict=verifier_verdict,
            runtime_sec=runtime_sec,
            corrective_retry_used=corrective_retry_used,
            export_success=export_success,
            final_url=final_url,
            error_message=error_message,
            notes=scenario.notes,
        )

    @staticmethod
    def _filter_scenarios(
        scenarios: list[BenchmarkScenario],
        selection: BenchmarkSelection | None,
    ) -> list[BenchmarkScenario]:
        if selection is None:
            return scenarios

        scenario_ids = set(selection.scenario_ids or [])
        categories = set(selection.categories or [])

        filtered = scenarios
        if scenario_ids:
            filtered = [scenario for scenario in filtered if scenario.scenario_id in scenario_ids]
        if categories:
            filtered = [scenario for scenario in filtered if scenario.category in categories]
        return filtered

    @staticmethod
    def _compute_metrics(results: list[BenchmarkScenarioResult]) -> BenchmarkMetrics:
        total = len(results)
        if total == 0:
            return BenchmarkMetrics(
                total_scenarios=0,
                execution_success_rate=0.0,
                verifier_accept_rate=0.0,
                correction_retry_usage_rate=0.0,
                export_success_rate=0.0,
                mean_runtime_sec=0.0,
            )

        execution_success = sum(1 for item in results if item.execution_status == "success")
        verifier_accept = sum(1 for item in results if item.verifier_verdict == "accept")
        correction_used = sum(1 for item in results if item.corrective_retry_used)
        export_success = sum(1 for item in results if item.export_success)
        mean_runtime = sum(item.runtime_sec for item in results) / total

        return BenchmarkMetrics(
            total_scenarios=total,
            execution_success_rate=execution_success / total,
            verifier_accept_rate=verifier_accept / total,
            correction_retry_usage_rate=correction_used / total,
            export_success_rate=export_success / total,
            mean_runtime_sec=round(mean_runtime, 3),
        )


def write_benchmark_report(report: BenchmarkRunReport) -> tuple[Path, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = BENCHMARKS_DIR / f"benchmark_summary_{report.suite_id}_{timestamp}.json"
    csv_path = BENCHMARKS_DIR / f"benchmark_summary_{report.suite_id}_{timestamp}.csv"

    json_path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario_id",
                "category",
                "should_succeed",
                "execution_status",
                "verifier_verdict",
                "runtime_sec",
                "corrective_retry_used",
                "export_success",
                "final_url",
                "error_message",
                "notes",
            ],
        )
        writer.writeheader()
        for scenario in report.scenarios:
            writer.writerow(scenario.model_dump(mode="json"))

    return json_path, csv_path
