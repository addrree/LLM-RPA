from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.benchmark.scenario_loader import BenchmarkScenario, ScenarioSuite
from app.benchmark.policies import build_benchmark_context
from app.config import BENCHMARKS_DIR
from app.orchestrator.persistence import export_results, save_artifacts
from app.orchestrator.workflow_manager import WorkflowStageError

UTC = timezone.utc
VALID_FAILURE_STAGES = {"planning", "validation", "execution", "verification", "export"}


class BenchmarkScenarioResult(BaseModel):
    scenario_id: str
    category: str
    should_succeed: bool
    execution_status: str
    verifier_verdict: str
    runtime_sec: float
    corrective_retry_used: bool
    correction_attempt_count: int = 0
    corrective_plan_valid_count: int = 0
    corrective_plan_invalid_count: int = 0
    initial_plan_valid: bool | None = None
    final_plan_valid: bool | None = None
    action_oov_detected: bool = False
    failure_stage: str | None = None
    export_success: bool
    final_url: str | None = None
    error_message: str | None = None
    execution_failure_type: str | None = None
    failed_action: str | None = None
    failed_args: dict = Field(default_factory=dict)
    execution_logs_tail: list[str] = Field(default_factory=list)
    retry_artifact_count: int = 0
    compare_status: str | None = None
    technical_failure: bool = False
    negative_outcome: str | None = None
    failure_bucket: str | None = None
    notes: str = ""


class BenchmarkMetrics(BaseModel):
    total_scenarios: int
    positive_execution_success_rate: float
    positive_verifier_accept_rate: float
    negative_expected_reject_rate: float
    plan_validation_pass_rate: float
    correction_recovery_rate: float
    corrective_plan_valid_count: int
    corrective_plan_invalid_count: int
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
        correction_attempt_count = 0
        corrective_plan_valid_count = 0
        corrective_plan_invalid_count = 0
        initial_plan_valid = None
        final_plan_valid = None
        action_oov_detected = False
        failure_stage = None
        export_success = False
        final_url = None
        error_message = None
        execution_failure_type = None
        failed_action = None
        failed_args = {}
        execution_logs_tail: list[str] = []
        retry_artifact_count = 0
        compare_status = None
        technical_failure = False

        try:
            benchmark_context = build_benchmark_context(
                category=scenario.category,
                task_family=scenario.task_family,
                scenario_id=scenario.scenario_id,
                expected_min_items=scenario.expected_min_items,
                expected_item_fields=scenario.expected_item_fields,
                required_top_level_fields=scenario.required_top_level_fields,
            )
            result = await workflow.run(
                self._build_grounded_goal(scenario, allowed_actions=benchmark_context["allowed_actions"]),
                benchmark_context=benchmark_context,
            )
            execution = result["execution_result"]
            verdict = result["verdict"]
            execution_status = execution.status
            verifier_verdict = verdict.verdict
            corrective_retry_used = bool(result.get("corrective_retry_used", False))
            correction_attempt_count = int(result.get("correction_attempt_count", result.get("corrective_retry_count", 0)))
            corrective_plan_valid_count = int(result.get("corrective_plan_valid_count", 0))
            corrective_plan_invalid_count = int(result.get("corrective_plan_invalid_count", 0))
            initial_plan_valid = result.get("initial_plan_valid")
            final_plan_valid = result.get("final_plan_valid")
            action_oov_detected = bool(result.get("action_oov_detected", False))
            final_url = execution.final_url
            error_message = execution.error_message
            execution_failure_type = execution.failure_type
            failed_action = execution.failed_action
            failed_args = execution.failed_args
            execution_logs_tail = [
                f"{entry.step_id}:{entry.action}:{entry.status}:{(entry.message or '').strip()}"
                for entry in execution.logs[-5:]
            ]
            technical_failure = bool(execution.technical_failure)
            retry_artifact_count = len(execution.retry_artifacts)
            if isinstance(execution.extracted_data.get("structured_comparison"), dict):
                compare_status = execution.extracted_data["structured_comparison"].get("status")

            save_artifacts(result, run_id=run_id)
            try:
                export_results(result, run_id=run_id, export_formats=self.export_formats)
                export_success = True
            except Exception as export_exc:  # noqa: BLE001
                export_success = False
                error_message = str(export_exc)
                failure_stage = "export"

            if failure_stage is None:
                failure_stage = self._infer_failure_stage(
                    should_succeed=scenario.should_succeed,
                    execution_status=execution_status,
                    verifier_verdict=verifier_verdict,
                    initial_plan_valid=initial_plan_valid,
                    final_plan_valid=final_plan_valid,
                    export_success=export_success,
                )
        except WorkflowStageError as exc:
            failure_stage = exc.stage if exc.stage in VALID_FAILURE_STAGES else "execution"
            error_message = str(exc)
            if failure_stage == "validation":
                if initial_plan_valid is None:
                    initial_plan_valid = False
                if final_plan_valid is None:
                    final_plan_valid = False
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            if failure_stage is None:
                failure_stage = "execution"

        runtime_sec = round(perf_counter() - started, 3)
        scenario_result = BenchmarkScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            should_succeed=scenario.should_succeed,
            execution_status=execution_status,
            verifier_verdict=verifier_verdict,
            runtime_sec=runtime_sec,
            corrective_retry_used=corrective_retry_used,
            correction_attempt_count=correction_attempt_count,
            corrective_plan_valid_count=corrective_plan_valid_count,
            corrective_plan_invalid_count=corrective_plan_invalid_count,
            initial_plan_valid=initial_plan_valid,
            final_plan_valid=final_plan_valid,
            action_oov_detected=action_oov_detected,
            failure_stage=failure_stage,
            export_success=export_success,
            final_url=final_url,
            error_message=error_message,
            execution_failure_type=execution_failure_type,
            failed_action=failed_action,
            failed_args=failed_args,
            execution_logs_tail=execution_logs_tail,
            technical_failure=technical_failure,
            retry_artifact_count=retry_artifact_count,
            compare_status=compare_status,
            notes=scenario.notes,
        )
        scenario_result.negative_outcome = (
            self._classify_negative_outcome(scenario_result)
            if not scenario.should_succeed
            else None
        )
        scenario_result.failure_bucket = self._classify_failure_bucket(scenario_result)
        return scenario_result

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
                positive_execution_success_rate=0.0,
                positive_verifier_accept_rate=0.0,
                negative_expected_reject_rate=0.0,
                plan_validation_pass_rate=0.0,
                correction_recovery_rate=0.0,
                corrective_plan_valid_count=0,
                corrective_plan_invalid_count=0,
                export_success_rate=0.0,
                mean_runtime_sec=0.0,
            )

        positive = [item for item in results if item.should_succeed]
        negative = [item for item in results if not item.should_succeed]
        positive_execution_success = sum(1 for item in positive if item.execution_status == "success")
        positive_verifier_accept = sum(1 for item in positive if item.verifier_verdict == "accept")
        semantic_negative = [item for item in negative if BenchmarkRunner._is_semantic_negative_trial(item)]
        negative_expected_reject = sum(1 for item in semantic_negative if BenchmarkRunner._is_expected_negative_reject(item))
        plan_validation_pass = sum(1 for item in results if BenchmarkRunner._plan_validation_passed(item))
        correction_attempted = [item for item in results if item.correction_attempt_count > 0]
        correction_recovered = sum(
            1
            for item in correction_attempted
            if BenchmarkRunner._is_expected_outcome(item)
        )
        export_success = sum(1 for item in results if item.export_success)
        corrective_plan_valid_total = sum(item.corrective_plan_valid_count for item in results)
        corrective_plan_invalid_total = sum(item.corrective_plan_invalid_count for item in results)
        mean_runtime = sum(item.runtime_sec for item in results) / total

        return BenchmarkMetrics(
            total_scenarios=total,
            positive_execution_success_rate=BenchmarkRunner._safe_ratio(positive_execution_success, len(positive)),
            positive_verifier_accept_rate=BenchmarkRunner._safe_ratio(positive_verifier_accept, len(positive)),
            negative_expected_reject_rate=BenchmarkRunner._safe_ratio(negative_expected_reject, len(semantic_negative)),
            plan_validation_pass_rate=plan_validation_pass / total,
            correction_recovery_rate=BenchmarkRunner._safe_ratio(correction_recovered, len(correction_attempted)),
            corrective_plan_valid_count=corrective_plan_valid_total,
            corrective_plan_invalid_count=corrective_plan_invalid_total,
            export_success_rate=export_success / total,
            mean_runtime_sec=round(mean_runtime, 3),
        )

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _plan_validation_passed(item: BenchmarkScenarioResult) -> bool:
        if item.failure_stage == "validation":
            return False
        if item.initial_plan_valid is False or item.final_plan_valid is False:
            return False
        if item.corrective_plan_invalid_count > 0:
            return False
        return True

    @staticmethod
    def _is_expected_outcome(item: BenchmarkScenarioResult) -> bool:
        if item.should_succeed:
            return item.execution_status == "success" and item.verifier_verdict == "accept"
        return BenchmarkRunner._is_expected_negative_reject(item)

    @staticmethod
    def _is_expected_negative_reject(item: BenchmarkScenarioResult) -> bool:
        return BenchmarkRunner._classify_negative_outcome(item) == "expected_reject"

    @staticmethod
    def _is_semantic_negative_trial(item: BenchmarkScenarioResult) -> bool:
        return not item.technical_failure

    @staticmethod
    def _classify_negative_outcome(item: BenchmarkScenarioResult) -> str:
        if item.technical_failure:
            return "technical_failure"
        if item.verifier_verdict == "reject":
            return "expected_reject"
        if item.verifier_verdict == "accept":
            return "unexpected_accept"
        return "unexpected_uncertain"

    @staticmethod
    def _classify_failure_bucket(item: BenchmarkScenarioResult) -> str:
        if item.failure_stage is None:
            return "none"
        if item.failure_stage in {"planning", "validation"}:
            return "plan_failure"
        if item.failure_stage == "execution":
            if item.technical_failure or str(item.execution_failure_type).startswith("browser_"):
                return "browser_flaky_failure"
            return "execution_failure"
        if item.failure_stage == "verification":
            return "semantic_failure"
        if item.failure_stage == "export":
            return "export_failure"
        return "unknown_failure"

    @staticmethod
    def _infer_failure_stage(
        *,
        should_succeed: bool,
        execution_status: str,
        verifier_verdict: str,
        initial_plan_valid: bool | None,
        final_plan_valid: bool | None,
        export_success: bool,
    ) -> str | None:
        if initial_plan_valid is False or final_plan_valid is False:
            return "validation"
        if execution_status != "success":
            return "execution"
        if should_succeed and verifier_verdict != "accept":
            return "verification"
        if not should_succeed and verifier_verdict != "reject":
            return "verification"
        if not export_success:
            return "export"
        return None

    @staticmethod
    def _build_grounded_goal(scenario: BenchmarkScenario, *, allowed_actions: list[str] | None = None) -> str:
        domain = urlparse(str(scenario.start_url)).netloc
        parts = [
            f"Task family is {scenario.task_family or scenario.category}.",
            f"User goal: {scenario.goal.strip()}",
            f"Open this URL first: {scenario.start_url}.",
        ]
        if domain:
            parts.append(f"Stay within domain: {domain}.")
        parts.append("Infer anchors/headings/value patterns from observe_page or page snapshot.")
        parts.append("Do not rely on pre-specified expected candidates or expected values.")
        if allowed_actions:
            parts.append(f"Use only these actions: {', '.join(allowed_actions)}.")
        if scenario.preconditions:
            parts.append(f"Constraints: {'; '.join(scenario.preconditions[:3])}.")
        return "\n".join(parts)


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
                "correction_attempt_count",
                "corrective_plan_valid_count",
                "corrective_plan_invalid_count",
                "initial_plan_valid",
                "final_plan_valid",
                "action_oov_detected",
                "failure_stage",
                "export_success",
                "final_url",
                "error_message",
                "execution_failure_type",
                "failed_action",
                "failed_args",
                "execution_logs_tail",
                "technical_failure",
                "retry_artifact_count",
                "compare_status",
                "negative_outcome",
                "failure_bucket",
                "notes",
            ],
        )
        writer.writeheader()
        for scenario in report.scenarios:
            writer.writerow(scenario.model_dump(mode="json"))

    return json_path, csv_path
