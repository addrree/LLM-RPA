from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.runner import BenchmarkRunner, BenchmarkScenarioResult, write_benchmark_report
from app.benchmark.scenario_loader import load_scenario_suite
from app.config import BENCHMARKS_DIR
from app.main import build_llm_client, build_workflow

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite N times and aggregate scenario stability metrics.")
    parser.add_argument("--suite", type=Path, required=True, help="Path to benchmark suite JSON")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs")
    parser.add_argument("--backend", choices=["ollama", "ollama_cloud"], default=None)
    parser.add_argument("--two-stage-planning", action="store_true")
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--export-format", action="append", choices=["json", "csv"], default=["json"])
    return parser.parse_args()


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_expected_outcome(item: BenchmarkScenarioResult) -> bool:
    if item.should_succeed:
        return item.execution_status == "success" and item.verifier_verdict == "accept"
    return item.negative_outcome == "expected_reject"


def _summarize_by_scenario(reports: list[dict]) -> list[dict]:
    per_scenario: dict[str, list[BenchmarkScenarioResult]] = defaultdict(list)
    for report in reports:
        for item in report["scenarios"]:
            per_scenario[item.scenario_id].append(item)

    summary_rows: list[dict] = []
    for scenario_id, attempts in sorted(per_scenario.items()):
        runtime_values = [item.runtime_sec for item in attempts]
        planning_values = [item.planning_time_sec for item in attempts]
        execution_values = [item.execution_time_sec for item in attempts]
        verification_values = [item.verification_time_sec for item in attempts]
        correction_values = [item.correction_time_sec for item in attempts]
        failure_buckets = Counter(item.failure_bucket or "none" for item in attempts)

        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "category": attempts[0].category,
                "runs": len(attempts),
                "pass_rate": round(_safe_ratio(sum(1 for item in attempts if _is_expected_outcome(item)), len(attempts)), 6),
                "verifier_accept_rate": round(_safe_ratio(sum(1 for item in attempts if item.verifier_verdict == "accept"), len(attempts)), 6),
                "runtime_mean_sec": round(mean(runtime_values), 6),
                "runtime_std_sec": round(pstdev(runtime_values), 6),
                "planning_time_mean_sec": round(mean(planning_values), 6),
                "execution_time_mean_sec": round(mean(execution_values), 6),
                "verification_time_mean_sec": round(mean(verification_values), 6),
                "correction_time_mean_sec": round(mean(correction_values), 6),
                "correction_usage_rate": round(_safe_ratio(sum(1 for item in attempts if item.corrective_retry_used), len(attempts)), 6),
                "failure_buckets": dict(sorted(failure_buckets.items())),
            }
        )
    return summary_rows


async def main() -> None:
    args = parse_args()
    suite = load_scenario_suite(args.suite)
    llm_client = build_llm_client(backend=args.backend)

    runner = BenchmarkRunner(
        workflow_factory=lambda: build_workflow(
            llm_client=llm_client,
            show_browser=args.show_browser,
            slow_mo=args.slow_mo,
            record_video=args.record_video,
            two_stage_planning=args.two_stage_planning,
        ),
        export_formats=list(dict.fromkeys(args.export_format or ["json"])),
    )

    reports: list[dict] = []
    report_paths: list[dict] = []
    for idx in range(max(args.runs, 1)):
        report = await runner.run_suite(suite)
        json_path, csv_path = write_benchmark_report(report)
        reports.append({"metrics": report.metrics, "scenarios": report.scenarios})
        report_paths.append({"run_index": idx + 1, "json_path": str(json_path), "csv_path": str(csv_path)})
        print(f"[run {idx + 1}] json={json_path} csv={csv_path}")

    by_scenario = _summarize_by_scenario(reports)
    now = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_json = BENCHMARKS_DIR / f"benchmark_repeats_aggregate_{suite.suite_id}_{now}.json"
    out_csv = BENCHMARKS_DIR / f"benchmark_repeats_aggregate_{suite.suite_id}_{now}.csv"

    payload = {
        "suite_id": suite.suite_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": max(args.runs, 1),
        "source_reports": report_paths,
        "scenario_aggregate": by_scenario,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "category",
                "runs",
                "pass_rate",
                "verifier_accept_rate",
                "runtime_mean_sec",
                "runtime_std_sec",
                "planning_time_mean_sec",
                "execution_time_mean_sec",
                "verification_time_mean_sec",
                "correction_time_mean_sec",
                "correction_usage_rate",
                "failure_buckets",
            ],
        )
        writer.writeheader()
        for row in by_scenario:
            csv_row = row.copy()
            csv_row["failure_buckets"] = json.dumps(csv_row["failure_buckets"], ensure_ascii=False)
            writer.writerow(csv_row)

    print(f"aggregate_json={out_json}")
    print(f"aggregate_csv={out_csv}")


if __name__ == "__main__":
    asyncio.run(main())
