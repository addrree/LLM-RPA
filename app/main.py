import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.benchmark import (
    BenchmarkRunner,
    BenchmarkSelection,
    load_reports,
    load_scenario_suite,
    summarize_reports,
    write_benchmark_report,
    write_multi_run_summary,
)
from app.executor.playwright_executor import PlaywrightExecutor
from app.orchestrator.persistence import export_results, save_artifacts
from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.utils.llm_client import LLMClient, LLMClientError
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier

UTC = timezone.utc


def build_llm_client(backend: str | None = None):
    selected_backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()

    return LLMClient(
        backend=selected_backend,
        planner_model=os.getenv("OLLAMA_PLANNER_MODEL", os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")),
        verifier_model=os.getenv("OLLAMA_VERIFIER_MODEL", os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "https://ollama.com" if selected_backend == "ollama_cloud" else "http://localhost:11434",
        ),
    )


def build_workflow(*, llm_client, show_browser: bool, slow_mo: int, record_video: bool, two_stage_planning: bool, interaction_mode: str = "plan"):
    return WorkflowManager(
        planner=Planner(llm_client),
        validator=PlanValidator(),
        executor=PlaywrightExecutor(headless=not show_browser, slow_mo=slow_mo, record_video=record_video),
        verifier=LLMVerifier(llm_client),
        replanner=Replanner(llm_client),
        two_stage_planning=two_stage_planning,
        interaction_mode=interaction_mode,
    )


async def run(
    user_goal: str,
    backend: str | None = None,
    show_browser: bool = False,
    slow_mo: int = 0,
    record_video: bool = False,
    export_formats: list[str] | None = None,
    two_stage_planning: bool = False,
    interaction_mode: str = "plan",
):
    export_formats = export_formats or ["json"]
    export_formats = list(dict.fromkeys(export_formats))
    llm_client = build_llm_client(backend=backend)

    workflow = build_workflow(
        llm_client=llm_client,
        show_browser=show_browser,
        slow_mo=slow_mo,
        record_video=record_video,
        two_stage_planning=two_stage_planning,
        interaction_mode=interaction_mode,
    )

    result = await workflow.run(user_goal)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    artifact_paths = save_artifacts(result, run_id=run_id)
    export_paths = export_results(result, run_id=run_id, export_formats=export_formats)

    print("\nWORKFLOW SUMMARY:")
    print(json.dumps({
        "execution_status": result["execution_result"].status,
        "verdict": result["verdict"].verdict,
        "confidence": result["verdict"].confidence,
        "planning_mode": result.get("planning_mode", "single_stage"),
        "corrective_retry_used": result.get("corrective_retry_used", False),
        "corrective_retry_count": result.get("corrective_retry_count", 0),
        "extracted_keys": sorted(list(result["execution_result"].extracted_data.keys())),
    }, ensure_ascii=False, indent=2))

    print("\nARTIFACTS:")
    for name, path in artifact_paths.items():
        if path:
            print(f"- {name}: {path}")

    print("\nEXPORTS:")
    for path in export_paths:
        print(f"- {path}")


async def run_benchmark(
    *,
    suite_path: Path,
    scenario_ids: list[str] | None,
    categories: list[str] | None,
    backend: str | None,
    show_browser: bool,
    slow_mo: int,
    record_video: bool,
    export_formats: list[str] | None,
    two_stage_planning: bool,
    benchmark_runs: int,
    interaction_mode: str = "plan",
):
    export_formats = export_formats or ["json"]
    export_formats = list(dict.fromkeys(export_formats))
    llm_client = build_llm_client(backend=backend)
    suite = load_scenario_suite(suite_path)

    runner = BenchmarkRunner(
        workflow_factory=lambda: build_workflow(
            llm_client=llm_client,
            show_browser=show_browser,
            slow_mo=slow_mo,
            record_video=record_video,
            two_stage_planning=two_stage_planning,
            interaction_mode=interaction_mode,
        ),
        export_formats=export_formats,
    )
    selection = BenchmarkSelection(scenario_ids=scenario_ids, categories=categories)
    run_reports = []
    for run_index in range(benchmark_runs):
        report = await runner.run_suite(suite, selection=selection)
        report_json, report_csv = write_benchmark_report(report)
        run_reports.append(report)

        print(f"\nBENCHMARK RUN {run_index + 1}/{benchmark_runs} SUMMARY:")
        print(json.dumps(report.metrics.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("\nBENCHMARK REPORTS:")
        print(f"- json: {report_json}")
        print(f"- csv: {report_csv}")

    if benchmark_runs > 1:
        summary = summarize_reports(run_reports)
        summary_path = write_multi_run_summary(summary)
        print("\nMULTI-RUN SUMMARY:")
        print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print(f"- json: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM-RPA MVP pipeline")
    parser.add_argument(
        "--goal",
        default="Open https://www.wikipedia.org, extract the h1 text, take screenshot and finish.",
        help="User goal in natural language",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "ollama_cloud"],
        default=None,
        help="LLM backend to use (default from LLM_BACKEND env, fallback: ollama)",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Playwright in headed mode so browser actions are visible",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Delay between Playwright actions in milliseconds",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record Playwright session video to artifacts/videos",
    )
    parser.add_argument(
        "--two-stage-planning",
        action="store_true",
        help="Enable two-stage planning: initial observation plan + context-aware replanning",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=["plan", "observe_action"],
        default="plan",
        help="Use the existing planning pipeline or opt-in observe/action loop",
    )
    parser.add_argument(
        "--export-format",
        action="append",
        choices=["json", "csv"],
        default=None,
        help="Export format for workflow result (can be specified multiple times)",
    )
    parser.add_argument(
        "--benchmark-suite",
        type=Path,
        default=Path("benchmarks/scenarios/core_task_suite.json"),
        help="Path to benchmark scenario suite JSON",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Run all scenarios from benchmark suite",
    )
    parser.add_argument(
        "--benchmark-scenario",
        action="append",
        default=None,
        help="Run only specific scenario_id (can be repeated)",
    )
    parser.add_argument(
        "--benchmark-category",
        action="append",
        default=None,
        help="Run only specific category (can be repeated)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1,
        help="Number of times to execute selected benchmark scenarios sequentially",
    )
    parser.add_argument(
        "--benchmark-summarize-report",
        action="append",
        default=None,
        help="Path to existing benchmark_summary_*.json report (can be repeated) for offline multi-run summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    benchmark_requested = args.benchmark_all or args.benchmark_scenario or args.benchmark_category
    benchmark_summary_requested = bool(args.benchmark_summarize_report)

    try:
        if benchmark_summary_requested:
            report_paths = [Path(path) for path in args.benchmark_summarize_report]
            reports = load_reports(report_paths)
            summary = summarize_reports(reports)
            summary_path = write_multi_run_summary(summary)
            print("\nMULTI-RUN SUMMARY:")
            print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
            print(f"- json: {summary_path}")
        elif benchmark_requested:
            asyncio.run(
                run_benchmark(
                    suite_path=args.benchmark_suite,
                    scenario_ids=args.benchmark_scenario,
                    categories=args.benchmark_category,
                    backend=args.backend,
                    show_browser=args.show_browser,
                    slow_mo=args.slow_mo,
                    record_video=args.record_video,
                    export_formats=args.export_format,
                    two_stage_planning=args.two_stage_planning,
                    benchmark_runs=max(args.benchmark_runs, 1),
                    interaction_mode=args.interaction_mode,
                )
            )
        else:
            asyncio.run(
                run(
                    user_goal=args.goal,
                    backend=args.backend,
                    show_browser=args.show_browser,
                    slow_mo=args.slow_mo,
                    record_video=args.record_video,
                    export_formats=args.export_format,
                    two_stage_planning=args.two_stage_planning,
                    interaction_mode=args.interaction_mode,
                )
            )
    except LLMClientError as exc:
        raise SystemExit(
            "LLM backend error: no fallback was used, planning/verifying requires a working backend. "
            f"Details: {exc}"
        )
