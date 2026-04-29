from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.benchmark.runner import BenchmarkRunner, BenchmarkSelection, write_benchmark_report
from app.benchmark.scenario_loader import ScenarioSuite
from app.main import build_llm_client, build_workflow
from app.webarena import WebArenaTaskAdapter, load_webarena_tasks


def build_suite_from_webarena(*, input_path: Path, suite_id: str, category: str) -> ScenarioSuite:
    tasks = load_webarena_tasks(input_path)
    scenarios = [WebArenaTaskAdapter.to_scenario(task, category=category) for task in tasks]
    return ScenarioSuite(
        suite_id=suite_id,
        description=f"Generated from WebArena-like input: {input_path}",
        scenarios=scenarios,
    )


async def run(args) -> None:
    llm_client = build_llm_client(force_dummy=args.dummy, backend=args.backend)
    suite = build_suite_from_webarena(
        input_path=args.input,
        suite_id=args.suite_id,
        category=args.category,
    )

    runner = BenchmarkRunner(
        workflow_factory=lambda: build_workflow(
            llm_client=llm_client,
            show_browser=args.show_browser,
            slow_mo=args.slow_mo,
            record_video=args.record_video,
            two_stage_planning=args.two_stage_planning,
        ),
        export_formats=args.export_format,
    )

    report = await runner.run_suite(
        suite,
        selection=BenchmarkSelection(
            scenario_ids=args.scenario_id,
            categories=args.benchmark_category,
        ),
    )
    report_json, report_csv = write_benchmark_report(report)

    print("\nWEBARENA-LIKE BENCHMARK SUMMARY:")
    print(json.dumps(report.metrics.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("\nWEBARENA-LIKE BENCHMARK REPORTS:")
    print(f"- json: {report_json}")
    print(f"- csv: {report_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run WebArena-like tasks through LLM-RPA benchmark pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Path to WebArena-like tasks JSON")
    parser.add_argument("--suite-id", default="webarena_like_suite", help="Generated suite_id")
    parser.add_argument(
        "--category",
        default="navigation_then_extraction",
        choices=[
            "single_value_extraction",
            "anchored_value_extraction",
            "repeated_structured_items",
            "navigation_then_extraction",
            "multi_step_information_retrieval",
            "negative_or_ambiguous_case",
        ],
    )

    parser.add_argument("--backend", choices=["ollama", "ollama_cloud", "dummy"], default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--two-stage-planning", action="store_true")
    parser.add_argument("--export-format", action="append", choices=["json", "csv"], default=["json"])

    parser.add_argument("--scenario-id", action="append", default=None)
    parser.add_argument("--benchmark-category", action="append", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
