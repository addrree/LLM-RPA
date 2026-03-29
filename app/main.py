import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.config import EXPORTS_DIR, LOGS_DIR, RAW_LLM_DIR, RESULTS_DIR
from app.executor.playwright_executor import PlaywrightExecutor
from app.exporters import CSVExporter, JSONExporter
from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.planner import Planner
from app.utils.llm_client import DummyLLMClient, LLMClient, LLMClientError
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier

UTC = timezone.utc


def build_llm_client(force_dummy: bool = False, backend: str | None = None):
    if force_dummy:
        return DummyLLMClient()

    selected_backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
    if selected_backend == "dummy":
        return DummyLLMClient()

    return LLMClient(
        backend=selected_backend,
        planner_model=os.getenv("OLLAMA_PLANNER_MODEL", os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")),
        verifier_model=os.getenv("OLLAMA_VERIFIER_MODEL", os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifacts(result: dict, run_id: str) -> dict:
    plan_path = RESULTS_DIR / f"plan_{run_id}.json"
    execution_path = RESULTS_DIR / f"execution_{run_id}.json"
    verdict_path = RESULTS_DIR / f"verdict_{run_id}.json"
    logs_path = LOGS_DIR / f"logs_{run_id}.json"

    plan_json = result["plan"].model_dump(mode="json")
    execution_json = result["execution_result"].model_dump(mode="json")
    verdict_json = result["verdict"].model_dump(mode="json")

    _write_json(plan_path, plan_json)
    _write_json(execution_path, execution_json)
    _write_json(verdict_path, verdict_json)
    _write_json(logs_path, {"logs": execution_json.get("logs", [])})

    planner_artifact = result.get("planner_artifact")
    if planner_artifact is not None:
        _write_json(
            RAW_LLM_DIR / f"planner_raw_{run_id}.json",
            {
                "raw_response": planner_artifact.raw_response,
                "parsed_json": planner_artifact.parsed_response,
                "generation_metadata": planner_artifact.generation.model_dump(),
            },
        )

    verifier_artifact = result.get("verifier_artifact")
    if verifier_artifact is not None:
        _write_json(
            RAW_LLM_DIR / f"verifier_raw_{run_id}.json",
            {
                "raw_response": verifier_artifact.raw_response,
                "parsed_json": verifier_artifact.parsed_response,
                "generation_metadata": verifier_artifact.generation.model_dump(),
            },
        )

    return {
        "plan": plan_path,
        "execution": execution_path,
        "verdict": verdict_path,
        "logs": logs_path,
        "planner_raw": RAW_LLM_DIR / f"planner_raw_{run_id}.json" if planner_artifact else None,
        "verifier_raw": RAW_LLM_DIR / f"verifier_raw_{run_id}.json" if verifier_artifact else None,
    }


def export_results(result: dict, run_id: str, export_formats: list[str]) -> list[Path]:
    extracted_data = result["execution_result"].extracted_data
    structured_output = {
        "status": result["execution_result"].status,
        "verdict": result["verdict"].model_dump(mode="json"),
        "final_url": result["execution_result"].final_url,
        "screenshot_path": result["execution_result"].screenshot_path,
    }

    exporters = {
        "json": JSONExporter(EXPORTS_DIR),
        "csv": CSVExporter(EXPORTS_DIR),
    }

    paths = []
    for export_format in export_formats:
        exporter = exporters[export_format]
        paths.append(
            exporter.export(
                run_id=run_id,
                extracted_data=extracted_data,
                structured_output=structured_output,
            )
        )
    return paths


async def run(
    user_goal: str,
    force_dummy: bool = False,
    backend: str | None = None,
    show_browser: bool = False,
    slow_mo: int = 0,
    record_video: bool = False,
    export_formats: list[str] | None = None,
):
    export_formats = export_formats or ["json"]
    export_formats = list(dict.fromkeys(export_formats))
    llm_client = build_llm_client(force_dummy=force_dummy, backend=backend)

    workflow = WorkflowManager(
        planner=Planner(llm_client),
        validator=PlanValidator(),
        executor=PlaywrightExecutor(headless=not show_browser, slow_mo=slow_mo, record_video=record_video),
        verifier=LLMVerifier(llm_client),
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
        "extracted_keys": sorted(list(result["execution_result"].extracted_data.keys())),
    }, ensure_ascii=False, indent=2))

    print("\nARTIFACTS:")
    for name, path in artifact_paths.items():
        if path:
            print(f"- {name}: {path}")

    print("\nEXPORTS:")
    for path in export_paths:
        print(f"- {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM-RPA MVP pipeline")
    parser.add_argument(
        "--goal",
        default="Open https://www.wikipedia.org, extract the h1 text, take screenshot and finish.",
        help="User goal in natural language",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Force DummyLLMClient",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "dummy"],
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
        "--export-format",
        action="append",
        choices=["json", "csv"],
        default=None,
        help="Export format for workflow result (can be specified multiple times)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    try:
        asyncio.run(
            run(
                user_goal=args.goal,
                force_dummy=args.dummy,
                backend=args.backend,
                show_browser=args.show_browser,
                slow_mo=args.slow_mo,
                record_video=args.record_video,
                export_formats=args.export_format,
            )
        )
    except LLMClientError as exc:
        raise SystemExit(
            "LLM backend error: no fallback was used, planning/verifying requires a working backend. "
            f"Details: {exc}"
        )
