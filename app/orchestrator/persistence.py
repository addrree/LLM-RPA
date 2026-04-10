from __future__ import annotations

import json
from pathlib import Path

from app.config import EXPORTS_DIR, LOGS_DIR, RAW_LLM_DIR, RESULTS_DIR
from app.exporters import CSVExporter, JSONExporter


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_artifacts(result: dict, run_id: str) -> dict:
    plan_path = RESULTS_DIR / f"plan_{run_id}.json"
    execution_path = RESULTS_DIR / f"execution_{run_id}.json"
    verdict_path = RESULTS_DIR / f"verdict_{run_id}.json"
    logs_path = LOGS_DIR / f"logs_{run_id}.json"

    plan_json = result["plan"].model_dump(mode="json")
    initial_plan = result.get("initial_plan")
    final_plan = result.get("final_plan")
    planning_mode = result.get("planning_mode", "single_stage")
    execution_json = result["execution_result"].model_dump(mode="json")
    verdict_json = result["verdict"].model_dump(mode="json")

    _write_json(plan_path, {
        "planning_mode": planning_mode,
        "plan": plan_json,
        "initial_plan": initial_plan.model_dump(mode="json") if initial_plan else None,
        "final_plan": final_plan.model_dump(mode="json") if final_plan else None,
    })
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

    replanner_artifact = result.get("replanner_artifact")
    if replanner_artifact is not None:
        _write_json(
            RAW_LLM_DIR / f"replanner_raw_{run_id}.json",
            {
                "raw_response": replanner_artifact.raw_response,
                "parsed_json": replanner_artifact.parsed_response,
                "generation_metadata": replanner_artifact.generation.model_dump(),
            },
        )

    initial_planner_artifact = result.get("initial_planner_artifact")
    if initial_planner_artifact is not None:
        _write_json(
            RAW_LLM_DIR / f"initial_planner_raw_{run_id}.json",
            {
                "raw_response": initial_planner_artifact.raw_response,
                "parsed_json": initial_planner_artifact.parsed_response,
                "generation_metadata": initial_planner_artifact.generation.model_dump(),
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
        "initial_planner_raw": RAW_LLM_DIR / f"initial_planner_raw_{run_id}.json" if initial_planner_artifact else None,
        "replanner_raw": RAW_LLM_DIR / f"replanner_raw_{run_id}.json" if replanner_artifact else None,
        "verifier_raw": RAW_LLM_DIR / f"verifier_raw_{run_id}.json" if verifier_artifact else None,
    }


def export_results(result: dict, run_id: str, export_formats: list[str]) -> list[Path]:
    extracted_data = result["execution_result"].extracted_data
    exported_page_snapshot = (
        extracted_data.get("page_snapshot")
        or result.get("page_snapshot")
        or (result.get("initial_execution_result").extracted_data.get("page_snapshot") if result.get("initial_execution_result") else None)
    )
    structured_output = {
        "planning_mode": result.get("planning_mode", "single_stage"),
        "initial_plan": result["initial_plan"].model_dump(mode="json") if result.get("initial_plan") else None,
        "final_plan": result["final_plan"].model_dump(mode="json") if result.get("final_plan") else None,
        "page_snapshot": exported_page_snapshot,
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
