#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration import BrowserGymAgentAdapter, BrowserGymRunConfig, BrowserGymRunner
from app.browsergym_integration.miniwob_tasks import ACTION_COMPLEX_MINIWOB_TASK_NAMES, EXTRACTION_TEXT_MINIWOB_TASK_NAMES, EXTRACTION_MINIWOB_TASK_NAMES, VISUAL_SPATIAL_MINIWOB_TASK_NAMES, list_minwob_env_ids, select_minwob_subset, task_name_from_env_id
from app.main import build_llm_client
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier

SUITE_ID = "browsergym_miniwob_subset_v1"
DEFAULT_GOAL = "Complete the MiniWoB task according to the page instruction"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args(argv=None):
    ts = _timestamp()
    parser = argparse.ArgumentParser(description="Run a BrowserGym MiniWoB++ subset")
    parser.add_argument("--backend", default=os.getenv("LLM_BACKEND", "ollama_cloud"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", default=None, help="Comma-separated full env IDs or MiniWoB task names")
    parser.add_argument("--subset", choices=["action", "extraction", "visual", "action-complex", "basic", "complex", "all"], default=None, help="action/basic excludes book-flight; extraction uses extraction tasks; visual uses visual-spatial tasks; complex includes only book-flight; all includes every selected task")
    parser.add_argument("--include", default=None, help="Comma-separated regex patterns to include")
    parser.add_argument("--exclude", default=None, help="Comma-separated regex patterns to exclude")
    parser.add_argument("--use-vision", action="store_true", help="Send BrowserGym screenshot to the planner LLM payload")
    parser.add_argument("--output-json", default=f"artifacts/browsergym/miniwob_results_{ts}.json")
    parser.add_argument("--output-csv", default=f"artifacts/browsergym/miniwob_results_{ts}.csv")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Print per-task and per-step MiniWoB progress")
    parser.add_argument("--allow-playwright-fallback", action="store_true", help="MiniWoB-only fallback: try direct Playwright page.mouse.click(page_center) after failed scaled mouse_click")
    parser.add_argument("--allow-extraction-llm-fallback", action="store_true", help="Allow MiniWoB extraction tasks to fallback to LLM when extraction controller has no decision")
    parser.add_argument("--task-timeout-sec", type=float, default=None, help="Per-task max runtime in seconds")
    return parser.parse_args(argv)


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def result_from_report(report, *, env_id: str, use_vision: bool) -> dict[str, Any]:
    steps = list(getattr(report, "steps", []) or [])
    reward = getattr(report, "reward", None)
    success = reward is not None and float(reward) > 0
    failure_stage = getattr(report, "failure_stage", None)
    status = getattr(report, "status", "unknown")
    if not success and not failure_stage:
        if any(getattr(step, "mapping_error", None) or getattr(step, "rationale", None) == "action mapping failure" for step in steps):
            failure_stage = "action_mapping_failure"
        else:
            failure_stage = status or "unsuccessful"
    return {
        "env_id": env_id,
        "task_name": task_name_from_env_id(env_id),
        "status": status,
        "reward": float(reward) if reward is not None else None,
        "success": bool(success),
        "terminated": bool(getattr(report, "terminated", False)),
        "truncated": bool(getattr(report, "truncated", False)),
        "steps_count": getattr(report, "steps_count", None) if getattr(report, "steps_count", None) is not None else len(steps),
        "runtime_sec": float(getattr(report, "runtime_sec", 0.0) or 0.0),
        "failure_stage": failure_stage,
        "error_message": getattr(report, "error_message", None),
        "final_answer": getattr(report, "final_answer", None),
        "vision_used": bool(use_vision or any(bool(getattr(step, "vision_used", False)) for step in steps)),
        "vision_image_present": bool(any(bool(getattr(step, "vision_image_present", False)) for step in steps)),
        "extraction_intent": next((((getattr(step, "mapping_diagnostics", {}) or {}).get("extraction_intent")) for step in steps if isinstance(getattr(step, "mapping_diagnostics", None), dict) and (getattr(step, "mapping_diagnostics", {}) or {}).get("extraction_intent")), None),
        "extracted_answer": next((getattr(step, "extracted_value", None) for step in steps if getattr(step, "extracted_value", None)), None),
        "extracted_data": next(((getattr(step, "mapping_diagnostics", {}) or {}).get("extracted_data") for step in steps if isinstance(getattr(step, "mapping_diagnostics", None), dict) and (getattr(step, "mapping_diagnostics", {}) or {}).get("extracted_data") is not None), None),
        "strategy": next((getattr(step, "mapping_strategy", None) for step in steps if getattr(step, "mapping_strategy", None)), None),
        "confidence": next(((getattr(step, "mapping_diagnostics", {}) or {}).get("confidence") for step in steps if isinstance(getattr(step, "mapping_diagnostics", None), dict) and (getattr(step, "mapping_diagnostics", {}) or {}).get("confidence") is not None), None),
        "action_taken": bool(steps),
        "selected_candidate_text": next((((getattr(step, "selected_candidate", None) or {}).get("text")) for step in steps if isinstance(getattr(step, "selected_candidate", None), dict)), None),
        "failure_reason": failure_stage,
        "output_path": getattr(report, "output_path", None),
        "steps": [
            {
                "step_idx": getattr(step, "step_idx", None),
                "action": getattr(step, "action", None),
                "action_string": getattr(step, "action_string", None),
                "rationale": getattr(step, "action_rationale", None) or getattr(step, "rationale", None),
                "action_rationale": getattr(step, "action_rationale", None) or getattr(step, "rationale", None),
                "reward": getattr(step, "reward", None),
                "terminated": getattr(step, "terminated", None),
                "truncated": getattr(step, "truncated", None),
                "current_url": getattr(step, "url", None),
                "mapping_error": getattr(step, "mapping_error", None),
                "miniwob_instruction": getattr(step, "miniwob_instruction", None),
                "selected_candidate": getattr(step, "selected_candidate", None),
                "selected_candidate_verbose": getattr(step, "selected_candidate_verbose", None),
                "selected_candidate_bid": (getattr(step, "selected_candidate", None) or {}).get("bid") if isinstance(getattr(step, "selected_candidate", None), dict) else None,
                "bid_source": (getattr(step, "selected_candidate", None) or {}).get("bid_source") if isinstance(getattr(step, "selected_candidate", None), dict) else None,
                "mapping_strategy": getattr(step, "mapping_strategy", None),
                "mapping_diagnostics": getattr(step, "mapping_diagnostics", None),
                "action_string_before_mapping": getattr(step, "action_string_before_mapping", None),
                "action_string_after_mapping": getattr(step, "action_string_after_mapping", None),
                "fallback_used": bool(getattr(step, "fallback_used", False)),
                "fallback_type": getattr(step, "fallback_type", None),
                "fallback_reward": getattr(step, "fallback_reward", None),
                "fallback_terminated": getattr(step, "fallback_terminated", None),
            }
            for step in steps
        ],
    }


def build_aggregate(results: list[dict[str, Any]], *, use_vision: bool, generated_at: str | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    total = len(results)
    success_count = sum(1 for result in results if result.get("success"))
    rewards = [float(result["reward"]) for result in results if result.get("reward") is not None]
    steps = [float(result["steps_count"]) for result in results if result.get("steps_count") is not None]
    runtimes = [float(result["runtime_sec"]) for result in results if result.get("runtime_sec") is not None]
    failure_buckets = Counter(
        str(result.get("failure_stage") or result.get("status") or "unsuccessful")
        for result in results
        if not result.get("success")
    )
    return {
        "suite_id": SUITE_ID,
        "generated_at": generated_at,
        "total_tasks": total,
        "success_count": success_count,
        "success_rate": (success_count / total) if total else 0.0,
        "mean_reward": _mean_or_none(rewards),
        "mean_steps": _mean_or_none(steps),
        "mean_runtime_sec": _mean_or_none(runtimes),
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "use_vision": use_vision,
        "results": results,
    }


def write_outputs(aggregate: dict[str, Any], output_json: str | Path, output_csv: str | Path) -> tuple[Path, Path]:
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "env_id",
        "task_name",
        "status",
        "reward",
        "success",
        "terminated",
        "truncated",
        "steps_count",
        "runtime_sec",
        "failure_stage",
        "error_message",
        "final_answer",
        "vision_used",
        "vision_image_present",
        "output_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in aggregate.get("results", []):
            writer.writerow({name: result.get(name) for name in fieldnames})
    return json_path, csv_path


def skipped_result(env_id: str, message: str, *, use_vision: bool) -> dict[str, Any]:
    return {
        "env_id": env_id,
        "task_name": task_name_from_env_id(env_id),
        "status": "skipped",
        "reward": None,
        "success": False,
        "terminated": False,
        "truncated": False,
        "steps_count": 0,
        "runtime_sec": 0.0,
        "failure_stage": "env_validation",
        "error_message": message,
        "final_answer": None,
        "vision_used": use_vision,
        "vision_image_present": False,
        "output_path": None,
    }

def _suite_type(subset: str | None) -> str:
    if subset == "extraction":
        return "miniwob_extraction"
    if subset == "visual":
        return "miniwob_visual_spatial"
    return "miniwob_action"


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.task_timeout_sec is None:
        if args.subset == "extraction":
            args.task_timeout_sec = 90
        elif args.subset in {"complex", "action-complex"}:
            args.task_timeout_sec = 300
        else:
            args.task_timeout_sec = 180
    env_ids = list_minwob_env_ids()
    task_ids = args.task_ids
    excluded_tasks = [part.strip() for part in str(args.exclude or "").split(",") if part.strip()]
    requested_task_ids: list[str] = []
    if args.subset in {"basic", "action"} and not task_ids:
        task_ids = ",".join(t for t in [
            "click-button","click-button-sequence","click-checkboxes","click-dialog","click-link","click-menu","click-option","click-test","enter-text","focus-text","login-user","choose-list","choose-date","use-autocomplete"
        ])
    if args.subset == "extraction" and not task_ids:
        task_ids = ",".join(EXTRACTION_TEXT_MINIWOB_TASK_NAMES)
    if args.subset == "action-complex" and not task_ids:
        task_ids = ",".join(ACTION_COMPLEX_MINIWOB_TASK_NAMES)
    if args.subset == "complex" and not task_ids:
        task_ids = "book-flight"
    if args.subset == "visual" and not task_ids:
        task_ids = ",".join(VISUAL_SPATIAL_MINIWOB_TASK_NAMES)
    if args.subset == "all" and not task_ids:
        task_ids = None
    if task_ids:
        requested_task_ids = [part.strip() for part in str(task_ids).split(",") if part.strip()]
    effective_limit = args.limit
    if effective_limit is None and args.subset != "extraction":
        effective_limit = 10
    selected = select_minwob_subset(
        env_ids,
        limit=effective_limit,
        task_ids=task_ids,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )
    available_envs = set(env_ids)
    requested_envs = [rid if rid.startswith("browsergym/miniwob.") else f"browsergym/miniwob.{rid}" for rid in requested_task_ids]
    missing_task_ids = [rid for rid in requested_envs if rid not in available_envs]
    dropped_by_limit_task_ids = [rid for rid in requested_envs if rid in available_envs and rid not in selected]
    if any(task_name_from_env_id(env_id) == "book-flight" for env_id in selected) and args.max_steps < 20:
        print("[MiniWoB] warning: book-flight is complex and should be run with max_steps >= 20 or --subset complex.", flush=True)
        args.max_steps = 25
    if any(task_name_from_env_id(env_id) == "click-checkboxes" for env_id in selected) and args.max_steps < 5:
        print("[MiniWoB] warning: click-checkboxes benefits from max_steps >= 5; raising to 5", flush=True)
        args.max_steps = 5
    if any(task_name_from_env_id(env_id) in {"use-autocomplete", "choose-date"} for env_id in selected) and args.max_steps < 6:
        print("[MiniWoB] warning: choose-date/use-autocomplete benefit from max_steps >= 6; raising to 6", flush=True)
        args.max_steps = 6

    if not os.getenv("MINIWOB_URL"):
        message = "MINIWOB_URL is not set. Start MiniWoB++ HTTP server and set MINIWOB_URL=http://127.0.0.1:8765 before running tasks."
        placeholder_ids = selected or ["browsergym/miniwob.unavailable"]
        aggregate = build_aggregate([skipped_result(env_id, message, use_vision=args.use_vision) for env_id in placeholder_ids], use_vision=args.use_vision)
        aggregate.update({"subset": args.subset or "default", "suite_type": _suite_type(args.subset), "excluded_tasks": excluded_tasks, "requested_task_ids": requested_envs, "effective_task_ids": list(selected), "missing_task_ids": missing_task_ids, "dropped_by_limit_task_ids": dropped_by_limit_task_ids, "total_tasks": len(selected)})
        json_path, csv_path = write_outputs(aggregate, args.output_json, args.output_csv)
        print(message)
        print(json.dumps({"status": "skipped", "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
        return 0

    if not selected:
        message = "No BrowserGym MiniWoB env IDs were registered. Install browsergym-miniwob and verify the package imports."
        aggregate = build_aggregate([skipped_result("browsergym/miniwob.unavailable", message, use_vision=args.use_vision)], use_vision=args.use_vision)
        aggregate.update({"subset": args.subset or "default", "suite_type": _suite_type(args.subset), "excluded_tasks": excluded_tasks, "requested_task_ids": requested_envs, "effective_task_ids": list(selected), "missing_task_ids": missing_task_ids, "dropped_by_limit_task_ids": dropped_by_limit_task_ids, "total_tasks": len(selected)})
        json_path, csv_path = write_outputs(aggregate, args.output_json, args.output_csv)
        print(message)
        print(json.dumps({"status": "skipped", "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
        return 0

    llm = build_llm_client(backend=args.backend)

    def agent_factory():
        agent = BrowserGymAgentAdapter(
            planner=Planner(llm),
            replanner=Replanner(llm),
            validator=PlanValidator(),
            verifier=LLMVerifier(llm),
            max_steps=args.max_steps,
            use_vision=args.use_vision,
        )
        agent.allow_extraction_llm_fallback = bool(args.allow_extraction_llm_fallback)
        return agent

    results = []
    total_selected = len(selected)
    for task_idx, env_id in enumerate(selected, start=1):
        if args.verbose:
            print(f"[MiniWoB] task {task_idx}/{total_selected} {env_id} started", flush=True)
        started = time.time()
        try:
            report = BrowserGymRunner(
                agent_factory=agent_factory,
                config=BrowserGymRunConfig(
                    env_id=env_id,
                    goal=DEFAULT_GOAL,
                    backend=args.backend,
                    max_steps=args.max_steps,
                    use_vision=args.use_vision,
                    headless=args.headless,
                    benchmark="miniwob",
                    task_name=task_name_from_env_id(env_id),
                    allow_playwright_fallback=args.allow_playwright_fallback,
                    task_timeout_sec=args.task_timeout_sec,
                    allow_extraction_llm_fallback=args.allow_extraction_llm_fallback,
                ),
            ).run_one()
            result = result_from_report(report, env_id=env_id, use_vision=args.use_vision)
            if args.verbose:
                for step in result.get("steps", []):
                    step_no = (step.get("step_idx") if step.get("step_idx") is not None else 0) + 1
                    print(f"[MiniWoB] step {step_no}/{args.max_steps} action={step.get('action_string') or step.get('action')} before_grounding={step.get('action_string_before_mapping')} after_grounding={step.get('action_string_after_mapping')} mapping_strategy={step.get('mapping_strategy')}", flush=True)
                    print(f"[MiniWoB] step {step_no} selected_candidate.bid={step.get('selected_candidate_bid')} bid_source={step.get('bid_source')} selected_candidate={json.dumps(step.get('selected_candidate'), ensure_ascii=False, default=str)} selected_candidate_verbose={json.dumps(step.get('selected_candidate_verbose'), ensure_ascii=False, default=str)}", flush=True)
                    print(f"[MiniWoB] step {step_no} reward={step.get('reward')} terminated={step.get('terminated')} truncated={step.get('truncated')}", flush=True)
                print(f"[MiniWoB] task done success={result.get('success')} reward={result.get('reward')}", flush=True)
            results.append(result)
        except Exception as exc:
            result = {
                **skipped_result(env_id, str(exc), use_vision=args.use_vision),
                "status": "failed",
                "runtime_sec": time.time() - started,
                "failure_stage": "runtime",
            }
            if args.verbose:
                print(f"[MiniWoB] task done success=False reward=None", flush=True)
            results.append(result)

    aggregate = build_aggregate(results, use_vision=args.use_vision)
    aggregate.update({
        "subset": args.subset or "default",
        "suite_type": _suite_type(args.subset),
        "requested_task_ids": requested_envs,
        "excluded_tasks": excluded_tasks,
        "effective_task_ids": list(selected),
        "missing_task_ids": missing_task_ids,
        "dropped_by_limit_task_ids": dropped_by_limit_task_ids,
        "total_tasks": len(selected),
    })
    json_path, csv_path = write_outputs(aggregate, args.output_json, args.output_csv)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "summary": {k: aggregate[k] for k in ["total_tasks", "success_count", "success_rate", "mean_reward", "mean_steps", "mean_runtime_sec", "failure_buckets"]}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
