#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
import time
from collections import Counter
from app.browsergym_integration import BrowserGymAgentAdapter, BrowserGymRunConfig, BrowserGymRunner
from app.browsergym_integration.config import WEBARENA_REQUIRED_ENV_VARS
from app.browsergym_integration.webarena_tasks import discover_webarena_tasks
from app.main import build_llm_client
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


def compute_summary(rows: list[dict]) -> dict:
    attempted = [r for r in rows if r["status"] != "skipped"]
    success = [r for r in attempted if r["status"] in {"success", "success_by_agent_finish"}]
    failures = [r for r in attempted if r["status"] not in {"success", "success_by_agent_finish"}]
    failure_buckets = dict(Counter((r.get("failure_stage") or r["status"]) for r in failures))
    mean_steps = (sum(r.get("steps", 0) for r in attempted) / len(attempted)) if attempted else 0.0
    mean_runtime = (sum(r.get("runtime_sec", 0.0) for r in attempted) / len(attempted)) if attempted else 0.0
    return {
        "total_selected_tasks": len(rows),
        "attempted_tasks": len(attempted),
        "success_count": len(success),
        "success_rate": (len(success) / len(attempted)) if attempted else 0.0,
        "skipped_llm_judge_tasks": sum(1 for r in rows if r.get("skip_reason") == "llm_judge"),
        "mean_steps": mean_steps,
        "mean_runtime_sec": mean_runtime,
        "failure_buckets": failure_buckets,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ollama_cloud")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--site", default=None)
    p.add_argument("--output", default="artifacts/browsergym/webarena_deterministic_subset_report.json")
    p.add_argument("--skip-llm-fuzzy-match", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-vision", action="store_true", help="Send BrowserGym screenshot to the planner LLM payload")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    for mod in ("browsergym", "browsergym.webarena", "gymnasium"):
        __import__(mod)
    missing_env = [k for k in WEBARENA_REQUIRED_ENV_VARS if not __import__("os").getenv(k)]

    tasks, _ = discover_webarena_tasks()
    selected = []
    for t in tasks:
        if args.task_ids and t["task_id"] not in args.task_ids and t["env_id"] not in args.task_ids:
            continue
        if args.site and args.site not in t.get("sites", []):
            continue
        if args.skip_llm_fuzzy_match and t.get("requires_llm_judge"):
            selected.append({**t, "status": "skipped", "skip_reason": "llm_judge"})
            continue
        selected.append(t)
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    llm = build_llm_client(backend=args.backend)

    def agent_factory():
        return BrowserGymAgentAdapter(Planner(llm), Replanner(llm), PlanValidator(), LLMVerifier(llm), max_steps=args.max_steps, use_vision=args.use_vision)

    rows = []
    for task in selected:
        if task.get("status") == "skipped":
            rows.append(task)
            continue
        if missing_env:
            rows.append({**task, "status": "skipped", "failure_stage": "env_validation", "error_message": f"Missing WA vars: {', '.join(missing_env)}", "steps": 0, "runtime_sec": 0.0})
            continue
        started = time.time()
        report = BrowserGymRunner(
            agent_factory=agent_factory,
            config=BrowserGymRunConfig(env_id=task["env_id"], goal=task.get("intent") or task["task_id"], backend=args.backend, max_steps=args.max_steps, use_vision=args.use_vision),
        ).run_one()
        rows.append({
            **task,
            "status": report.status,
            "reward": report.reward,
            "terminated": report.terminated,
            "truncated": report.truncated,
            "failure_stage": report.failure_stage,
            "error_message": report.error_message,
            "steps": len(report.steps),
            "final_answer": report.final_answer,
            "runtime_sec": report.runtime_sec or (time.time() - started),
        })

    summary = compute_summary(rows)
    payload = {"ok": True, "backend": args.backend, "missing_env": missing_env, "summary": summary, "results": rows}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    fields = ["task_id", "env_id", "status", "reward", "terminated", "truncated", "failure_stage", "error_message", "steps", "runtime_sec"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    print(json.dumps({"ok": True, "output": str(out), "csv": str(csv_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
