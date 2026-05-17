from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.main import build_llm_client, build_workflow

TASKS = [
    ("click-button", "Click the 'Click me' button"),
    ("click-button-sequence", "Click 'One' then click 'Two'"),
    ("click-checkboxes", "Check the Accept checkbox"),
    ("click-dialog", "Open dialog and click OK"),
    ("click-link", "Click the Continue link"),
    ("click-menu", "Open Menu and click Settings"),
    ("click-option-radio", "Click the Blue radio option"),
    ("enter-text", "Enter 'Alice' into the Name field and submit"),
    ("login-user", "Login with username 'alice' and password 'secret'"),
    ("choose-list-select", "Choose Paris from the list"),
    ("choose-date", "Choose date 2026-05-17"),
    ("use-autocomplete", "Type 'App' and select Apple autocomplete suggestion"),
    ("book-flight", "Book flight from SFO to JFK on 2026-05-17"),
]


def bucket(result: dict) -> str:
    execution = result["execution_result"]
    if execution.status != "success":
        return execution.failure_type or "execution_failed"
    state = execution.extracted_data.get("result_state", {})
    if not state.get("success"):
        return "partial"
    return "success"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run internal MiniWoB-like tasks through main observe_action pipeline")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures/web_tasks"))
    parser.add_argument("--backend", choices=["dummy", "ollama", "ollama_cloud"], default="dummy")
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()

    llm = build_llm_client(force_dummy=args.backend == "dummy", backend=args.backend)
    workflow = build_workflow(
        llm_client=llm,
        show_browser=args.show_browser,
        slow_mo=0,
        record_video=False,
        two_stage_planning=False,
        interaction_mode="observe_action",
    )
    rows = []
    for task_id, instruction in TASKS:
        path = (args.fixtures_dir / f"{task_id}.html").resolve()
        goal = f"{instruction}. Open {path.as_uri()}"
        result = await workflow.run(goal, benchmark_context={"max_steps": 8})
        execution = result["execution_result"]
        state = execution.extracted_data.get("result_state", {})
        rows.append({
            "task_id": task_id,
            "success": bool(state.get("success")),
            "steps": len(execution.logs),
            "failure_bucket": bucket(result),
            "failure_stage": execution.failed_action or ("execution" if execution.status != "success" else ""),
            "final_url": execution.final_url,
            "extracted": state,
        })
    success_count = sum(1 for row in rows if row["success"])
    summary = {
        "total_tasks": len(rows),
        "success_count": success_count,
        "success_rate": round(success_count / len(rows), 4) if rows else 0.0,
        "mean_steps": round(sum(row["steps"] for row in rows) / len(rows), 3) if rows else 0.0,
        "failure_bucket": {name: sum(1 for row in rows if row["failure_bucket"] == name) for name in sorted({row["failure_bucket"] for row in rows})},
        "tasks": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
