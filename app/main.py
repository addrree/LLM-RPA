import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from app.config import LOGS_DIR, RESULTS_DIR
from app.executor.playwright_executor import PlaywrightExecutor
from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.planner import Planner
from app.utils.llm_client import DummyLLMClient, LLMClient, LLMClientError
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier

UTC = timezone.utc


def build_llm_client(force_dummy: bool = False):
    if force_dummy:
        return DummyLLMClient()

    try:
        return LLMClient(
            planner_model=os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.0-flash"),
            verifier_model=os.getenv("GEMINI_VERIFIER_MODEL", "gemini-2.0-flash"),
        )
    except LLMClientError as exc:
        print(f"[WARN] {exc} Falling back to DummyLLMClient.")
        return DummyLLMClient()


def save_artifacts(result: dict) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    plan_path = RESULTS_DIR / f"plan_{timestamp}.json"
    execution_path = RESULTS_DIR / f"execution_{timestamp}.json"
    verdict_path = RESULTS_DIR / f"verdict_{timestamp}.json"
    logs_path = LOGS_DIR / f"logs_{timestamp}.json"

    plan_json = result["plan"].model_dump(mode="json")
    execution_json = result["execution_result"].model_dump(mode="json")
    verdict_json = result["verdict"].model_dump(mode="json")

    plan_path.write_text(json.dumps(plan_json, ensure_ascii=False, indent=2), encoding="utf-8")
    execution_path.write_text(json.dumps(execution_json, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict_path.write_text(json.dumps(verdict_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logs_path.write_text(json.dumps(execution_json.get("logs", []), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nARTIFACTS:")
    print(f"- Plan: {plan_path}")
    print(f"- Execution: {execution_path}")
    print(f"- Verdict: {verdict_path}")
    print(f"- Logs: {logs_path}")


async def run(user_goal: str, force_dummy: bool = False):
    llm_client = build_llm_client(force_dummy=force_dummy)

    workflow = WorkflowManager(
        planner=Planner(llm_client),
        validator=PlanValidator(),
        executor=PlaywrightExecutor(),
        verifier=LLMVerifier(llm_client),
    )

    result = await workflow.run(user_goal)

    print("\nPLAN:")
    print(result["plan"].model_dump_json(indent=2))

    print("\nEXECUTION RESULT:")
    print(result["execution_result"].model_dump_json(indent=2))

    print("\nVERDICT:")
    print(result["verdict"].model_dump_json(indent=2))

    save_artifacts(result)


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
        help="Force DummyLLMClient instead of Gemini API",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    asyncio.run(run(user_goal=args.goal, force_dummy=args.dummy))
