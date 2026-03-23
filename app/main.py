import argparse
import asyncio
import os

from dotenv import load_dotenv

from app.executor.playwright_executor import PlaywrightExecutor
from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.planner import Planner
from app.utils.llm_client import DummyLLMClient, LLMClient, LLMClientError
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM-RPA MVP pipeline")
    parser.add_argument(
        "--goal",
        default="Open https://example.com, extract the h1 text, take screenshot and finish.",
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
