import asyncio
import json

from app.executor.playwright_executor import PlaywrightExecutor
from app.orchestrator.workflow_manager import WorkflowManager
from app.planner.planner import Planner
from app.utils.llm_client import LLMClient
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


class DummyLLMClient(LLMClient):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        # Временная заглушка.
        # Сначала с ней прогоняешь pipeline.
        if "модуль верификации" in system_prompt.lower():
            return {
                "task_completed": True,
                "confidence": 0.85,
                "verdict": "accept",
                "issues": [],
                "summary": "Result looks relevant to the task."
            }

        return {
            "goal": user_prompt,
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {
                "max_steps": 6,
                "max_replans": 1,
                "timeout_sec": 20
            },
            "expected_result": {
                "description": "Extract page heading text",
                "required_fields": ["heading"]
            },
            "steps": [
                {
                    "step_id": 1,
                    "action": "open_url",
                    "args": {"url": "https://example.com"}
                },
                {
                    "step_id": 2,
                    "action": "extract_text",
                    "args": {"selector": "h1"},
                    "save_as": "heading"
                },
                {
                    "step_id": 3,
                    "action": "screenshot",
                    "args": {}
                },
                {
                    "step_id": 4,
                    "action": "finish",
                    "args": {}
                }
            ]
        }


async def main():
    llm_client = DummyLLMClient()

    workflow = WorkflowManager(
        planner=Planner(llm_client),
        validator=PlanValidator(),
        executor=PlaywrightExecutor(),
        verifier=LLMVerifier(llm_client),
    )

    user_goal = "Открой example.com и выгрузи главный заголовок страницы"

    result = await workflow.run(user_goal)

    print("\nPLAN:")
    print(result["plan"].model_dump_json(indent=2))

    print("\nEXECUTION RESULT:")
    print(result["execution_result"].model_dump_json(indent=2))

    print("\nVERDICT:")
    print(result["verdict"].model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())