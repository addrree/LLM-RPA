from app.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def build_plan(self, user_goal: str) -> TaskSpec:
        raw_json = self.llm_client.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_goal,
        )
        return TaskSpec.model_validate(raw_json)