from app.planner.prompts import PLANNER_SYSTEM_PROMPT
from app.schemas.execution import LLMArtifact
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None

    def build_plan(self, user_goal: str) -> TaskSpec:
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_goal,
        )
        self.last_artifact = artifact
        return TaskSpec.model_validate(artifact.parsed_response)
