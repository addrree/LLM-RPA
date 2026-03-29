import json

from app.planner.prompts import REPLANNER_SYSTEM_PROMPT
from app.schemas.execution import LLMArtifact
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Replanner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None

    def revise_plan(
        self,
        user_goal: str,
        page_snapshot: PageSnapshot,
        previous_plan: TaskSpec | None = None,
    ) -> TaskSpec:
        payload = {
            "user_goal": user_goal,
            "page_snapshot": page_snapshot.model_dump(mode="json"),
            "previous_plan": previous_plan.model_dump(mode="json") if previous_plan else None,
        }
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=REPLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        self.last_artifact = artifact
        return TaskSpec.model_validate(artifact.parsed_response)
