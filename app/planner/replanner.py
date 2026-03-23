from app.schemas.execution import ExecutionResult
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Replanner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def revise_plan(self, user_goal: str, old_plan: TaskSpec, result: ExecutionResult) -> TaskSpec:
        raise NotImplementedError("Implement re-planning after base workflow works.")