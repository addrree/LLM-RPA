from app.executor.playwright_executor import PlaywrightExecutor
from app.planner.planner import Planner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


class WorkflowManager:
    def __init__(
        self,
        planner: Planner,
        validator: PlanValidator,
        executor: PlaywrightExecutor,
        verifier: LLMVerifier,
    ):
        self.planner = planner
        self.validator = validator
        self.executor = executor
        self.verifier = verifier

    async def run(self, user_goal: str):
        plan = self.planner.build_plan(user_goal)
        self.validator.validate(plan)

        execution_result = await self.executor.execute(plan)
        verdict = self.verifier.verify(plan, execution_result)

        return {
            "plan": plan,
            "execution_result": execution_result,
            "verdict": verdict,
            "planner_artifact": self.planner.last_artifact,
            "verifier_artifact": self.verifier.last_artifact,
        }
