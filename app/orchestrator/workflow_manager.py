from app.executor.playwright_executor import PlaywrightExecutor
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.page_snapshot import PageSnapshot
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


class WorkflowManager:
    def __init__(
        self,
        planner: Planner,
        validator: PlanValidator,
        executor: PlaywrightExecutor,
        verifier: LLMVerifier,
        replanner: Replanner | None = None,
        two_stage_planning: bool = False,
    ):
        self.planner = planner
        self.validator = validator
        self.executor = executor
        self.verifier = verifier
        self.replanner = replanner
        self.two_stage_planning = two_stage_planning

    async def run(self, user_goal: str):
        planning_mode = "two_stage" if self.two_stage_planning else "single_stage"
        initial_plan = None
        final_plan = None
        replanner_artifact = None

        if not self.two_stage_planning:
            plan = self.planner.build_plan(user_goal)
            self.validator.validate(plan)
            execution_result = await self.executor.execute(plan)
        else:
            if self.replanner is None:
                raise ValueError("two_stage_planning requires replanner")

            initial_plan = self.planner.build_initial_plan(user_goal)
            self.validator.validate(initial_plan)
            initial_execution = await self.executor.execute(initial_plan)
            if initial_execution.status != "success":
                verdict = self.verifier.verify(initial_plan, initial_execution)
                return {
                    "plan": initial_plan,
                    "initial_plan": initial_plan,
                    "final_plan": None,
                    "planning_mode": planning_mode,
                    "execution_result": initial_execution,
                    "verdict": verdict,
                    "planner_artifact": self.planner.last_artifact,
                    "initial_planner_artifact": self.planner.last_initial_artifact,
                    "replanner_artifact": None,
                    "verifier_artifact": self.verifier.last_artifact,
                }

            snapshot_payload = initial_execution.extracted_data.get("page_snapshot")
            if not snapshot_payload:
                raise ValueError("Initial plan did not produce 'page_snapshot'")
            page_snapshot = PageSnapshot.model_validate(snapshot_payload)

            final_plan = self.replanner.revise_plan(
                user_goal=user_goal,
                page_snapshot=page_snapshot,
                previous_plan=initial_plan,
            )
            replanner_artifact = self.replanner.last_artifact
            self.validator.validate(final_plan)
            execution_result = await self.executor.execute(final_plan)
            plan = final_plan

        verdict = self.verifier.verify(plan, execution_result)

        return {
            "plan": plan,
            "initial_plan": initial_plan,
            "final_plan": final_plan,
            "planning_mode": planning_mode,
            "execution_result": execution_result,
            "verdict": verdict,
            "planner_artifact": self.planner.last_artifact,
            "initial_planner_artifact": self.planner.last_initial_artifact,
            "replanner_artifact": replanner_artifact,
            "verifier_artifact": self.verifier.last_artifact,
        }
