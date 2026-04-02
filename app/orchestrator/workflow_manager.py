import json
from datetime import datetime, timezone

from app.config import RAW_LLM_DIR
from app.executor.playwright_executor import PlaywrightExecutor
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator
from app.verifier.llm_verifier import LLMVerifier

UTC = timezone.utc


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
        initial_execution_result = None
        shared_page_snapshot = None

        if not self.two_stage_planning:
            plan = self.planner.build_plan(user_goal)
            self.validator.validate(plan)
            execution_result = await self.executor.execute(plan)
        else:
            if self.replanner is None:
                raise ValueError("two_stage_planning requires replanner")

            initial_plan = self.planner.build_initial_plan(user_goal)
            self.validator.validate(initial_plan)
            session = await self.executor._start_session()
            shared_runtime_state = {}
            try:
                initial_execution = await self.executor.execute(
                    initial_plan,
                    session=session,
                    runtime_state=shared_runtime_state,
                )
                initial_execution_result = initial_execution
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
                        "initial_execution_result": initial_execution_result,
                        "page_snapshot": shared_page_snapshot,
                    }

                snapshot_payload = initial_execution.extracted_data.get("page_snapshot")
                if not snapshot_payload:
                    snapshot_payload = shared_runtime_state.get("last_page_snapshot")
                if not snapshot_payload:
                    raise ValueError("Initial plan did not produce 'page_snapshot'")
                page_snapshot = PageSnapshot.model_validate(snapshot_payload)
                shared_page_snapshot = page_snapshot.model_dump(mode="json")

                final_plan = self.replanner.revise_plan(
                    user_goal=user_goal,
                    page_snapshot=page_snapshot,
                    previous_plan=initial_plan,
                )
                replanner_artifact = self.replanner.last_artifact
                final_plan = self._ensure_open_url_for_final_plan(final_plan)
                try:
                    self.validator.validate(final_plan)
                except PlanValidationError as first_error:
                    invalid_plan_dump = final_plan.model_dump(mode="json")
                    repaired_plan = self.replanner.revise_plan(
                        user_goal=user_goal,
                        page_snapshot=page_snapshot,
                        previous_plan=initial_plan,
                        validation_error=str(first_error),
                        invalid_plan=invalid_plan_dump,
                    )
                    replanner_artifact = self.replanner.last_artifact
                    final_plan = self._ensure_open_url_for_final_plan(repaired_plan)
                    try:
                        self.validator.validate(final_plan)
                    except PlanValidationError as second_error:
                        self._persist_final_plan_repair_failure(
                            invalid_plan=invalid_plan_dump,
                            validation_error=str(second_error),
                            repaired_raw_response=replanner_artifact.raw_response if replanner_artifact else None,
                        )
                        raise
                execution_result = await self.executor.execute(
                    final_plan,
                    session=session,
                    runtime_state=shared_runtime_state,
                )
                plan = final_plan
            finally:
                await self.executor._close_session(session)

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
            "initial_execution_result": initial_execution_result,
            "page_snapshot": shared_page_snapshot,
        }

    @staticmethod
    def _ensure_open_url_for_final_plan(plan: TaskSpec) -> TaskSpec:
        if not plan.steps:
            return plan
        if plan.steps[0].action == "open_url":
            return plan

        injected = {
            "step_id": 1,
            "action": "open_url",
            "args": {"url": str(plan.start_url)},
        }
        existing = [step.model_dump(mode="json") for step in plan.steps]
        normalized_steps = [injected]
        for idx, step in enumerate(existing, start=2):
            step["step_id"] = idx
            normalized_steps.append(step)

        return plan.model_validate({
            **plan.model_dump(mode="json"),
            "steps": normalized_steps,
        })

    @staticmethod
    def _persist_final_plan_repair_failure(
        invalid_plan: dict,
        validation_error: str,
        repaired_raw_response: str | None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = RAW_LLM_DIR / f"replanner_repair_failed_{timestamp}.json"
        payload = {
            "raw_invalid_plan": invalid_plan,
            "validation_error": validation_error,
            "repaired_raw_response": repaired_raw_response,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
