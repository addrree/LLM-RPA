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
            execution_result, verdict, final_plan, replanner_artifact = await self._execute_verify_with_correction_loop(
                user_goal=user_goal,
                initial_plan=plan,
                session=None,
                runtime_state=None,
                page_snapshot=None,
            )
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

                execution_result, verdict, final_plan, replanner_artifact = await self._execute_verify_with_correction_loop(
                    user_goal=user_goal,
                    initial_plan=final_plan,
                    session=session,
                    runtime_state=shared_runtime_state,
                    page_snapshot=page_snapshot,
                )
                plan = final_plan
            finally:
                await self.executor._close_session(session)

        return {
            "plan": final_plan if self.two_stage_planning else plan,
            "initial_plan": initial_plan,
            "final_plan": final_plan if self.two_stage_planning else None,
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

    async def _execute_verify_with_correction_loop(
        self,
        *,
        user_goal: str,
        initial_plan: TaskSpec,
        session,
        runtime_state,
        page_snapshot: PageSnapshot | None,
    ):
        current_plan = initial_plan
        max_retries = max(0, int(current_plan.constraints.max_verification_retries))
        attempt = 0
        replanner_artifact = self.replanner.last_artifact if self.replanner else None

        while True:
            execution_result = await self.executor.execute(current_plan, session=session, runtime_state=runtime_state)
            verdict = self.verifier.verify(current_plan, execution_result)
            if verdict.verdict == "accept":
                return execution_result, verdict, current_plan, replanner_artifact

            if attempt >= max_retries or self.replanner is None:
                return execution_result, verdict, current_plan, replanner_artifact

            effective_snapshot = page_snapshot or self._build_page_snapshot_from_execution(execution_result)
            corrective_plan = self.replanner.build_corrective_plan(
                user_goal=user_goal,
                page_snapshot=effective_snapshot,
                previous_plan=current_plan,
                execution_result=execution_result.model_dump(mode="json"),
                verifier_verdict=verdict.model_dump(mode="json"),
            )
            corrective_plan = self._ensure_open_url_for_final_plan(corrective_plan)
            self.validator.validate(corrective_plan)
            current_plan = corrective_plan
            replanner_artifact = self.replanner.last_artifact
            attempt += 1

    @staticmethod
    def _build_page_snapshot_from_execution(execution_result):
        return PageSnapshot.model_validate(
            {
                "url": execution_result.final_url or "about:blank",
                "title": execution_result.page_title or "",
                "screenshot_path": execution_result.screenshot_path or "",
                "page_text_excerpt": execution_result.page_text_excerpt or "",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    @staticmethod
    def _ensure_open_url_for_final_plan(plan: TaskSpec) -> TaskSpec:
        if not plan.steps:
            return plan
        if plan.steps[0].action == "open_url":
            first = plan.steps[0].model_dump(mode="json")
            if not str(first.get("args", {}).get("url", "")).strip() and str(plan.start_url):
                first["args"] = {"url": str(plan.start_url)}
                steps = [first] + [step.model_dump(mode="json") for step in plan.steps[1:]]
                for idx, step in enumerate(steps, start=1):
                    step["step_id"] = idx
                return plan.model_validate({**plan.model_dump(mode="json"), "steps": steps})
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
