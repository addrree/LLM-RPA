import json
import logging
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
logger = logging.getLogger(__name__)


class WorkflowStageError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class WorkflowManager:
    SAFE_CORRECTIVE_ACTIONS = {
        "open_url",
        "click",
        "type",
        "wait_for",
        "extract_text",
        "extract_html",
        "extract_items",
        "extract_structured_items",
        "observe_page",
        "extract_pattern_from_page_text",
        "extract_text_near_text",
        "extract_value_near_anchor",
        "finish",
    }

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
        corrective_retry_used = False
        corrective_retry_count = 0
        corrective_plan_valid_count = 0
        corrective_plan_invalid_count = 0
        initial_plan_valid: bool | None = None
        final_plan_valid: bool | None = None
        action_oov_detected = False

        if not self.two_stage_planning:
            try:
                plan = self.planner.build_plan(user_goal)
            except Exception as exc:  # noqa: BLE001
                raise WorkflowStageError("planning", str(exc)) from exc
            plan = self._normalize_plan_for_validation(plan)
            action_oov_detected = bool(getattr(self.planner, "last_action_oov_detected", False))
            try:
                self.validator.validate(plan)
                initial_plan_valid = True
                final_plan_valid = True
            except PlanValidationError as exc:
                initial_plan_valid = False
                final_plan_valid = False
                raise WorkflowStageError("validation", str(exc)) from exc
            execution_result, verdict, final_plan, replanner_artifact, corrective_retry_used, corrective_retry_count, corrective_plan_valid_count, corrective_plan_invalid_count = await self._execute_verify_with_correction_loop(
                user_goal=user_goal,
                initial_plan=plan,
                session=None,
                runtime_state=None,
                page_snapshot=None,
            )
        else:
            if self.replanner is None:
                raise ValueError("two_stage_planning requires replanner")

            try:
                initial_plan = self.planner.build_initial_plan(user_goal)
            except Exception as exc:  # noqa: BLE001
                raise WorkflowStageError("planning", str(exc)) from exc
            initial_plan = self._normalize_plan_for_validation(initial_plan)
            action_oov_detected = bool(getattr(self.planner, "last_action_oov_detected", False))
            try:
                self.validator.validate(initial_plan)
                initial_plan_valid = True
            except PlanValidationError as exc:
                initial_plan_valid = False
                raise WorkflowStageError("validation", str(exc)) from exc
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
                        "corrective_retry_used": False,
                        "corrective_retry_count": 0,
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
                action_oov_detected = action_oov_detected or bool(
                    getattr(self.replanner, "last_action_oov_detected", False)
                )
                replanner_artifact = self.replanner.last_artifact
                final_plan = self._normalize_plan_for_validation(final_plan)
                try:
                    self.validator.validate(final_plan)
                    final_plan_valid = True
                except PlanValidationError as first_error:
                    final_plan_valid = False
                    invalid_plan_dump = final_plan.model_dump(mode="json")
                    repaired_plan = self.replanner.revise_plan(
                        user_goal=user_goal,
                        page_snapshot=page_snapshot,
                        previous_plan=initial_plan,
                        validation_error=str(first_error),
                        invalid_plan=invalid_plan_dump,
                    )
                    action_oov_detected = action_oov_detected or bool(
                        getattr(self.replanner, "last_action_oov_detected", False)
                    )
                    replanner_artifact = self.replanner.last_artifact
                    final_plan = self._normalize_plan_for_validation(repaired_plan)
                    try:
                        self.validator.validate(final_plan)
                        final_plan_valid = True
                    except PlanValidationError as second_error:
                        self._persist_final_plan_repair_failure(
                            invalid_plan=invalid_plan_dump,
                            validation_error=str(second_error),
                            repaired_raw_response=replanner_artifact.raw_response if replanner_artifact else None,
                        )
                        raise WorkflowStageError("validation", str(second_error)) from second_error

                execution_result, verdict, final_plan, replanner_artifact, corrective_retry_used, corrective_retry_count, corrective_plan_valid_count, corrective_plan_invalid_count = await self._execute_verify_with_correction_loop(
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
            "corrective_retry_used": corrective_retry_used,
            "corrective_retry_count": corrective_retry_count,
            "correction_attempt_count": corrective_retry_count,
            "corrective_plan_valid_count": corrective_plan_valid_count,
            "corrective_plan_invalid_count": corrective_plan_invalid_count,
            "initial_plan_valid": initial_plan_valid,
            "final_plan_valid": final_plan_valid,
            "action_oov_detected": action_oov_detected,
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
        corrective_attempt_count = 0
        corrective_plan_valid_count = 0
        corrective_plan_invalid_count = 0
        replanner_artifact = self.replanner.last_artifact if self.replanner else None
        prior_corrective_attempts: list[dict] = []
        prior_signatures: set[str] = set()

        while True:
            execution_result = await self.executor.execute(current_plan, session=session, runtime_state=runtime_state)
            verdict = self.verifier.verify(current_plan, execution_result)
            if verdict.verdict == "accept":
                return (
                    execution_result,
                    verdict,
                    current_plan,
                    replanner_artifact,
                    corrective_attempt_count > 0,
                    corrective_attempt_count,
                    corrective_plan_valid_count,
                    corrective_plan_invalid_count,
                )

            if corrective_attempt_count >= max_retries or self.replanner is None:
                return (
                    execution_result,
                    verdict,
                    current_plan,
                    replanner_artifact,
                    corrective_attempt_count > 0,
                    corrective_attempt_count,
                    corrective_plan_valid_count,
                    corrective_plan_invalid_count,
                )

            corrective_attempt_count += 1
            effective_snapshot = page_snapshot or self._build_page_snapshot_from_execution(execution_result)
            try:
                corrective_plan = self.replanner.build_corrective_plan(
                    user_goal=user_goal,
                    page_snapshot=effective_snapshot,
                    previous_plan=current_plan,
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                    prior_corrective_attempts=prior_corrective_attempts,
                )
            except Exception as corrective_error:  # noqa: BLE001
                logger.error("Corrective plan generation failed: %s", corrective_error)
                corrective_plan_invalid_count += 1
                self._persist_corrective_plan_generation_failure(
                    attempt=corrective_attempt_count,
                    generation_error=str(corrective_error),
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                    replanner_raw_response=(
                        self.replanner.last_artifact.raw_response
                        if self.replanner and self.replanner.last_artifact is not None
                        else None
                    ),
                )
                prior_corrective_attempts.append(
                    {
                        "attempt": corrective_attempt_count,
                        "status": "generation_failed",
                        "error": str(corrective_error),
                    }
                )
                continue

            corrective_plan = self._normalize_plan_for_validation(corrective_plan)
            self._persist_corrective_plan_candidate(
                corrective_plan=corrective_plan,
                attempt=corrective_attempt_count,
                execution_result=execution_result.model_dump(mode="json"),
                verifier_verdict=verdict.model_dump(mode="json"),
                replanner_raw_response=(
                    self.replanner.last_artifact.raw_response
                    if self.replanner and self.replanner.last_artifact is not None
                    else None
                ),
            )
            unsupported_actions = self._unsupported_corrective_actions(corrective_plan)
            if unsupported_actions:
                corrective_plan_invalid_count += 1
                error_message = (
                    "Corrective retry policy rejected plan due to unsupported actions: "
                    f"{', '.join(unsupported_actions)}"
                )
                logger.error(error_message)
                self._persist_corrective_plan_validation_failure(
                    corrective_plan=corrective_plan,
                    attempt=corrective_attempt_count,
                    validation_error=error_message,
                    offending_step=None,
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                )
                prior_corrective_attempts.append(
                    {
                        "attempt": corrective_attempt_count,
                        "status": "invalid",
                        "error": error_message,
                    }
                )
                continue

            signature = self._plan_signature(corrective_plan)
            if signature in prior_signatures:
                corrective_plan_invalid_count += 1
                duplicate_error = "Corrective plan duplicates a previously failed corrective attempt."
                self._persist_corrective_plan_validation_failure(
                    corrective_plan=corrective_plan,
                    attempt=corrective_attempt_count,
                    validation_error=duplicate_error,
                    offending_step=None,
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                )
                prior_corrective_attempts.append(
                    {
                        "attempt": corrective_attempt_count,
                        "status": "invalid",
                        "error": duplicate_error,
                    }
                )
                continue

            try:
                self.validator.validate(corrective_plan)
            except PlanValidationError as validation_error:
                corrective_plan_invalid_count += 1
                offending_step = self._identify_offending_step(
                    corrective_plan=corrective_plan,
                    validation_error=str(validation_error),
                )
                logger.error(
                    "Corrective plan rejected by validator: %s | offending_step=%s",
                    validation_error,
                    offending_step,
                )
                self._persist_corrective_plan_validation_failure(
                    corrective_plan=corrective_plan,
                    attempt=corrective_attempt_count,
                    validation_error=str(validation_error),
                    offending_step=offending_step,
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                )
                prior_corrective_attempts.append(
                    {
                        "attempt": corrective_attempt_count,
                        "status": "invalid",
                        "error": str(validation_error),
                    }
                )
                continue

            corrective_plan_valid_count += 1
            prior_signatures.add(signature)
            prior_corrective_attempts.append(
                {
                    "attempt": corrective_attempt_count,
                    "status": "valid",
                    "plan_signature": signature,
                }
            )
            current_plan = corrective_plan
            replanner_artifact = self.replanner.last_artifact

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
    def _normalize_plan_for_validation(plan: TaskSpec) -> TaskSpec:
        normalized = WorkflowManager._ensure_open_url_for_final_plan(plan)
        payload = normalized.model_dump(mode="json")
        changed = False
        for step in payload.get("steps", []):
            if step.get("action") == "observe_page":
                save_as = step.get("save_as")
                if not isinstance(save_as, str) or not save_as.strip():
                    step["save_as"] = "page_snapshot"
                    changed = True
        if not changed:
            return normalized
        return normalized.model_validate(payload)

    @classmethod
    def _unsupported_corrective_actions(cls, corrective_plan: TaskSpec) -> list[str]:
        unsupported = {
            step.action
            for step in corrective_plan.steps
            if step.action not in cls.SAFE_CORRECTIVE_ACTIONS
        }
        return sorted(unsupported)

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

    @staticmethod
    def _plan_signature(plan: TaskSpec) -> str:
        material = {
            "start_url": str(plan.start_url),
            "steps": [step.model_dump(mode="json") for step in plan.steps],
            "required_fields": list(plan.expected_result.required_fields),
        }
        return json.dumps(material, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _identify_offending_step(corrective_plan: TaskSpec, validation_error: str) -> dict | None:
        if "extract_items missing required args: container_selector" in validation_error:
            for step in corrective_plan.steps:
                if step.action == "extract_items" and not str(step.args.get("container_selector", "")).strip():
                    return {"step_id": step.step_id, "action": step.action, "args": step.args}
        return None

    @staticmethod
    def _persist_corrective_plan_candidate(
        *,
        corrective_plan: TaskSpec,
        attempt: int,
        execution_result: dict,
        verifier_verdict: dict,
        replanner_raw_response: str | None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        path = RAW_LLM_DIR / f"corrective_plan_candidate_attempt{attempt}_{timestamp}.json"
        payload = {
            "attempt": attempt,
            "corrective_plan": corrective_plan.model_dump(mode="json"),
            "execution_result": execution_result,
            "verifier_verdict": verifier_verdict,
            "replanner_raw_response": replanner_raw_response,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _persist_corrective_plan_validation_failure(
        *,
        corrective_plan: TaskSpec,
        attempt: int,
        validation_error: str,
        offending_step: dict | None,
        execution_result: dict,
        verifier_verdict: dict,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        path = RAW_LLM_DIR / f"corrective_plan_validation_failed_attempt{attempt}_{timestamp}.json"
        payload = {
            "attempt": attempt,
            "validation_error": validation_error,
            "offending_step": offending_step,
            "corrective_plan": corrective_plan.model_dump(mode="json"),
            "execution_result": execution_result,
            "verifier_verdict": verifier_verdict,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _persist_corrective_plan_generation_failure(
        *,
        attempt: int,
        generation_error: str,
        execution_result: dict,
        verifier_verdict: dict,
        replanner_raw_response: str | None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        path = RAW_LLM_DIR / f"corrective_plan_generation_failed_attempt{attempt}_{timestamp}.json"
        payload = {
            "attempt": attempt,
            "generation_error": generation_error,
            "execution_result": execution_result,
            "verifier_verdict": verifier_verdict,
            "replanner_raw_response": replanner_raw_response,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
