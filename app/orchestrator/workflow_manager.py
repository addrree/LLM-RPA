import json
import logging
import re
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


def _snapshot_confirms_click_text(page_snapshot: PageSnapshot | None, text_value: str) -> bool:
    needle = str(text_value or "").strip().lower()
    if not needle or page_snapshot is None:
        return False
    candidates = [
        *(page_snapshot.visible_headings or []),
        *(page_snapshot.visible_labels or []),
        *(page_snapshot.visible_buttons or []),
    ]
    page_text = str(page_snapshot.page_text or "").strip() or str(page_snapshot.page_text_excerpt or "").strip()
    if page_text:
        candidates.append(page_text)
    for candidate in candidates:
        haystack = str(candidate).strip().lower()
        if haystack and needle in haystack:
            return True
    return False


def _is_weak_navigation_wait(args: dict) -> bool:
    has_selector = bool(str(args.get("selector", "")).strip())
    has_url_contains = bool(str(args.get("url_contains", "")).strip())
    text_value = str(args.get("text", "")).strip()
    has_text = bool(text_value)
    has_scope = bool(str(args.get("scope_selector", "")).strip())
    has_exact = bool(args.get("exact", False))
    weak_text_wait = has_text and not has_selector and not has_url_contains and not has_scope
    generic_text_wait = text_value.lower() in {"python", "home", "docs", "pricing", "policy"} and not has_exact
    return weak_text_wait or generic_text_wait


def _promote_navigation_wait_for(
    *,
    wait_args: dict,
    navigation_click_args: dict | None,
) -> bool:
    if not navigation_click_args:
        return False
    href_contains = str(navigation_click_args.get("href_contains", "")).strip()
    role = str(navigation_click_args.get("role", "")).strip()
    name = str(navigation_click_args.get("name", "")).strip()
    selector = str(navigation_click_args.get("selector", "")).strip()
    has_role_name = bool(role and name)
    has_selector = bool(selector)

    if href_contains:
        wait_args["url_contains"] = href_contains
        wait_args.pop("text", None)
        wait_args.pop("selector", None)
        wait_args.pop("scope_selector", None)
        wait_args.pop("exact", None)
        return True
    if has_role_name or has_selector:
        wait_args["selector"] = "main h1, article h1, [role='main'] h1, main, article, [role='main']"
        wait_args.pop("text", None)
        wait_args.pop("scope_selector", None)
        wait_args.pop("exact", None)
        return True
    return False


def normalize_benchmark_plan(
    plan: TaskSpec,
    benchmark_context: dict | None,
    page_snapshot: PageSnapshot | None = None,
) -> TaskSpec:
    if not benchmark_context:
        return plan

    payload = plan.model_dump(mode="json")
    steps = payload.get("steps", [])
    required_fields = payload.get("expected_result", {}).get("required_fields", [])
    allowed_actions = {
        str(action).strip()
        for action in (benchmark_context.get("allowed_actions") or [])
        if str(action).strip()
    }
    fallback_save_as = "value"
    task_family = str((benchmark_context or {}).get("task_family", "")).strip()
    benchmark_anchor_candidates = [
        str(item).strip()
        for item in (benchmark_context or {}).get("scenario_anchor_candidates", [])
        if str(item).strip()
    ]
    benchmark_anchor_matching_mode = str((benchmark_context or {}).get("scenario_anchor_matching_mode", "")).strip().lower()
    if isinstance(required_fields, list) and required_fields:
        normalized_required = [str(field).strip() for field in required_fields if str(field).strip()]
        if "value" in normalized_required:
            fallback_save_as = "value"
        elif normalized_required:
            fallback_save_as = normalized_required[0]

    normalized_steps: list[dict] = []
    compare_step_idx: int | None = None
    extraction_step_indices: list[int] = []
    last_navigation_click_args: dict | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip()
        if allowed_actions and action not in allowed_actions:
            if action == "finish":
                normalized_steps.append({"action": "finish", "args": {}})
            continue

        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
        step["args"] = args

        if action == "extract_text" and not str(args.get("selector", "")).strip():
            args["selector"] = "h1"
        if action == "extract_pattern_from_page_text" and not str(args.get("pattern", "")).strip():
            args["pattern"] = "(.{1,200})"
            args.setdefault("group_index", 1)
        if action == "extract_value_near_anchor":
            if not str(args.get("anchor_text", "")).strip() and not args.get("anchor_candidates"):
                args["anchor_candidates"] = ["Contact", "Support", "Email", "Help", "Phone"]
            if not str(args.get("value_type", "")).strip() and not str(args.get("value_pattern", "")).strip():
                args["value_type"] = "email"
            # Language is detected by executor from page content/DOM.
            args.pop("page_language", None)
            if task_family == "anchored_value_extraction":
                if benchmark_anchor_candidates:
                    args["anchor_candidates"] = list(benchmark_anchor_candidates)
                    provided_anchor_text = str(args.get("anchor_text", "")).strip()
                    if provided_anchor_text and provided_anchor_text not in benchmark_anchor_candidates:
                        args.pop("anchor_text", None)
                if benchmark_anchor_matching_mode in {"auto", "exact", "contains"}:
                    args["anchor_matching_mode"] = benchmark_anchor_matching_mode
        if action == "extract_structured_items":
            if not str(args.get("pattern", "")).strip():
                args["pattern"] = "(.+)"
            limit = args.get("limit")
            if not isinstance(limit, int) or limit <= 0:
                args["limit"] = 5
            if not isinstance(args.get("fields"), dict) or not args.get("fields"):
                args["fields"] = {"value": 1}
            if task_family == "multi_step_information_retrieval":
                section_hint = str(args.get("section", "")).strip()
                region_hint = str(args.get("region", "")).strip()
                if section_hint or region_hint:
                    args["__benchmark_guardrail_error"] = (
                        "benchmark guardrail: multi_step_information_retrieval disallows compare extraction via "
                        "extract_structured_items + section/region hints; use extract_value_from_section or "
                        "extract_structured_items_from_region and persist section_a_data/section_b_data"
                    )
                else:
                    args["__benchmark_guardrail_error"] = (
                        "benchmark guardrail: multi_step_information_retrieval disallows legacy "
                        "extract_structured_items compare path; use section-aware extraction actions"
                    )
        if action == "wait_for" and not any(str(args.get(k, "")).strip() for k in ("selector", "url_contains", "text")):
            args["text"] = "Python"
        if task_family == "navigation_then_extraction" and action == "wait_for":
            if _is_weak_navigation_wait(args):
                promoted = _promote_navigation_wait_for(
                    wait_args=args,
                    navigation_click_args=last_navigation_click_args,
                )
                if not promoted:
                    args["__benchmark_guardrail_error"] = (
                        "benchmark guardrail: navigation_then_extraction wait_for is too weak; "
                        "use url_contains, visible selector in main content, or scoped text wait "
                        "(scope_selector + exact=true)"
                    )
        if action in {"click", "navigate_to_relevant_section"} and task_family == "navigation_then_extraction":
            last_navigation_click_args = dict(args)
        if action == "wait_for" and not isinstance(args.get("timeout_ms"), int):
            args["timeout_ms"] = 12000
        if action == "open_url" and not isinstance(args.get("timeout_ms"), int):
            args["timeout_ms"] = 20000

        if task_family == "single_value_extraction" and action == "extract_pattern_from_page_text":
            pattern = str(args.get("pattern", "")).strip()
            has_capture_groups = False
            if pattern:
                try:
                    has_capture_groups = re.compile(pattern).groups > 0
                except re.error:
                    has_capture_groups = False
            html_only_pattern = "<" in pattern and ">" in pattern
            if not has_capture_groups and not bool(args.get("normalize_number", False)):
                step["action"] = "extract_text"
                step["args"] = {"selector": "h1"}
            elif html_only_pattern:
                step["action"] = "extract_text"
                step["args"] = {"selector": "h1"}

        if task_family == "navigation_then_extraction" and action == "click":
            text_value = str(args.get("text", "")).strip()
            has_text = bool(text_value)
            has_selector = bool(str(args.get("selector", "")).strip())
            has_role_name = bool(str(args.get("role", "")).strip() and str(args.get("name", "")).strip())
            has_href_contains = bool(str(args.get("href_contains", "")).strip())
            has_scope = bool(str(args.get("scope_selector", "")).strip())
            has_exact = bool(args.get("exact"))
            if has_text and not (has_selector or has_role_name or has_href_contains or has_scope or has_exact):
                args["__benchmark_guardrail_error"] = (
                    "benchmark guardrail: navigation_then_extraction click with bare text is disallowed; "
                    "prefer href_contains, role+name, or specific selector"
                )
            text_only_contract = has_text and not (has_selector or has_role_name or has_href_contains)
            if text_only_contract and has_scope and has_exact:
                if not _snapshot_confirms_click_text(page_snapshot=page_snapshot, text_value=text_value):
                    args["__benchmark_guardrail_error"] = (
                        "benchmark guardrail: navigation_then_extraction click is over-constrained "
                        "(text+scope_selector+exact=true) without explicit snapshot confirmation; "
                        "normalize to href_contains/role+name or trigger corrective replanning"
                    )

        if task_family == "repeated_structured_items" and action == "extract_structured_items":
            pattern = str(args.get("pattern", "")).strip()
            compiled = None
            if pattern:
                try:
                    compiled = re.compile(pattern)
                except re.error:
                    compiled = None
            fields = args.get("fields")
            if isinstance(fields, dict):
                string_only_fields = all(isinstance(spec, str) for spec in fields.values())
                if string_only_fields and fields:
                    available_groups = compiled.groups if compiled is not None else 0
                    required_groups = len(fields)
                    if available_groups >= required_groups:
                        args["fields"] = {
                            str(field_name): idx
                            for idx, field_name in enumerate(fields.keys(), start=1)
                        }
                        fields = args["fields"]
                    else:
                        args["__benchmark_guardrail_error"] = (
                            "benchmark guardrail: repeated_structured_items extract_structured_items cannot "
                            "normalize string field specs because pattern capture groups are insufficient; "
                            f"need >= {required_groups}, available {available_groups}"
                        )
                referenced_groups: set[int] = set()
                for spec in fields.values():
                    if isinstance(spec, int):
                        referenced_groups.add(spec)
                    elif isinstance(spec, dict) and isinstance(spec.get("group_index"), int):
                        referenced_groups.add(int(spec["group_index"]))
                max_requested_group = max([g for g in referenced_groups if g > 0], default=0)
                available_groups = compiled.groups if compiled is not None else 0
                if max_requested_group > available_groups:
                    args["__benchmark_guardrail_error"] = (
                        "benchmark guardrail: repeated_structured_items extract_structured_items requires capture "
                        f"groups in pattern; requested group {max_requested_group}, available {available_groups}"
                    )

        if task_family == "multi_step_information_retrieval":
            if action == "extract_pattern_from_page_text":
                args["__benchmark_guardrail_error"] = (
                    "benchmark guardrail: multi_step_information_retrieval disallows regex-only extraction path; "
                    "use extract_value_from_section/extract_structured_items_from_region + compare_structured_values"
                )
            if action == "compare_structured_values":
                compare_step_idx = len(normalized_steps)
                if not step.get("save_as"):
                    step["save_as"] = "structured_comparison"
            if action in {"extract_value_from_section", "extract_structured_items_from_region"}:
                extraction_step_indices.append(len(normalized_steps))
                if len(extraction_step_indices) == 1:
                    step["save_as"] = "section_a_data"
                elif len(extraction_step_indices) == 2:
                    step["save_as"] = "section_b_data"

        if task_family == "negative_or_ambiguous_case" and action == "extract_pattern_from_page_text":
            pattern = str(args.get("pattern", "")).strip()
            has_capture_groups = False
            try:
                has_capture_groups = re.compile(pattern).groups > 0 if pattern else False
            except re.error:
                has_capture_groups = False
            if not has_capture_groups:
                args["__benchmark_guardrail_error"] = (
                    "benchmark guardrail: negative_or_ambiguous_case requires explicit capture-group regex for "
                    "extract_pattern_from_page_text; broad prose matching is disallowed"
                )

        if action.startswith("extract"):
            if task_family == "single_value_extraction":
                # Benchmarks in this family require scalar output in top-level `value`
                # for deterministic verification fast-path compatibility.
                step["save_as"] = "value"
            elif not step.get("save_as"):
                step["save_as"] = fallback_save_as

        normalized_steps.append(step)

    if task_family == "multi_step_information_retrieval" and compare_step_idx is not None:
        compare_step = normalized_steps[compare_step_idx]
        compare_args = compare_step.get("args", {}) if isinstance(compare_step.get("args"), dict) else {}
        left_key = str(compare_args.get("left_key", "section_a_data")).strip() or "section_a_data"
        right_key = str(compare_args.get("right_key", "section_b_data")).strip() or "section_b_data"
        compare_args["left_key"] = left_key
        compare_args["right_key"] = right_key
        compare_step["args"] = compare_args

        produced_before_compare = {
            str(step.get("save_as")).strip()
            for step in normalized_steps[:compare_step_idx]
            if isinstance(step, dict) and isinstance(step.get("save_as"), str) and step.get("save_as").strip()
        }
        missing_key_targets = [key for key in (left_key, right_key) if key not in produced_before_compare]
        candidate_indices = [
            idx
            for idx in extraction_step_indices
            if idx < compare_step_idx
            and normalized_steps[idx].get("action") in {"extract_value_from_section", "extract_structured_items_from_region"}
        ]
        for missing_key, step_idx in zip(missing_key_targets, candidate_indices):
            normalized_steps[step_idx]["save_as"] = missing_key
            produced_before_compare.add(missing_key)
        if any(key not in produced_before_compare for key in (left_key, right_key)):
            compare_args["__benchmark_guardrail_error"] = (
                "benchmark guardrail: multi_step compare requires prior extraction of both "
                f"'{left_key}' and '{right_key}'"
            )

    if not any(str(step.get("action")) == "finish" for step in normalized_steps):
        normalized_steps.append({"action": "finish", "args": {}})
    for idx, step in enumerate(normalized_steps, start=1):
        step["step_id"] = idx

    payload["steps"] = normalized_steps
    produced = {
        str(step.get("save_as")).strip()
        for step in normalized_steps
        if isinstance(step, dict) and isinstance(step.get("save_as"), str) and step.get("save_as").strip()
    }
    expected = payload.get("expected_result")
    if isinstance(expected, dict) and isinstance(expected.get("required_fields"), list):
        filtered = [field for field in expected["required_fields"] if str(field).strip() in produced]
        expected["required_fields"] = filtered
        payload["expected_result"] = expected

    return TaskSpec.model_validate(payload)


class WorkflowStageError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class WorkflowManager:
    INITIAL_TWO_STAGE_ALLOWED_ACTIONS = {"open_url", "observe_page", "finish"}
    SAFE_CORRECTIVE_ACTIONS = {
        "open_url",
        "click",
        "navigate_to_relevant_section",
        "type",
        "wait_for",
        "extract_text",
        "extract_html",
        "extract_items",
        "extract_structured_items",
        "extract_value_from_section",
        "extract_structured_items_from_region",
        "compare_structured_values",
        "assert_page_contains",
        "observe_page",
        "extract_pattern_from_page_text",
        "extract_text_near_text",
        "extract_value_near_anchor",
        "finish",
    }
    RECOVERABLE_FAILURE_TYPES = {
        "verification_reject",
        "missing_required_field",
        "schema_mismatch",
        "anchor_not_found",
        "value_not_found_near_anchor",
        "regex_group_mismatch",
        "ambiguous_click_target",
        "weak_click_target",
        "bad_locator_choice",
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

    @staticmethod
    def _normalize_benchmark_plan(
        plan: TaskSpec,
        benchmark_context: dict | None,
        page_snapshot: PageSnapshot | None = None,
    ) -> TaskSpec:
        return normalize_benchmark_plan(plan=plan, benchmark_context=benchmark_context, page_snapshot=page_snapshot)

    async def run(self, user_goal: str, benchmark_context: dict | None = None):
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
                plan = self._planner_build_plan(user_goal=user_goal, benchmark_context=benchmark_context)
            except Exception as exc:  # noqa: BLE001
                raise WorkflowStageError("planning", str(exc)) from exc
            plan = self._normalize_plan_for_validation(plan)
            plan = self._normalize_benchmark_plan(plan=plan, benchmark_context=benchmark_context)
            action_oov_detected = bool(getattr(self.planner, "last_action_oov_detected", False))
            try:
                self._validator_validate(plan=plan, benchmark_context=benchmark_context)
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
                benchmark_context=benchmark_context,
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
                self._validator_validate(
                    plan=initial_plan,
                    benchmark_context=benchmark_context,
                    allowed_actions_override=self.INITIAL_TWO_STAGE_ALLOWED_ACTIONS,
                )
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
                    verdict = self.verifier.verify(
                        initial_plan,
                        initial_execution,
                        benchmark_context=benchmark_context,
                    )
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

                final_plan = self._replanner_revise_plan(
                    user_goal=user_goal,
                    page_snapshot=page_snapshot,
                    previous_plan=initial_plan,
                    benchmark_context=benchmark_context,
                )
                action_oov_detected = action_oov_detected or bool(
                    getattr(self.replanner, "last_action_oov_detected", False)
                )
                replanner_artifact = self.replanner.last_artifact
                final_plan = self._normalize_plan_for_validation(final_plan)
                final_plan = self._normalize_benchmark_plan(
                    plan=final_plan,
                    benchmark_context=benchmark_context,
                    page_snapshot=page_snapshot,
                )
                try:
                    self._validator_validate(plan=final_plan, benchmark_context=benchmark_context)
                    final_plan_valid = True
                except PlanValidationError as first_error:
                    final_plan_valid = False
                    invalid_plan_dump = final_plan.model_dump(mode="json")
                    repaired_plan = self._replanner_revise_plan(
                        user_goal=user_goal,
                        page_snapshot=page_snapshot,
                        previous_plan=initial_plan,
                        validation_error=str(first_error),
                        invalid_plan=invalid_plan_dump,
                        benchmark_context=benchmark_context,
                    )
                    action_oov_detected = action_oov_detected or bool(
                        getattr(self.replanner, "last_action_oov_detected", False)
                    )
                    replanner_artifact = self.replanner.last_artifact
                    final_plan = self._normalize_plan_for_validation(repaired_plan)
                    final_plan = self._normalize_benchmark_plan(
                        plan=final_plan,
                        benchmark_context=benchmark_context,
                        page_snapshot=page_snapshot,
                    )
                    try:
                        self._validator_validate(plan=final_plan, benchmark_context=benchmark_context)
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
                    benchmark_context=benchmark_context,
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
        benchmark_context: dict | None,
    ):
        runtime_state = runtime_state if runtime_state is not None else {}
        runtime_state["benchmark_context"] = benchmark_context or {}
        current_plan = initial_plan
        max_retries = self._effective_max_retries(current_plan.constraints.max_verification_retries)
        max_retries = self._effective_max_retries_for_context(max_retries=max_retries, benchmark_context=benchmark_context)
        corrective_attempt_count = 0
        corrective_plan_valid_count = 0
        corrective_plan_invalid_count = 0
        replanner_artifact = self.replanner.last_artifact if self.replanner else None
        prior_corrective_attempts: list[dict] = []
        prior_signatures: set[str] = set()

        while True:
            execution_result = await self.executor.execute(current_plan, session=session, runtime_state=runtime_state)
            self._augment_multi_step_comparison(execution_result)
            verdict = self.verifier.verify(
                current_plan,
                execution_result,
                benchmark_context=benchmark_context,
            )
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

            failure_context = self._build_failure_context(execution_result=execution_result, verdict=verdict)
            if not self._should_retry_corrective(
                failure_type=failure_context["failure_type"],
                prior_corrective_attempts=prior_corrective_attempts,
                max_retries=max_retries,
                corrective_attempt_count=corrective_attempt_count,
            ):
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
                corrective_plan = self._replanner_build_corrective_plan(
                    user_goal=user_goal,
                    page_snapshot=effective_snapshot,
                    previous_plan=current_plan,
                    execution_result=execution_result.model_dump(mode="json"),
                    verifier_verdict=verdict.model_dump(mode="json"),
                    prior_corrective_attempts=prior_corrective_attempts,
                    failure_type=failure_context["failure_type"],
                    failed_action=failure_context["failed_action"],
                    failed_args=failure_context["failed_args"],
                    error_message=failure_context["error_message"],
                    verifier_issues=failure_context["verifier_issues"],
                    previous_attempt_signatures=sorted(prior_signatures),
                    disallowed_next_patterns=self._build_disallowed_patterns(prior_corrective_attempts),
                    benchmark_context=benchmark_context,
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
                        "failure_type": failure_context["failure_type"],
                    }
                )
                continue

            corrective_plan = self._normalize_plan_for_validation(corrective_plan)
            corrective_plan = self._normalize_benchmark_plan(
                plan=corrective_plan,
                benchmark_context=benchmark_context,
                page_snapshot=effective_snapshot,
            )
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
                        "failure_type": failure_context["failure_type"],
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
                        "failure_type": failure_context["failure_type"],
                    }
                )
                continue

            try:
                self._validator_validate(plan=corrective_plan, benchmark_context=benchmark_context)
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
                        "failure_type": failure_context["failure_type"],
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
                    "failure_type": failure_context["failure_type"],
                }
            )
            current_plan = corrective_plan
            replanner_artifact = self.replanner.last_artifact


    @staticmethod
    def _allowed_actions(benchmark_context: dict | None) -> set[str] | None:
        if not benchmark_context:
            return None
        allowed = benchmark_context.get("allowed_actions")
        if not isinstance(allowed, list):
            return None
        return {str(action) for action in allowed if str(action).strip()}

    def _planner_build_plan(self, *, user_goal: str, benchmark_context: dict | None):
        try:
            return self.planner.build_plan(user_goal, benchmark_context=benchmark_context)
        except TypeError:
            return self.planner.build_plan(user_goal)

    def _replanner_revise_plan(self, **kwargs):
        if self.replanner is None:
            raise ValueError("replanner is required")
        try:
            return self.replanner.revise_plan(**kwargs)
        except TypeError:
            kwargs.pop("benchmark_context", None)
            return self.replanner.revise_plan(**kwargs)

    def _replanner_build_corrective_plan(self, **kwargs):
        if self.replanner is None:
            raise ValueError("replanner is required")
        try:
            return self.replanner.build_corrective_plan(**kwargs)
        except TypeError:
            kwargs.pop("benchmark_context", None)
            return self.replanner.build_corrective_plan(**kwargs)

    def _validator_validate(
        self,
        *,
        plan: TaskSpec,
        benchmark_context: dict | None,
        allowed_actions_override: set[str] | None = None,
    ) -> None:
        allowed_actions = allowed_actions_override or self._allowed_actions(benchmark_context)
        try:
            self.validator.validate(plan, allowed_actions=allowed_actions)
        except TypeError:
            self.validator.validate(plan)

    @staticmethod
    def _augment_multi_step_comparison(execution_result) -> None:
        data = execution_result.extracted_data
        if "structured_comparison" in data and isinstance(data["structured_comparison"], dict):
            comparison = data["structured_comparison"]
            data["comparison"] = comparison
            data["compare_status"] = comparison.get("status")
            left = data.get(comparison.get("left_key", "section_a_data"))
            right = data.get(comparison.get("right_key", "section_b_data"))
            data.setdefault(
                "comparison_left_summary",
                {
                    "label": comparison.get("left_key", "section_a_data"),
                    "type": type(left).__name__,
                    "size": len(left) if isinstance(left, (dict, list)) else None,
                },
            )
            data.setdefault(
                "comparison_right_summary",
                {
                    "label": comparison.get("right_key", "section_b_data"),
                    "type": type(right).__name__,
                    "size": len(right) if isinstance(right, (dict, list)) else None,
                },
            )
            data["combined_result"] = {
                "section_a_data": left,
                "section_b_data": right,
                "comparison": comparison,
            }
            return
        left = data.get("section_a_data")
        right = data.get("section_b_data")
        if left is None or right is None:
            return
        left_keys = sorted(left.keys()) if isinstance(left, dict) else []
        right_keys = sorted(right.keys()) if isinstance(right, dict) else []
        differing_keys: list[str] = []
        differing_values: dict[str, dict] = {}
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left.keys()).union(right.keys())):
                if left.get(key) != right.get(key):
                    differing_keys.append(str(key))
                    differing_values[str(key)] = {
                        "section_a": left.get(key),
                        "section_b": right.get(key),
                    }

        comparison = {
            "compared": True,
            "left_present": left is not None,
            "right_present": right is not None,
            "left_type": type(left).__name__,
            "right_type": type(right).__name__,
            "left_keys": left_keys,
            "right_keys": right_keys,
            "shared_keys": sorted(set(left_keys).intersection(right_keys)),
            "exact_match": left == right,
            "differing_keys": differing_keys,
            "differing_values": differing_values,
            "status": "equal" if left == right else "different",
        }
        data["structured_comparison"] = comparison
        data["comparison"] = comparison
        data["compare_status"] = comparison.get("status")
        data["combined_result"] = {
            "section_a_data": left,
            "section_b_data": right,
            "comparison": comparison,
        }

    @staticmethod
    def _effective_max_retries(raw_max_retries: int) -> int:
        return min(3, max(0, int(raw_max_retries)))

    @staticmethod
    def _effective_max_retries_for_context(*, max_retries: int, benchmark_context: dict | None) -> int:
        if not benchmark_context:
            return max_retries
        family = str(benchmark_context.get("task_family", "")).strip()
        if family in {"single_value_extraction", "anchored_value_extraction"}:
            return min(max_retries, 1)
        if family in {"navigation_then_extraction"}:
            return min(max_retries, 2)
        return max_retries

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

    @classmethod
    def _build_failure_context(cls, *, execution_result, verdict) -> dict:
        failure_type = execution_result.failure_type or "verification_reject"
        error_message = str(execution_result.error_message or "").lower()
        failed_action = execution_result.failed_action
        failed_args = execution_result.failed_args or {}
        if execution_result.status != "success":
            if "anchor text not found" in error_message:
                failure_type = "anchor_not_found"
            elif "value not found near anchor" in error_message or "pattern not found near anchor_text" in error_message:
                failure_type = "value_not_found_near_anchor"
            elif "ambiguous or weak click target" in error_message:
                failure_type = "ambiguous_click_target"
            elif "too broad" in error_message and failed_action == "click":
                failure_type = "weak_click_target"
            elif "regex group reference is out of range" in error_message:
                failure_type = "regex_group_mismatch"
            elif execution_result.failure_type == "browser_operation_failed" and failed_action == "click":
                failure_type = "bad_locator_choice"
        if verdict.verdict == "reject" and execution_result.status == "success":
            failure_type = cls._classify_verifier_failure(verdict.issues)
        return {
            "failure_type": failure_type,
            "failed_action": failed_action,
            "failed_args": failed_args,
            "error_message": str(execution_result.error_message or ""),
            "verifier_issues": list(verdict.issues or []),
        }

    @classmethod
    def _classify_verifier_failure(cls, issues: list[str]) -> str:
        text = " ".join(issues).lower()
        if "required" in text or "missing" in text:
            return "missing_required_field"
        if "schema" in text or "format" in text or "structure" in text:
            return "schema_mismatch"
        return "verification_reject"

    @classmethod
    def _should_retry_corrective(
        cls,
        *,
        failure_type: str,
        prior_corrective_attempts: list[dict],
        max_retries: int,
        corrective_attempt_count: int,
    ) -> bool:
        if corrective_attempt_count >= max_retries:
            return False
        if failure_type not in cls.RECOVERABLE_FAILURE_TYPES:
            return False
        previous_failure_types = [
            attempt.get("failure_type")
            for attempt in prior_corrective_attempts
            if attempt.get("failure_type")
        ]
        if len(previous_failure_types) >= 2 and previous_failure_types[-1] == failure_type == previous_failure_types[-2]:
            return False
        return True

    @staticmethod
    def _build_disallowed_patterns(prior_corrective_attempts: list[dict]) -> list[str]:
        disallowed: list[str] = []
        for attempt in prior_corrective_attempts:
            error = str(attempt.get("error", "")).lower()
            if "too broad" in error and "click" in error:
                disallowed.append("broad_click_selector")
            if "ambiguous or weak click target" in error:
                disallowed.append("ambiguous_click_target")
            if "missing required args" in error:
                disallowed.append("missing_required_args")
            if "duplicate" in error:
                disallowed.append("duplicate_plan_signature")
            failure_type = str(attempt.get("failure_type", "")).lower()
            if failure_type in {"anchor_not_found", "value_not_found_near_anchor"}:
                disallowed.append("same_anchor_retry_without_candidates")
            if failure_type == "regex_group_mismatch":
                disallowed.append("same_regex_group_mismatch")
            if failure_type in {"ambiguous_click_target", "weak_click_target", "bad_locator_choice"}:
                disallowed.append("generic_click_target")
        return sorted(set(disallowed))

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
