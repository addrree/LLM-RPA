import json
import logging
import re
from urllib.parse import urlparse
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


_BENCHMARK_MINIMAL_REQUIRED_FIELDS_BY_FAMILY = {
    "single_value_extraction": ["value"],
    "anchored_value_extraction": ["value"],
    "repeated_structured_items": ["items"],
    "navigation_then_extraction": ["value"],
    "multi_step_information_retrieval": ["combined_result"],
    "negative_or_ambiguous_case": [],
}


def _is_non_empty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and value > 0


def _fallback_required_fields(*, normalized_required: list[str], task_family: str) -> list[str]:
    if normalized_required:
        return normalized_required
    return list(_BENCHMARK_MINIMAL_REQUIRED_FIELDS_BY_FAMILY.get(task_family, []))


def _infer_anchor_value_type(goal_text: str) -> str:
    normalized = goal_text.lower()
    if any(token in normalized for token in ("email", "mail", "почт")):
        return "email"
    if any(token in normalized for token in ("phone", "tel", "тел")):
        return "phone"
    if any(token in normalized for token in ("contact", "support", "help", "контакт", "поддерж")):
        return "email_or_phone"
    return "number"


def _capture_group_count(pattern: str) -> int:
    try:
        return re.compile(pattern).groups
    except re.error:
        return 0


def _canonical_structured_args(args: dict | None, *, default_limit: int) -> dict:
    payload = args if isinstance(args, dict) else {}
    pattern = str(payload.get("pattern", "")).strip() or r"(.+)"
    groups = _capture_group_count(pattern)
    if groups == 0:
        pattern = r"(.+)"
        groups = 1

    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        if groups >= 2:
            fields = {"name": 1, "detail": 2}
        else:
            fields = {"value": 1}
    limit = payload.get("limit")
    if not _is_positive_int(limit):
        limit = default_limit
    return {"pattern": pattern, "fields": fields, "limit": limit}


def _canonical_section_lines_args(args: dict | None, *, default_heading: str, default_limit: int) -> dict:
    payload = args if isinstance(args, dict) else {}
    heading_text = ""
    for key in ("heading_text", "section_heading", "section_title", "title", "label", "anchor_text"):
        value = payload.get(key)
        if _is_non_empty_str(value):
            heading_text = str(value).strip()
            break
    if not heading_text:
        heading_text = default_heading

    limit = payload.get("limit")
    if not _is_positive_int(limit):
        limit = default_limit

    canonical = {
        "heading_text": heading_text,
        "limit": limit,
        "stop_at_heading": bool(payload.get("stop_at_heading", True)),
    }
    if _is_positive_int(payload.get("min_line_length")):
        canonical["min_line_length"] = int(payload["min_line_length"])
    if "ignore_case" in payload:
        canonical["ignore_case"] = bool(payload.get("ignore_case"))
    return canonical


_PLACEHOLDER_HEADING_PATTERNS = [
    re.compile(r"^\s*section\s+[ab12]\s*$", re.IGNORECASE),
    re.compile(r"^\s*(first|second)\s+section\s*$", re.IGNORECASE),
    re.compile(r"^\s*source\s+[ab12]\s*$", re.IGNORECASE),
]
_NON_LABEL_SELECTOR_TOKENS = re.compile(r"[#.\[\]>:,+~]|//|/|\s")
_SIMPLE_LABEL_PATTERN = re.compile(r"^[\w\-]{1,40}$", re.UNICODE)
_URLISH_OR_SLUG_PATTERN = re.compile(r"^(https?://|/)[^\s]+$|^[a-z0-9]+(?:[-_/][a-z0-9]+)+$", re.IGNORECASE)
_GOAL_QUOTED_TARGET_PATTERN = re.compile(r"[\"'“”«»]([^\"'“”«»]{1,60})[\"'“”«»]")
_GOAL_CAPITALIZED_TARGET_PATTERN = re.compile(r"\b([A-ZА-Я][\w-]{1,30}(?:\s+[A-ZА-Я][\w-]{1,30}){0,2})\b")
_GOAL_STOP_WORDS = {"Find", "Extract", "Get", "Open", "Navigate", "Click", "Then", "And"}
_GENERIC_SELECTOR_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9]{0,9}$")


def _is_placeholder_heading(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return True
    return any(pattern.match(candidate) for pattern in _PLACEHOLDER_HEADING_PATTERNS)


def _extract_page_text_heading_candidates(page_text: str | None) -> list[str]:
    if not _is_non_empty_str(page_text):
        return []
    candidates: list[str] = []
    for raw_line in str(page_text).splitlines():
        line = raw_line.strip().strip("#").strip()
        if not line or len(line) < 3 or len(line) > 80:
            continue
        if line.endswith((".", "!", "?", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue
        if _is_placeholder_heading(line):
            continue
        if line not in candidates:
            candidates.append(line)
    return candidates


def _resolve_compare_headings(page_snapshot: PageSnapshot | None) -> tuple[str | None, str | None]:
    if page_snapshot is None:
        return (None, None)
    ordered_unique: list[str] = []
    for heading in page_snapshot.visible_headings:
        clean = str(heading or "").strip()
        if not clean or clean in ordered_unique or _is_placeholder_heading(clean):
            continue
        ordered_unique.append(clean)
    for heading in _extract_page_text_heading_candidates(page_snapshot.page_text):
        if heading not in ordered_unique:
            ordered_unique.append(heading)
    if len(ordered_unique) >= 2:
        return (ordered_unique[0], ordered_unique[1])
    return (None, None)


def _selector_looks_like_plain_label(selector: str) -> bool:
    candidate = str(selector or "").strip()
    if not candidate:
        return False
    if _NON_LABEL_SELECTOR_TOKENS.search(candidate):
        return False
    return bool(_SIMPLE_LABEL_PATTERN.match(candidate))


def _label_looks_like_url_or_slug(label: str) -> bool:
    return bool(_URLISH_OR_SLUG_PATTERN.match(str(label or "").strip()))


def _step_has_valid_click_target(args: dict | None) -> bool:
    payload = args if isinstance(args, dict) else {}
    has_selector = _is_non_empty_str(payload.get("selector"))
    has_text = _is_non_empty_str(payload.get("text"))
    has_href = _is_non_empty_str(payload.get("href_contains"))
    has_role_name = _is_non_empty_str(payload.get("role")) and _is_non_empty_str(payload.get("name"))
    return bool(has_selector or has_text or has_href or has_role_name)


def _candidate_to_href_contains(value: str) -> str | None:
    candidate = str(value or "").strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        parsed = urlparse(candidate)
        if parsed.path and parsed.path != "/":
            return parsed.path
        if parsed.fragment:
            return parsed.fragment
        return None
    if candidate.startswith("/"):
        return candidate
    if _label_looks_like_url_or_slug(candidate):
        return candidate
    return None


def _infer_click_target_from_later_steps(*, steps: list[dict], start_index: int) -> dict | None:
    for later_step in steps[start_index + 1 :]:
        later_args = later_step.get("args")
        if not isinstance(later_args, dict):
            continue

        direct_href = _candidate_to_href_contains(str(later_args.get("href_contains", "")).strip())
        if direct_href:
            return {"href_contains": direct_href}

        for key in ("url_contains", "url", "path", "slug"):
            candidate_href = _candidate_to_href_contains(str(later_args.get(key, "")).strip())
            if candidate_href:
                return {"href_contains": candidate_href}

        selector_value = str(later_args.get("selector", "")).strip()
        if (
            selector_value
            and _selector_looks_like_plain_label(selector_value)
            and not _GENERIC_SELECTOR_TAG_PATTERN.match(selector_value.lower())
        ):
            candidate_href = _candidate_to_href_contains(selector_value)
            if candidate_href:
                return {"href_contains": candidate_href}
            return {"text": selector_value, "exact": True}
    return None


def _infer_click_text_from_goal(goal_text: str) -> str | None:
    for match in _GOAL_QUOTED_TARGET_PATTERN.finditer(str(goal_text or "")):
        candidate = match.group(1).strip()
        if 1 <= len(candidate.split()) <= 4:
            return candidate

    for match in _GOAL_CAPITALIZED_TARGET_PATTERN.finditer(str(goal_text or "")):
        candidate = match.group(1).strip()
        if candidate in _GOAL_STOP_WORDS:
            continue
        if len(candidate.split()) <= 4:
            return candidate
    return None


def _canonicalize_multi_step_compare_steps(steps: list[dict], *, page_snapshot: PageSnapshot | None = None) -> list[dict]:
    stable_candidates = [
        step
        for step in steps
        if str(step.get("action", "")).strip()
        in {"extract_structured_items", "extract_structured_items_from_region", "extract_value_from_section", "extract_section_lines"}
    ]
    source_a_candidate = stable_candidates[0] if len(stable_candidates) >= 1 else None
    source_b_candidate = stable_candidates[1] if len(stable_candidates) >= 2 else source_a_candidate

    open_step = None
    observe_step = None
    for step in steps:
        action = str(step.get("action", "")).strip()
        if action == "open_url" and open_step is None:
            open_step = step
        if action == "observe_page" and observe_step is None:
            observe_step = step

    preferred_a, preferred_b = _resolve_compare_headings(page_snapshot)
    rewritten: list[dict] = []
    if open_step is not None:
        rewritten.append(open_step)
    if observe_step is not None:
        rewritten.append(observe_step)
    else:
        rewritten.append({"action": "observe_page", "args": {}, "save_as": "page_snapshot"})
    rewritten.append(
        {
            "action": "extract_section_lines",
            "args": _canonical_section_lines_args(
                source_a_candidate.get("args") if source_a_candidate else None,
                default_heading=preferred_a or "Section A",
                default_limit=7,
            ),
            "save_as": "source_a",
        }
    )
    rewritten.append(
        {
            "action": "extract_section_lines",
            "args": _canonical_section_lines_args(
                source_b_candidate.get("args") if source_b_candidate else None,
                default_heading=preferred_b or "Section B",
                default_limit=7,
            ),
            "save_as": "source_b",
        }
    )
    rewritten.append(
        {
            "action": "compare_structured_values",
            "args": {"left_key": "source_a", "right_key": "source_b"},
            "save_as": "combined_result",
        }
    )
    return rewritten


def _canonicalize_family_steps(
    *,
    steps: list[dict],
    task_family: str,
    goal_text: str,
    fallback_save_as: str | None,
    page_snapshot: PageSnapshot | None = None,
) -> list[dict]:
    if not steps:
        return steps

    if task_family == "multi_step_information_retrieval":
        steps = _canonicalize_multi_step_compare_steps(steps, page_snapshot=page_snapshot)

    extraction_indices = [
        index for index, step in enumerate(steps) if str(step.get("action", "")).strip().startswith("extract")
    ]
    final_extraction_index = extraction_indices[-1] if extraction_indices else None

    for index, step in enumerate(steps):
        action = str(step.get("action", "")).strip()
        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
            step["args"] = args

        if task_family == "anchored_value_extraction" and action == "extract_value_near_anchor":
            if not _is_non_empty_str(step.get("save_as")):
                step["save_as"] = "value"
            has_value_type = _is_non_empty_str(args.get("value_type"))
            has_value_pattern = _is_non_empty_str(args.get("value_pattern"))
            if not has_value_type and not has_value_pattern:
                args["value_type"] = _infer_anchor_value_type(goal_text)

        if task_family == "repeated_structured_items" and action == "extract_structured_items":
            if not _is_non_empty_str(step.get("save_as")):
                step["save_as"] = "items"
            canonical = _canonical_structured_args(args, default_limit=10)
            step["args"] = canonical
            args = canonical

        if task_family == "multi_step_information_retrieval" and action == "extract_section_lines":
            canonical = _canonical_section_lines_args(args, default_heading="Section", default_limit=7)
            if _is_placeholder_heading(canonical.get("heading_text", "")):
                preferred_a, preferred_b = _resolve_compare_headings(page_snapshot)
                replacement = preferred_a if str(step.get("save_as", "")).strip() == "source_a" else preferred_b
                if replacement:
                    canonical["heading_text"] = replacement
            step["args"] = canonical
            args = canonical

        if task_family == "navigation_then_extraction" and action == "click":
            selector_value = str(args.get("selector", "")).strip()
            if selector_value and _selector_looks_like_plain_label(selector_value):
                args.pop("selector", None)
                if _label_looks_like_url_or_slug(selector_value):
                    args["href_contains"] = selector_value
                else:
                    args["text"] = selector_value
                    args["exact"] = True
            has_text = _is_non_empty_str(args.get("text"))
            has_exact = bool(args.get("exact"))
            has_scope = _is_non_empty_str(args.get("scope_selector"))
            has_href = _is_non_empty_str(args.get("href_contains"))
            has_role_name = _is_non_empty_str(args.get("role")) and _is_non_empty_str(args.get("name"))
            if has_text and not (has_exact or has_scope or has_href or has_role_name):
                args["exact"] = True
            if not _step_has_valid_click_target(args):
                recovered_from_wait = False
                for look_ahead_index in range(index + 1, len(steps)):
                    wait_step = steps[look_ahead_index]
                    if str(wait_step.get("action", "")).strip() != "wait_for":
                        continue
                    wait_args = wait_step.get("args")
                    if not isinstance(wait_args, dict):
                        wait_args = {}
                        wait_step["args"] = wait_args
                    wait_text = str(wait_args.get("text", "")).strip()
                    wait_url_contains = str(wait_args.get("url_contains", "")).strip()
                    if wait_text:
                        args["text"] = wait_text
                        args["exact"] = True
                        wait_args.pop("text", None)
                        recovered_from_wait = True
                        break
                    if wait_url_contains:
                        args["href_contains"] = wait_url_contains
                        wait_args.pop("url_contains", None)
                        recovered_from_wait = True
                        break

                if not recovered_from_wait:
                    inferred_target = _infer_click_target_from_later_steps(steps=steps, start_index=index)
                    if inferred_target:
                        args.update(inferred_target)
                    else:
                        goal_label = _infer_click_text_from_goal(goal_text)
                        if goal_label:
                            args["text"] = goal_label
                            args["exact"] = True

        if task_family == "navigation_then_extraction" and action == "wait_for":
            has_text = _is_non_empty_str(args.get("text"))
            has_selector = _is_non_empty_str(args.get("selector"))
            has_url_contains = _is_non_empty_str(args.get("url_contains"))
            has_scope = _is_non_empty_str(args.get("scope_selector"))
            has_exact = bool(args.get("exact"))
            if not (has_selector or has_url_contains or has_text):
                args["selector"] = "h1"
                has_selector = True
            if has_text and not (has_selector or has_url_contains or has_scope or has_exact):
                final_extraction_action = None
                if final_extraction_index is not None:
                    final_extraction_action = str(steps[final_extraction_index].get("action", "")).strip()
                if final_extraction_action == "extract_text":
                    args.pop("text", None)
                    args["selector"] = "h1"
                else:
                    args["exact"] = True

        if final_extraction_index is not None and index == final_extraction_index:
            if task_family in {"single_value_extraction", "navigation_then_extraction"}:
                step["save_as"] = "value"
            if task_family == "multi_step_information_retrieval" and action == "compare_structured_values":
                step["save_as"] = "combined_result"
            if action == "extract_text" and not _is_non_empty_str(args.get("selector")):
                args["selector"] = "h1"

    if task_family == "single_value_extraction" and fallback_save_as == "value" and final_extraction_index is not None:
        steps[final_extraction_index]["save_as"] = "value"
    return steps


def normalize_benchmark_plan(
    plan: TaskSpec,
    benchmark_context: dict | None,
    page_snapshot: PageSnapshot | None = None,
) -> TaskSpec:
    if not benchmark_context:
        return plan

    payload = plan.model_dump(mode="json")
    steps = payload.get("steps", [])
    task_family = str(benchmark_context.get("task_family", "")).strip().lower()
    required_fields = benchmark_context.get("required_top_level_fields", [])
    allowed_actions = {
        str(action).strip()
        for action in (benchmark_context.get("allowed_actions") or [])
        if str(action).strip()
    }
    normalized_required = [
        str(field).strip()
        for field in required_fields
        if str(field).strip() and str(field).strip() != "page_snapshot"
    ]
    normalized_required = _fallback_required_fields(normalized_required=normalized_required, task_family=task_family)
    fallback_save_as = normalized_required[0] if len(normalized_required) == 1 else None

    normalized_steps: list[dict] = []
    unstable_multi_step_actions = {
        "extract_structured_items",
        "extract_value_from_section",
        "extract_structured_items_from_region",
    }
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip()
        if allowed_actions and action not in allowed_actions:
            if task_family == "multi_step_information_retrieval" and action in unstable_multi_step_actions:
                pass
            else:
                if action == "finish":
                    normalized_steps.append({"action": "finish", "args": {}})
                continue

        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
        step["args"] = args
        args.pop("page_language", None)

        if action == "extract_text" and not str(args.get("selector", "")).strip():
            args["selector"] = "h1"
        if action == "wait_for" and not isinstance(args.get("timeout_ms"), int):
            args["timeout_ms"] = 12000
        if action == "open_url" and not isinstance(args.get("timeout_ms"), int):
            args["timeout_ms"] = 20000

        if (
            fallback_save_as
            and action.startswith("extract")
            and not (isinstance(step.get("save_as"), str) and step.get("save_as", "").strip())
        ):
            step["save_as"] = fallback_save_as
        if (
            task_family == "negative_or_ambiguous_case"
            and action.startswith("extract")
            and not (isinstance(step.get("save_as"), str) and step.get("save_as", "").strip())
        ):
            step["save_as"] = "_probe"

        normalized_steps.append(step)

    normalized_steps = _canonicalize_family_steps(
        steps=normalized_steps,
        task_family=task_family,
        goal_text=str(payload.get("goal", "")),
        fallback_save_as=fallback_save_as,
        page_snapshot=page_snapshot,
    )

    if not any(str(step.get("action")) == "finish" for step in normalized_steps):
        normalized_steps.append({"action": "finish", "args": {}})
    for idx, step in enumerate(normalized_steps, start=1):
        step["step_id"] = idx

    payload["steps"] = normalized_steps
    payload.setdefault("expected_result", {})
    payload["expected_result"]["required_fields"] = normalized_required

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
        "extract_section_lines",
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
        "missing_probe_attempt",
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
            verdict = self._verifier_verify(
                plan=current_plan,
                execution_result=execution_result,
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
        benchmark_contract_context = benchmark_context if allowed_actions_override is None else None
        try:
            self.validator.validate(
                plan,
                allowed_actions=allowed_actions,
                benchmark_context=benchmark_contract_context,
            )
        except TypeError:
            try:
                self.validator.validate(plan, allowed_actions=allowed_actions)
            except TypeError:
                self.validator.validate(plan)


    def _verifier_verify(self, *, plan: TaskSpec, execution_result, benchmark_context: dict | None):
        try:
            return self.verifier.verify(plan, execution_result, benchmark_context=benchmark_context)
        except TypeError:
            return self.verifier.verify(plan, execution_result)

    @staticmethod
    def _augment_multi_step_comparison(execution_result) -> None:
        data = execution_result.extracted_data
        if "structured_comparison" in data and isinstance(data["structured_comparison"], dict):
            comparison = data["structured_comparison"]
            data["comparison"] = comparison
            data["compare_status"] = comparison.get("status")
            left_key = comparison.get("left_key") or ("source_a" if "source_a" in data else "section_a_data")
            right_key = comparison.get("right_key") or ("source_b" if "source_b" in data else "section_b_data")
            left = data.get(left_key)
            right = data.get(right_key)
            data.setdefault(
                "comparison_left_summary",
                {
                    "label": left_key,
                    "type": type(left).__name__,
                    "size": len(left) if isinstance(left, (dict, list)) else None,
                },
            )
            data.setdefault(
                "comparison_right_summary",
                {
                    "label": right_key,
                    "type": type(right).__name__,
                    "size": len(right) if isinstance(right, (dict, list)) else None,
                },
            )
            data["combined_result"] = {
                "source_a": left,
                "source_b": right,
                "comparison": comparison,
            }
            return
        left = data.get("source_a", data.get("section_a_data"))
        right = data.get("source_b", data.get("section_b_data"))
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
            "source_a": left,
            "source_b": right,
            "comparison": comparison,
        }

    @staticmethod
    def _effective_max_retries(raw_max_retries: int) -> int:
        return min(3, max(1, int(raw_max_retries)))

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
        if "probe/extraction" in text or "open_url -> finish only" in text:
            return "missing_probe_attempt"
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
