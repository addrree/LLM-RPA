from urllib.parse import urlparse

from app.config import GLOBAL_MAX_REPLANS, GLOBAL_MAX_STEPS, GLOBAL_TIMEOUT_SEC
from app.schemas.task_spec import TaskSpec


ALLOWED_ACTIONS = {
    "open_url",
    "click",
    "type",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_items",
    "screenshot",
    "observe_page",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "finish",
}

TECHNICAL_ARTIFACT_FIELDS = {"screenshot_path", "screenshot", "artifact_screenshot"}


class PlanValidationError(Exception):
    pass


class PlanValidator:
    def validate(self, plan: TaskSpec) -> None:
        self._validate_steps_not_empty(plan)
        self._validate_step_count(plan)
        self._validate_actions(plan)
        self._validate_step_order(plan)
        self._validate_finish_step(plan)
        self._validate_constraints(plan)
        self._validate_domains(plan)
        self._validate_expected_result_consistency(plan)

    def _validate_steps_not_empty(self, plan: TaskSpec) -> None:
        if not plan.steps:
            raise PlanValidationError("Plan contains no steps.")

    def _validate_step_count(self, plan: TaskSpec) -> None:
        if len(plan.steps) > GLOBAL_MAX_STEPS:
            raise PlanValidationError("Plan exceeds global step limit.")

    def _validate_actions(self, plan: TaskSpec) -> None:
        for step in plan.steps:
            if step.action not in ALLOWED_ACTIONS:
                raise PlanValidationError(f"Unsupported action: {step.action}")

            if step.action == "open_url" and not step.args.get("url"):
                raise PlanValidationError(
                    f"open_url requires non-empty 'url'. Problematic step: {step.model_dump(mode='json')}"
                )
            if step.action == "click" and "selector" not in step.args:
                raise PlanValidationError("click requires 'selector'")
            if step.action == "type" and ("selector" not in step.args or "text" not in step.args):
                raise PlanValidationError("type requires 'selector' and 'text'")
            if step.action in {"extract_text", "extract_html"} and "selector" not in step.args:
                raise PlanValidationError(f"{step.action} requires 'selector'")
            if step.action == "extract_items":
                self._validate_extract_items(step.args, step.save_as)
            if step.action == "observe_page" and not step.save_as:
                raise PlanValidationError("observe_page requires 'save_as'")
            if step.action == "extract_pattern_from_page_text":
                self._validate_extract_pattern_from_page_text(step.args, step.save_as)
            if step.action == "extract_text_near_text":
                self._validate_extract_text_near_text(step.args, step.save_as)

    @staticmethod
    def _validate_extract_items(args: dict, save_as: str | None) -> None:
        required = {"container_selector", "limit", "fields"}
        missing = [key for key in required if key not in args]
        if missing:
            raise PlanValidationError(f"extract_items missing required args: {', '.join(missing)}")

        if not isinstance(args["fields"], dict) or not args["fields"]:
            raise PlanValidationError("extract_items requires non-empty 'fields' dict")

        limit = args.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            raise PlanValidationError("extract_items requires positive integer 'limit'")

        if not save_as:
            raise PlanValidationError("extract_items requires 'save_as'")

    @staticmethod
    def _validate_extract_pattern_from_page_text(args: dict, save_as: str | None) -> None:
        if "pattern" not in args or not str(args.get("pattern", "")).strip():
            raise PlanValidationError("extract_pattern_from_page_text requires non-empty 'pattern'")
        occurrence = args.get("occurrence", 1)
        if not isinstance(occurrence, int) or occurrence <= 0:
            raise PlanValidationError("extract_pattern_from_page_text requires positive integer 'occurrence'")
        group_index = args.get("group_index")
        if group_index is not None and (not isinstance(group_index, int) or group_index < 0):
            raise PlanValidationError(
                "extract_pattern_from_page_text requires non-negative integer 'group_index'"
            )
        normalize_number = args.get("normalize_number")
        if normalize_number is not None and not isinstance(normalize_number, bool):
            raise PlanValidationError("extract_pattern_from_page_text requires boolean 'normalize_number'")
        number_type = args.get("number_type")
        if number_type is not None and number_type != "int":
            raise PlanValidationError("extract_pattern_from_page_text supports only number_type='int'")
        strip_plus = args.get("strip_plus")
        if strip_plus is not None and not isinstance(strip_plus, bool):
            raise PlanValidationError("extract_pattern_from_page_text requires boolean 'strip_plus'")
        if not save_as:
            raise PlanValidationError("extract_pattern_from_page_text requires 'save_as'")

    @staticmethod
    def _validate_extract_text_near_text(args: dict, save_as: str | None) -> None:
        if "anchor_text" not in args or not str(args.get("anchor_text", "")).strip():
            raise PlanValidationError("extract_text_near_text requires non-empty 'anchor_text'")
        if "pattern" not in args or not str(args.get("pattern", "")).strip():
            raise PlanValidationError("extract_text_near_text requires non-empty 'pattern'")
        window_chars = args.get("window_chars", 200)
        if not isinstance(window_chars, int) or window_chars <= 0:
            raise PlanValidationError("extract_text_near_text requires positive integer 'window_chars'")
        if not save_as:
            raise PlanValidationError("extract_text_near_text requires 'save_as'")

    def _validate_step_order(self, plan: TaskSpec) -> None:
        expected_ids = list(range(1, len(plan.steps) + 1))
        actual_ids = [step.step_id for step in plan.steps]
        if actual_ids != expected_ids:
            raise PlanValidationError("Step IDs must be consecutive starting from 1.")

    def _validate_finish_step(self, plan: TaskSpec) -> None:
        if plan.steps[-1].action != "finish":
            raise PlanValidationError("Last step must be 'finish'.")

    def _validate_constraints(self, plan: TaskSpec) -> None:
        if plan.constraints.max_steps > GLOBAL_MAX_STEPS:
            raise PlanValidationError("max_steps exceeds global limit.")
        if plan.constraints.max_replans > GLOBAL_MAX_REPLANS:
            raise PlanValidationError("max_replans exceeds global limit.")
        if plan.constraints.timeout_sec > GLOBAL_TIMEOUT_SEC:
            raise PlanValidationError("timeout_sec exceeds global limit.")

    def _validate_domains(self, plan: TaskSpec) -> None:
        parsed = urlparse(str(plan.start_url))
        start_netloc = parsed.netloc
        if not plan.allowed_domains:
            return

        for allowed in plan.allowed_domains:
            if start_netloc == allowed or start_netloc.endswith(f".{allowed}"):
                return
        raise PlanValidationError(
            "start_url domain is not allowed. "
            f"start_url={plan.start_url}, start_netloc={start_netloc}, allowed_domains={plan.allowed_domains}"
        )

    def _validate_expected_result_consistency(self, plan: TaskSpec) -> None:
        saved_fields = {step.save_as for step in plan.steps if step.save_as}
        for field in plan.expected_result.required_fields:
            if field in TECHNICAL_ARTIFACT_FIELDS:
                continue
            if field not in saved_fields:
                raise PlanValidationError(
                    f"Required field '{field}' is not produced by any step."
                )
