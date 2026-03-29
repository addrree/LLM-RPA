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
    "finish",
}


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

            if step.action == "open_url" and "url" not in step.args:
                raise PlanValidationError("open_url requires 'url'")
            if step.action == "click" and "selector" not in step.args:
                raise PlanValidationError("click requires 'selector'")
            if step.action == "type" and ("selector" not in step.args or "text" not in step.args):
                raise PlanValidationError("type requires 'selector' and 'text'")
            if step.action in {"extract_text", "extract_html"} and "selector" not in step.args:
                raise PlanValidationError(f"{step.action} requires 'selector'")
            if step.action == "extract_items":
                self._validate_extract_items(step.args, step.save_as)

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
        if plan.allowed_domains and parsed.netloc not in plan.allowed_domains:
            raise PlanValidationError("start_url domain is not allowed.")

    def _validate_expected_result_consistency(self, plan: TaskSpec) -> None:
        saved_fields = {step.save_as for step in plan.steps if step.save_as}
        for field in plan.expected_result.required_fields:
            if field not in saved_fields:
                raise PlanValidationError(
                    f"Required field '{field}' is not produced by any step."
                )
