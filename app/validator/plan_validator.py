from urllib.parse import urlparse
import re

from app.benchmark.contract import required_contract_fields
from app.config import GLOBAL_MAX_REPLANS, GLOBAL_MAX_STEPS, GLOBAL_MAX_VERIFICATION_RETRIES, GLOBAL_TIMEOUT_SEC
from app.planner.action_vocab import CANONICAL_ACTIONS
from app.schemas.task_spec import TaskSpec


ALLOWED_ACTIONS = CANONICAL_ACTIONS

TECHNICAL_ARTIFACT_FIELDS = {"screenshot_path", "screenshot", "artifact_screenshot"}
TOO_BROAD_CLICK_SELECTORS = {"a", "button", "*", "[role='button']", '[role="button"]'}


class PlanValidationError(Exception):
    pass


class PlanValidator:
    def validate(
        self,
        plan: TaskSpec,
        allowed_actions: set[str] | None = None,
        benchmark_context: dict | None = None,
    ) -> None:
        self._validate_steps_not_empty(plan)
        self._validate_step_count(plan)
        self._validate_actions(plan, allowed_actions=allowed_actions)
        self._validate_step_order(plan)
        self._validate_finish_step(plan)
        self._validate_constraints(plan)
        self._validate_domains(plan)
        self._validate_expected_result_consistency(plan)
        self._validate_benchmark_contract(plan, benchmark_context=benchmark_context)

    def _validate_steps_not_empty(self, plan: TaskSpec) -> None:
        if not plan.steps:
            raise PlanValidationError("Plan contains no steps.")

    def _validate_step_count(self, plan: TaskSpec) -> None:
        if len(plan.steps) > GLOBAL_MAX_STEPS:
            raise PlanValidationError("Plan exceeds global step limit.")

    def _validate_actions(self, plan: TaskSpec, allowed_actions: set[str] | None = None) -> None:
        effective_allowed_actions = allowed_actions or ALLOWED_ACTIONS
        for step in plan.steps:
            benchmark_guardrail_error = str(step.args.get("__benchmark_guardrail_error", "")).strip()
            if benchmark_guardrail_error:
                raise PlanValidationError(benchmark_guardrail_error)
            if step.action not in effective_allowed_actions:
                raise PlanValidationError(f"Unsupported action: {step.action}")

            if step.action == "open_url" and not step.args.get("url"):
                raise PlanValidationError(
                    f"open_url requires non-empty 'url'. Problematic step: {step.model_dump(mode='json')}"
                )
            if step.action == "click":
                self._validate_click_args(step.args)
            if step.action == "type" and ("selector" not in step.args or "text" not in step.args):
                raise PlanValidationError("type requires 'selector' and 'text'")
            if step.action == "navigate_to_relevant_section":
                self._validate_click_args(step.args)
            if step.action == "wait_for":
                self._validate_wait_for_args(step.args)
            if step.action in {"extract_text", "extract_html"} and "selector" not in step.args:
                raise PlanValidationError(f"{step.action} requires 'selector'")
            if step.action == "extract_items":
                self._validate_extract_items(step.args, step.save_as)
            if step.action == "extract_structured_items":
                self._validate_extract_structured_items(step.args, step.save_as)
            if step.action == "extract_value_from_section":
                self._validate_extract_value_from_section(step.args, step.save_as)
            if step.action == "extract_structured_items_from_region":
                self._validate_extract_structured_items_from_region(step.args, step.save_as)
            if step.action == "compare_structured_values":
                self._validate_compare_structured_values(step.args, step.save_as)
            if step.action == "assert_page_contains":
                self._validate_assert_page_contains(step.args)
            if step.action == "observe_page" and not step.save_as:
                raise PlanValidationError("observe_page requires 'save_as'")
            if step.action == "extract_pattern_from_page_text":
                self._validate_extract_pattern_from_page_text(step.args, step.save_as)
            if step.action == "extract_text_near_text":
                self._validate_extract_text_near_text(step.args, step.save_as)
            if step.action == "extract_value_near_anchor":
                self._validate_extract_value_near_anchor(step.args, step.save_as)

    @staticmethod
    def _validate_click_args(args: dict) -> None:
        selector = str(args.get("selector", "")).strip()
        text = str(args.get("text", "")).strip()
        role = str(args.get("role", "")).strip()
        name = str(args.get("name", "")).strip()
        href_contains = str(args.get("href_contains", "")).strip()
        scope_selector = str(args.get("scope_selector", "")).strip()
        exact = args.get("exact")

        has_selector = bool(selector)
        has_text = bool(text)
        has_role_name = bool(role and name)
        has_href_filter = bool(href_contains)
        strategy_count = sum([has_selector, has_role_name, has_href_filter, has_text])

        if not (has_selector or has_text or has_role_name or has_href_filter):
            raise PlanValidationError(
                "click requires one of: non-empty 'selector', 'text', 'role'+'name', or 'href_contains'"
            )
        if strategy_count > 2:
            raise PlanValidationError(
                "click mixes too many target strategies; keep it deterministic (prefer role+name or href/text within scope)"
            )

        if has_selector and selector.lower() in TOO_BROAD_CLICK_SELECTORS:
            raise PlanValidationError(
                f"click selector is too broad: {selector!r}. Use a more specific selector or text/role contract."
            )
        if has_text and not (has_role_name or has_href_filter or scope_selector or bool(exact)):
            raise PlanValidationError(
                "click with bare text is too weak; add exact=true, scope_selector, role+name, or href_contains"
            )

    @staticmethod
    def _validate_wait_for_args(args: dict) -> None:
        selector = str(args.get("selector", "")).strip()
        url_contains = str(args.get("url_contains", "")).strip()
        text = str(args.get("text", "")).strip()
        scope_selector = str(args.get("scope_selector", "")).strip()

        if not (selector or url_contains or text):
            raise PlanValidationError("wait_for requires one of: selector | url_contains | text")
        if text and not (selector or url_contains or scope_selector or bool(args.get("exact", False))):
            raise PlanValidationError(
                "wait_for with bare text is too weak; add scope_selector, exact=true, or prefer selector/url_contains"
            )
        timeout_ms = args.get("timeout_ms")
        if timeout_ms is not None and (not isinstance(timeout_ms, int) or timeout_ms <= 0):
            raise PlanValidationError("wait_for timeout_ms must be a positive integer")

    @staticmethod
    def _validate_extract_items(args: dict, save_as: str | None) -> None:
        required = {"container_selector", "limit", "fields"}
        missing = [key for key in required if key not in args]
        if missing:
            raise PlanValidationError(f"extract_items missing required args: {', '.join(missing)}")

        if not isinstance(args["fields"], dict) or not args["fields"]:
            raise PlanValidationError("extract_items requires non-empty 'fields' dict")
        for field_name, rule in args["fields"].items():
            PlanValidator._validate_extract_items_field_rule(field_name, rule)

        limit = args.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            raise PlanValidationError("extract_items requires positive integer 'limit'")

        if not save_as:
            raise PlanValidationError("extract_items requires 'save_as'")

    @staticmethod
    def _validate_extract_items_field_rule(field_name: str, rule) -> None:
        if isinstance(rule, str):
            if not rule.strip():
                raise PlanValidationError(
                    f"extract_items field '{field_name}' requires non-empty selector string"
                )
            return

        if not isinstance(rule, dict):
            raise PlanValidationError(
                f"extract_items field '{field_name}' must be string selector or object rule"
            )

        selector = rule.get("selector")
        anchor_text = rule.get("anchor_text")
        value_pattern = rule.get("value_pattern")
        pattern = rule.get("pattern")

        if selector is not None and (not isinstance(selector, str) or not selector.strip()):
            raise PlanValidationError(
                f"extract_items field '{field_name}' has invalid 'selector'"
            )
        if rule.get("attr") is not None and (not isinstance(rule.get("attr"), str) or not rule["attr"].strip()):
            raise PlanValidationError(
                f"extract_items field '{field_name}' has invalid 'attr'"
            )
        if pattern is not None and (not isinstance(pattern, str) or not pattern.strip()):
            raise PlanValidationError(
                f"extract_items field '{field_name}' has invalid 'pattern'"
            )
        compiled_pattern = None
        if isinstance(pattern, str) and pattern.strip():
            compiled_pattern = PlanValidator._compile_pattern(
                pattern,
                action_name=f"extract_items field '{field_name}'",
            )
        if anchor_text is not None and (not isinstance(anchor_text, str) or not anchor_text.strip()):
            raise PlanValidationError(
                f"extract_items field '{field_name}' has invalid 'anchor_text'"
            )
        if value_pattern is not None and (not isinstance(value_pattern, str) or not value_pattern.strip()):
            raise PlanValidationError(
                f"extract_items field '{field_name}' has invalid 'value_pattern'"
            )

        if (anchor_text is None) ^ (value_pattern is None):
            raise PlanValidationError(
                f"extract_items field '{field_name}' must provide both 'anchor_text' and 'value_pattern'"
            )

        if not selector and pattern is None and anchor_text is None:
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires 'selector' or extraction rule"
            )

        if "group_index" in rule and (not isinstance(rule["group_index"], int) or rule["group_index"] < 0):
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires non-negative integer 'group_index'"
            )
        if compiled_pattern is not None:
            PlanValidator._validate_group_index_reference(
                group_index=rule.get("group_index"),
                compiled_pattern=compiled_pattern,
                action_name="extract_items",
                field_name=field_name,
            )
        if "normalize_number" in rule and not isinstance(rule["normalize_number"], bool):
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires boolean 'normalize_number'"
            )
        if "strip_plus" in rule and not isinstance(rule["strip_plus"], bool):
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires boolean 'strip_plus'"
            )
        if "max_distance_chars" in rule and (
            not isinstance(rule["max_distance_chars"], int) or rule["max_distance_chars"] <= 0
        ):
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires positive integer 'max_distance_chars'"
            )
        if "search_direction" in rule and rule["search_direction"] not in {"after", "before", "around"}:
            raise PlanValidationError(
                f"extract_items field '{field_name}' requires search_direction in {{'after','before','around'}}"
            )

    @staticmethod
    def _validate_extract_pattern_from_page_text(args: dict, save_as: str | None) -> None:
        if "pattern" not in args or not str(args.get("pattern", "")).strip():
            raise PlanValidationError("extract_pattern_from_page_text requires non-empty 'pattern'")
        compiled_pattern = PlanValidator._compile_pattern(
            str(args.get("pattern")),
            action_name="extract_pattern_from_page_text",
        )
        occurrence = args.get("occurrence", 1)
        if not isinstance(occurrence, int) or occurrence <= 0:
            raise PlanValidationError("extract_pattern_from_page_text requires positive integer 'occurrence'")
        group_index = args.get("group_index")
        if group_index is not None and (not isinstance(group_index, int) or group_index < 0):
            raise PlanValidationError(
                "extract_pattern_from_page_text requires non-negative integer 'group_index'"
            )
        PlanValidator._validate_group_index_reference(
            group_index=group_index,
            compiled_pattern=compiled_pattern,
            action_name="extract_pattern_from_page_text",
            field_name=None,
        )
        normalize_number = args.get("normalize_number")
        if normalize_number is not None and not isinstance(normalize_number, bool):
            raise PlanValidationError("extract_pattern_from_page_text requires boolean 'normalize_number'")
        number_type = args.get("number_type")
        if number_type is not None and number_type not in {"int", "float"}:
            raise PlanValidationError(
                "extract_pattern_from_page_text supports number_type in {'int','float'}"
            )
        strip_plus = args.get("strip_plus")
        if strip_plus is not None and not isinstance(strip_plus, bool):
            raise PlanValidationError("extract_pattern_from_page_text requires boolean 'strip_plus'")
        if not save_as:
            raise PlanValidationError("extract_pattern_from_page_text requires 'save_as'")

    @staticmethod
    def _validate_extract_structured_items(args: dict, save_as: str | None) -> None:
        if "pattern" not in args or not str(args.get("pattern", "")).strip():
            raise PlanValidationError("extract_structured_items requires non-empty 'pattern'")
        compiled_pattern = PlanValidator._compile_pattern(
            str(args.get("pattern")),
            action_name="extract_structured_items",
        )
        limit = args.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            raise PlanValidationError("extract_structured_items requires positive integer 'limit'")
        fields = args.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise PlanValidationError("extract_structured_items requires non-empty 'fields' dict")
        for field_name, spec in fields.items():
            if isinstance(spec, int):
                if spec <= 0:
                    raise PlanValidationError(
                        f"extract_structured_items field '{field_name}' requires positive group index"
                    )
                PlanValidator._validate_group_index_reference(
                    group_index=spec,
                    compiled_pattern=compiled_pattern,
                    action_name="extract_structured_items",
                    field_name=str(field_name),
                )
                continue
            if not isinstance(spec, dict):
                raise PlanValidationError(
                    f"extract_structured_items field '{field_name}' must be int group index or object rule"
                )
            group_index = spec.get("group_index", 1)
            if not isinstance(group_index, int) or group_index <= 0:
                raise PlanValidationError(
                    f"extract_structured_items field '{field_name}' requires positive integer 'group_index'"
                )
            PlanValidator._validate_group_index_reference(
                group_index=group_index,
                compiled_pattern=compiled_pattern,
                action_name="extract_structured_items",
                field_name=str(field_name),
            )
            if "normalize_number" in spec and not isinstance(spec["normalize_number"], bool):
                raise PlanValidationError(
                    f"extract_structured_items field '{field_name}' requires boolean 'normalize_number'"
                )
            if "number_type" in spec and spec["number_type"] not in {"int", "float"}:
                raise PlanValidationError(
                    f"extract_structured_items field '{field_name}' supports number_type in {{'int','float'}}"
                )
            if "strip_plus" in spec and not isinstance(spec["strip_plus"], bool):
                raise PlanValidationError(
                    f"extract_structured_items field '{field_name}' requires boolean 'strip_plus'"
                )
        if not save_as:
            raise PlanValidationError("extract_structured_items requires 'save_as'")

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

    @staticmethod
    def _validate_extract_value_from_section(args: dict, save_as: str | None) -> None:
        if not str(args.get("section_selector", "")).strip():
            raise PlanValidationError("extract_value_from_section requires non-empty 'section_selector'")
        if not (str(args.get("field_selector", "")).strip() or str(args.get("pattern", "")).strip()):
            raise PlanValidationError("extract_value_from_section requires 'field_selector' or 'pattern'")
        if not save_as:
            raise PlanValidationError("extract_value_from_section requires 'save_as'")

    @staticmethod
    def _validate_extract_structured_items_from_region(args: dict, save_as: str | None) -> None:
        if not str(args.get("region_selector", "")).strip():
            raise PlanValidationError("extract_structured_items_from_region requires non-empty 'region_selector'")
        if not str(args.get("container_selector", "")).strip():
            raise PlanValidationError("extract_structured_items_from_region requires non-empty 'container_selector'")
        PlanValidator._validate_extract_items(args, save_as)

    @staticmethod
    def _validate_compare_structured_values(args: dict, save_as: str | None) -> None:
        left_key = str(args.get("left_key", "section_a_data")).strip()
        right_key = str(args.get("right_key", "section_b_data")).strip()
        if not left_key or not right_key:
            raise PlanValidationError("compare_structured_values requires non-empty left_key/right_key")
        if not save_as:
            raise PlanValidationError("compare_structured_values requires 'save_as'")

    @staticmethod
    def _validate_assert_page_contains(args: dict) -> None:
        if not any(str(args.get(field, "")).strip() for field in ("selector", "text", "pattern")):
            raise PlanValidationError("assert_page_contains requires one of selector|text|pattern")

    @staticmethod
    def _validate_extract_value_near_anchor(args: dict, save_as: str | None) -> None:
        has_anchor_text = bool(str(args.get("anchor_text", "")).strip())
        anchor_candidates = args.get("anchor_candidates")
        has_anchor_candidates = isinstance(anchor_candidates, list) and any(
            isinstance(item, str) and item.strip() for item in anchor_candidates
        )
        if not has_anchor_text and not has_anchor_candidates:
            raise PlanValidationError(
                "extract_value_near_anchor requires non-empty 'anchor_text' or non-empty 'anchor_candidates'"
            )
        has_pattern = bool(str(args.get("value_pattern", "")).strip())
        has_type = bool(str(args.get("value_type", "")).strip())
        if not has_pattern and not has_type:
            raise PlanValidationError("extract_value_near_anchor requires non-empty 'value_pattern' or 'value_type'")
        if has_type and str(args.get("value_type", "")).strip().lower() not in {
            "article_count",
            "count",
            "number",
            "float",
            "rating",
            "email",
            "phone",
        }:
            raise PlanValidationError(
                "extract_value_near_anchor supports value_type in "
                "{'article_count','count','number','float','rating','email','phone'}"
            )
        direction = args.get("search_direction", "after")
        if direction not in {"after", "before", "around"}:
            raise PlanValidationError(
                "extract_value_near_anchor requires search_direction in {'after','before','around'}"
            )
        same_block_only = args.get("same_block_only")
        if same_block_only is not None and not isinstance(same_block_only, bool):
            raise PlanValidationError("extract_value_near_anchor requires boolean 'same_block_only'")
        max_distance_chars = args.get("max_distance_chars")
        if max_distance_chars is not None and (not isinstance(max_distance_chars, int) or max_distance_chars <= 0):
            raise PlanValidationError(
                "extract_value_near_anchor requires positive integer 'max_distance_chars'"
            )
        group_index = args.get("group_index")
        if group_index is not None and (not isinstance(group_index, int) or group_index < 0):
            raise PlanValidationError("extract_value_near_anchor requires non-negative integer 'group_index'")
        value_pattern = str(args.get("value_pattern", "")).strip()
        if value_pattern:
            compiled_pattern = PlanValidator._compile_pattern(
                value_pattern,
                action_name="extract_value_near_anchor",
            )
            PlanValidator._validate_group_index_reference(
                group_index=group_index,
                compiled_pattern=compiled_pattern,
                action_name="extract_value_near_anchor",
                field_name=None,
            )
        elif has_type and group_index is not None and group_index > 1:
            raise PlanValidationError(
                "extract_value_near_anchor with typed value_type supports only group_index in {0,1}"
            )
        normalize_number = args.get("normalize_number")
        if normalize_number is not None and not isinstance(normalize_number, bool):
            raise PlanValidationError("extract_value_near_anchor requires boolean 'normalize_number'")
        number_type = args.get("number_type")
        if number_type is not None and number_type not in {"int", "float"}:
            raise PlanValidationError("extract_value_near_anchor supports number_type in {'int','float'}")
        strip_plus = args.get("strip_plus")
        if strip_plus is not None and not isinstance(strip_plus, bool):
            raise PlanValidationError("extract_value_near_anchor requires boolean 'strip_plus'")
        if not save_as:
            raise PlanValidationError("extract_value_near_anchor requires 'save_as'")
        if anchor_candidates is not None:
            if not isinstance(anchor_candidates, list) or not all(
                isinstance(item, str) and item.strip() for item in anchor_candidates
            ):
                raise PlanValidationError("extract_value_near_anchor requires non-empty string entries in 'anchor_candidates'")
        anchor_matching_mode = args.get("anchor_matching_mode")
        if anchor_matching_mode is not None and anchor_matching_mode not in {"auto", "exact", "contains"}:
            raise PlanValidationError("extract_value_near_anchor supports anchor_matching_mode in {'auto','exact','contains'}")

    @staticmethod
    def _compile_pattern(pattern: str, *, action_name: str) -> re.Pattern[str]:
        try:
            return re.compile(pattern)
        except re.error as exc:
            raise PlanValidationError(
                f"{action_name} has invalid regex pattern: {exc}"
            ) from exc

    @staticmethod
    def _validate_group_index_reference(
        *,
        group_index: int | None,
        compiled_pattern: re.Pattern[str],
        action_name: str,
        field_name: str | None,
    ) -> None:
        if group_index is None:
            return
        if group_index == 0:
            return
        available_groups = compiled_pattern.groups
        if group_index > available_groups:
            field_suffix = f" field '{field_name}'" if field_name else ""
            raise PlanValidationError(
                f"{action_name}{field_suffix} references non-existent regex group_index={group_index}; "
                f"pattern exposes only {available_groups} capture group(s)"
            )

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
        if plan.constraints.max_verification_retries > GLOBAL_MAX_VERIFICATION_RETRIES:
            raise PlanValidationError("max_verification_retries exceeds global limit.")
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
        structured_nested_fields = self._collect_structured_nested_fields(plan)
        for field in plan.expected_result.required_fields:
            if field in TECHNICAL_ARTIFACT_FIELDS:
                continue
            if field in structured_nested_fields:
                continue
            if field not in saved_fields:
                raise PlanValidationError(
                    f"Required field '{field}' is not produced by any step."
                )


    def _validate_benchmark_contract(self, plan: TaskSpec, *, benchmark_context: dict | None) -> None:
        if not benchmark_context:
            return
        task_family = str(benchmark_context.get("task_family", "")).strip()
        required = required_contract_fields(
            task_family=task_family,
            scenario_required_fields=benchmark_context.get("required_top_level_fields"),
        )
        if not required:
            return

        if list(plan.expected_result.required_fields) != list(required):
            raise PlanValidationError(
                "Benchmark contract mismatch in expected_result.required_fields: "
                f"expected {required}, got {list(plan.expected_result.required_fields)}"
            )

        produced = [
            (step.action, str(step.save_as).strip())
            for step in plan.steps
            if isinstance(step.save_as, str) and step.save_as.strip()
        ]
        produced_set = {name for _, name in produced}
        missing = [field for field in required if field not in produced_set]
        if missing:
            raise PlanValidationError(
                "Benchmark contract missing required top-level fields: "
                f"{missing}. Produced={sorted(produced_set)}"
            )

        helper_actions = {"observe_page", "screenshot"}
        extra_business_fields = sorted(
            name
            for action, name in produced
            if name not in set(required)
            and name not in TECHNICAL_ARTIFACT_FIELDS
            and action not in helper_actions
        )
        if extra_business_fields:
            raise PlanValidationError(
                "Benchmark contract disallows extra top-level business fields: "
                f"{extra_business_fields}. Required={required}"
            )

    @staticmethod
    def _collect_structured_nested_fields(plan: TaskSpec) -> set[str]:
        nested_fields: set[str] = set()
        for step in plan.steps:
            if step.action == "extract_items":
                fields = step.args.get("fields")
                if isinstance(fields, dict):
                    nested_fields.update(str(name) for name in fields.keys())
            if step.action == "extract_structured_items":
                fields = step.args.get("fields")
                if isinstance(fields, dict):
                    nested_fields.update(str(name) for name in fields.keys())
        return nested_fields
