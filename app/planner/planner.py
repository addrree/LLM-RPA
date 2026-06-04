import json
import re
from typing import Any
from urllib.parse import urlparse

from app.planner.action_vocab import (
    canonical_structured_intent,
    coalesce_field_schema_steps,
    looks_like_css_selector,
    normalize_plan_action_aliases,
    normalize_required_field_aliases,
    normalize_intent_alias,
    normalize_schema_fields_step,
    PlannerValidationFailed,
    raise_for_invalid_plan_actions,
    semantic_intent_for_structured_step,
)
from app.planner.prompts import (
    INITIAL_PLANNER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_profile_planner_prompt,
    build_benchmark_planner_prompt,
)
from app.planner.task_router import TaskRoute, TaskRouter
from app.schemas.execution import GenerationMetadata, LLMArtifact
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None
        self.last_initial_artifact: LLMArtifact | None = None
        self.last_action_oov_detected = False
        self.last_normalized_action_aliases: list[dict[str, str]] = []
        self.task_router = TaskRouter()
        self.last_task_route: TaskRoute | None = None
        self.last_profile_diagnostics: dict[str, Any] = {}

    def route_goal(self, user_goal: str, benchmark_context: dict | None = None) -> TaskRoute:
        route = self.task_router.route(user_goal, benchmark_context=benchmark_context)
        self.last_task_route = route
        self.last_profile_diagnostics = route.diagnostics()
        return route

    def build_plan(
        self,
        user_goal: str,
        benchmark_context: dict | None = None,
        images_base64: list[str] | None = None,
        task_route: TaskRoute | None = None,
    ) -> TaskSpec:
        route = task_route or self.route_goal(user_goal, benchmark_context=benchmark_context)
        system_prompt = build_profile_planner_prompt(route.profile, stage="planner")
        if benchmark_context:
            system_prompt = build_benchmark_planner_prompt(
                task_family=str(benchmark_context.get("task_family", "unknown")),
                allowed_actions=list(benchmark_context.get("allowed_actions", [])),
            )
        route.profile.profile_prompt_length = len(system_prompt)
        self.last_task_route = route
        self.last_profile_diagnostics = route.diagnostics()
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=system_prompt,
            user_prompt=user_goal,
            stage="planner",
            images_base64=images_base64,
        )
        self.last_artifact = artifact
        normalized, action_oov_detected = normalize_plan_action_aliases(artifact.parsed_response)
        self.last_normalized_action_aliases = list(normalized.get("_normalized_action_aliases") or [])
        raise_for_invalid_plan_actions(
            normalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        normalized = self._normalize_plan_envelope(normalized, user_goal, benchmark_context=benchmark_context)
        normalized = self._normalize_required_fields_against_steps(normalized)
        self.last_action_oov_detected = action_oov_detected
        raise_for_invalid_plan_actions(
            normalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        return TaskSpec.model_validate(normalized)

    def build_initial_plan(self, user_goal: str) -> TaskSpec:
        try:
            artifact = self.llm_client.generate_planner_artifact(
                system_prompt=INITIAL_PLANNER_SYSTEM_PROMPT,
                user_prompt=user_goal,
                stage="initial_planner",
            )
        except Exception as exc:  # noqa: BLE001
            try:
                artifact = self._retry_initial_generation_after_error(user_goal=user_goal, error=str(exc))
            except Exception as retry_exc:  # noqa: BLE001
                fallback = self._build_initial_fallback(user_goal=user_goal)
                artifact = LLMArtifact(
                    raw_response=json.dumps(
                        {
                            "fallback_reason": "initial_planner_generation_failed",
                            "error": str(exc),
                            "retry_error": str(retry_exc),
                            "plan": fallback,
                        },
                        ensure_ascii=False,
                    ),
                    parsed_response=fallback,
                    generation=GenerationMetadata(
                        backend="local",
                        model="initial_planner_fallback",
                        source="fallback",
                        fallback_used=True,
                    ),
                )
        self.last_initial_artifact = artifact

        parsed = artifact.parsed_response
        # Fail-safe for non-compliant outputs.
        if not self._is_valid_initial_shape(parsed):
            parsed = self._repair_initial_plan(
                user_goal=user_goal,
                invalid_payload=parsed,
                artifact=artifact,
            )
        normalized = self._normalize_initial_plan(parsed, user_goal)
        normalized, action_oov_detected = normalize_plan_action_aliases(normalized)
        self.last_normalized_action_aliases = list(normalized.get("_normalized_action_aliases") or [])
        self.last_action_oov_detected = action_oov_detected
        raise_for_invalid_plan_actions(normalized)
        return TaskSpec.model_validate(normalized)

    def _retry_initial_generation_after_error(self, *, user_goal: str, error: str) -> LLMArtifact:
        repair_payload = {
            "user_goal": user_goal,
            "previous_error": error,
            "repair_request": (
                "Return only one valid JSON TaskSpec object for initial observation. "
                "Use exactly open_url, observe_page, finish. If the goal names a public website/service "
                "without a URL, infer its canonical public HTTPS homepage from general knowledge. "
                "Do not use placeholder or dummy URLs."
            ),
        }
        return self.llm_client.generate_planner_artifact(
            system_prompt=INITIAL_PLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(repair_payload, ensure_ascii=False),
            stage="initial_planner_repair",
        )

    @staticmethod
    def _is_valid_initial_shape(payload: dict) -> bool:
        try:
            steps = payload.get("steps", [])
            actions = [s.get("action") for s in steps]
            return actions == ["open_url", "observe_page", "finish"] and bool(Planner._extract_first_url(payload))
        except Exception:
            return False

    def _repair_initial_plan(self, *, user_goal: str, invalid_payload: dict, artifact: LLMArtifact) -> dict:
        try:
            fallback = self._build_initial_fallback(user_goal=user_goal, candidate_payload=invalid_payload)
            self.last_initial_artifact = LLMArtifact(
                raw_response=json.dumps(fallback, ensure_ascii=False),
                parsed_response=fallback,
                generation=artifact.generation,
            )
            return fallback
        except PlannerValidationFailed:
            pass

        repair_payload = {
            "user_goal": user_goal,
            "invalid_initial_plan": invalid_payload,
            "repair_request": (
                "Return a valid initial observation TaskSpec only. It must have exactly these actions in order: "
                "open_url, observe_page, finish. Infer start_url from the user goal if a public site/service is named. "
                "Use its canonical public HTTPS homepage from general knowledge, and never use placeholder or dummy URLs. "
                "Do not extract final data in this initial plan."
            ),
        }
        try:
            repaired = self.llm_client.generate_planner_artifact(
                system_prompt=INITIAL_PLANNER_SYSTEM_PROMPT,
                user_prompt=json.dumps(repair_payload, ensure_ascii=False, indent=2),
                stage="initial_planner_repair",
            )
        except Exception as exc:  # noqa: BLE001
            try:
                fallback = self._build_initial_fallback(user_goal=user_goal, candidate_payload=invalid_payload)
            except PlannerValidationFailed as fallback_exc:
                details = dict(fallback_exc.diagnostics)
                details["repair_error"] = str(exc)
                raise PlannerValidationFailed(details) from exc
            self.last_initial_artifact = LLMArtifact(
                raw_response=json.dumps(
                    {
                        "fallback_reason": "initial_planner_repair_failed",
                        "repair_error": str(exc),
                        "plan": fallback,
                    },
                    ensure_ascii=False,
                ),
                parsed_response=fallback,
                generation=artifact.generation,
            )
            return fallback
        self.last_initial_artifact = repaired
        if self._is_valid_initial_shape(repaired.parsed_response):
            return repaired.parsed_response
        fallback = self._build_initial_fallback(
            user_goal=user_goal,
            candidate_payload=repaired.parsed_response,
        )
        self.last_initial_artifact = LLMArtifact(
            raw_response=json.dumps(fallback, ensure_ascii=False),
            parsed_response=fallback,
            generation=repaired.generation,
        )
        return fallback

    @staticmethod
    def _build_initial_fallback(user_goal: str, candidate_payload: dict | None = None) -> dict:
        url = Planner._extract_first_url(candidate_payload) or Planner._extract_first_url(user_goal)
        url = Planner._normalize_url_candidate(url)
        if not url:
            raise PlannerValidationFailed(
                {
                    "failure_stage": "planner_validation_failed",
                    "reason": "missing_start_url",
                    "message": "Cannot build fallback plan without an explicit URL from the model, goal, or context.",
                }
            )
        domain = re.sub(r"^https?://", "", url).split("/")[0]
        return {
            "goal": user_goal,
            "start_url": url,
            "allowed_domains": [domain],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 30},
            "expected_result": {
                "description": "Observe landing page context",
                "required_fields": ["page_snapshot"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": url}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }

    @staticmethod
    def _extract_first_url(value: object) -> str:
        if isinstance(value, str):
            return Planner._normalize_url_candidate(value)
        if isinstance(value, dict):
            for key in ("start_url", "url"):
                found = Planner._extract_first_url(value.get(key))
                if found:
                    return found
            steps = value.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    found = Planner._extract_first_url(step)
                    if found:
                        return found
            args = value.get("args")
            if isinstance(args, dict):
                found = Planner._extract_first_url(args)
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = Planner._extract_first_url(item)
                if found:
                    return found
        return ""

    @staticmethod
    def _normalize_url_candidate(value: object) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip()
        if not text:
            return ""

        explicit = re.search(r"https?://[^\s\"'<>]+", text, flags=re.IGNORECASE)
        if explicit:
            candidate = explicit.group(0).rstrip(".,);]")
            if text.casefold().startswith(("http://", "https://")):
                remainder = text[len(explicit.group(0)):].strip()
                if remainder and not re.fullmatch(r"[.,);\\]]+", remainder):
                    return ""
            parsed = urlparse(candidate)
            if Planner._is_valid_url_host(parsed.netloc):
                return candidate
            return ""

        domain_like = re.search(
            r"(?<![@\w.-])(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+(?:/[^\s\"'<>]*)?",
            text,
            flags=re.IGNORECASE,
        )
        if not domain_like:
            return ""

        candidate = domain_like.group(0).rstrip(".,);]")
        host = candidate.split("/", 1)[0].lower()
        labels = host.split(".")
        if len(labels) < 2 or any(not label or label.startswith("-") or label.endswith("-") for label in labels):
            return ""
        tld = labels[-1]
        if not (2 <= len(tld) <= 24 and tld.isalpha()):
            return ""
        return f"https://{candidate}"

    @staticmethod
    def _is_valid_url_host(netloc: str) -> bool:
        host = str(netloc or "").split("@")[-1].split(":", 1)[0].strip().strip(".")
        if not host or host != str(netloc or "").split("@")[-1].split(":", 1)[0].strip():
            return False
        labels = host.split(".")
        if len(labels) < 2 or any(not label or label.startswith("-") or label.endswith("-") for label in labels):
            return False
        tld = labels[-1]
        return bool(2 <= len(tld) <= 24 and tld.isalpha())

    @staticmethod
    def _normalize_initial_plan(raw_plan: dict, user_goal: str) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}

        steps = plan.get("steps")
        if not isinstance(steps, list) and isinstance(plan.get("actions"), list):
            steps = plan.pop("actions")
        if not isinstance(steps, list) and isinstance(plan.get("tasks"), list):
            steps = plan.pop("tasks")
        if not isinstance(steps, list) and isinstance(plan.get("order"), list):
            steps = plan.pop("order")
        if not isinstance(steps, list):
            steps = []

        normalized_steps: list[dict] = []
        has_observe_page = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            current = dict(step)
            args = current.get("args")
            if not isinstance(args, dict):
                args = {}
            current["args"] = dict(args)

            if current.get("action") == "open_url" and "url" not in current["args"] and "url" in current:
                current["args"]["url"] = current["url"]
                current.pop("url", None)
            if current.get("action") == "observe_page":
                has_observe_page = True
                save_as = current.get("save_as")
                if not isinstance(save_as, str) or not save_as.strip():
                    current["save_as"] = "page_snapshot"

            normalized_steps.append(current)

        has_finish = any(step.get("action") == "finish" for step in normalized_steps)
        if not has_finish:
            normalized_steps.append({"action": "finish", "args": {}})

        for idx, step in enumerate(normalized_steps, start=1):
            if not isinstance(step.get("step_id"), int):
                step["step_id"] = idx

        expected_result = plan.get("expected_result")
        if not isinstance(expected_result, dict):
            expected_result = {}
        if not expected_result.get("description"):
            expected_result["description"] = "Collect page snapshot for replanning"
        if not isinstance(expected_result.get("required_fields"), list):
            expected_result["required_fields"] = ["page_snapshot"]
        elif has_observe_page and "page_snapshot" not in expected_result["required_fields"]:
            expected_result["required_fields"] = [*expected_result["required_fields"], "page_snapshot"]

        start_url = plan.get("start_url")
        if not start_url:
            for step in normalized_steps:
                if step.get("action") == "open_url":
                    candidate_url = step.get("args", {}).get("url")
                    if candidate_url:
                        start_url = candidate_url
                        break
        start_url = Planner._normalize_url_candidate(start_url)

        if not start_url:
            raise PlannerValidationFailed(
                {
                    "failure_stage": "planner_validation_failed",
                    "reason": "missing_start_url",
                    "message": "Initial plan has no start_url and no open_url step with args.url.",
                }
        )

        for step in normalized_steps:
            if step.get("action") == "open_url":
                url = Planner._normalize_url_candidate(step.get("args", {}).get("url") or start_url)
                step.setdefault("args", {})["url"] = url or start_url
                break

        allowed_domains = plan.get("allowed_domains")
        netloc = urlparse(str(start_url)).netloc
        if (
            not isinstance(allowed_domains, list)
            or not allowed_domains
            or (netloc and netloc not in {str(domain).strip() for domain in allowed_domains})
        ):
            allowed_domains = [netloc] if netloc else []

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 30}
        else:
            constraints = {
                "max_steps": constraints.get("max_steps", 4),
                "max_replans": constraints.get("max_replans", 1),
                "max_verification_retries": constraints.get("max_verification_retries", 3),
                "timeout_sec": constraints.get("timeout_sec", 30),
            }

        return {
            "goal": plan.get("goal") or user_goal,
            "start_url": start_url,
            "allowed_domains": allowed_domains,
            "constraints": constraints,
            "expected_result": expected_result,
            "steps": normalized_steps,
        }

    @staticmethod
    def _normalize_required_fields_against_steps(plan: dict) -> dict:
        payload = dict(plan) if isinstance(plan, dict) else {}
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return payload

        produced = {
            step.get("save_as")
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str) and step.get("save_as").strip()
        }
        expected = payload.get("expected_result")
        if not isinstance(expected, dict):
            return payload
        required = expected.get("required_fields")
        if not isinstance(required, list):
            return payload

        business_produced = sorted(
            str(field).strip()
            for field in produced
            if str(field).strip() and str(field).strip() not in {"page_snapshot", "clicked_text", "final_url", "page_title"}
        )
        required_clean = [str(field).strip() for field in required if str(field).strip()]
        filtered = [str(field).strip() for field in required if str(field).strip() in produced]
        if business_produced and any(field not in produced for field in required_clean):
            expected["required_fields"] = business_produced
        elif filtered:
            expected["required_fields"] = filtered
        elif produced:
            expected["required_fields"] = sorted(produced)
        payload["expected_result"] = expected
        return payload

    @staticmethod
    def _normalize_structured_fields(fields: object, required_fields: list[str]) -> object:
        if isinstance(fields, dict):
            return fields
        if not isinstance(fields, list) or not fields:
            return fields
        names = [field for field in required_fields if field]
        normalized: dict[str, object] = {}
        for index, spec in enumerate(fields, start=1):
            if isinstance(spec, str) and spec.strip():
                normalized[spec.strip()] = {"group_index": index}
                continue
            field_name = names[index - 1] if index - 1 < len(names) else f"field_{index}"
            if isinstance(spec, dict):
                normalized[field_name] = dict(spec)
            elif isinstance(spec, int):
                normalized[field_name] = {"group_index": spec}
            else:
                normalized[field_name] = {"group_index": index}
        return normalized

    @classmethod
    def _normalize_extract_item_fields(cls, fields: object) -> object:
        if not isinstance(fields, dict):
            return fields
        normalized: dict[str, object] = {}
        for field_name, rule in fields.items():
            if not isinstance(rule, dict):
                normalized[field_name] = rule
                continue
            current = dict(rule)
            if "attr" not in current and current.get("attribute") is not None:
                attribute = str(current.pop("attribute") or "").strip()
                if attribute and attribute.lower() not in {"text", "inner_text", "innertext"}:
                    current["attr"] = attribute
            normalized[field_name] = current
        return normalized

    @staticmethod
    def _simplify_semantic_click_target(target_text: str) -> str:
        text = str(target_text or "").strip()
        if not text:
            return ""
        patterns = [
            r"^(?:.+?\s+)?(?:link|result)\s+for\s+(.+)$",
            r"^visible\s+link\s+with\s+text\s+(.+)$",
            r"^link\s+with\s+text\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                simplified = match.group(1).strip().strip("'\"")
                return simplified or text
        return text

    @staticmethod
    def _goal_requests_clicked_text(user_goal: str) -> bool:
        folded = str(user_goal or "").casefold()
        if not folded:
            return False
        return any(token in folded for token in ("click", "clicked", "press", "open link", "нажм", "клик", "перейд"))

    @staticmethod
    def _infer_collection_output_key(*, user_goal: str, args: dict[str, Any], required_fields: list[str]) -> str:
        explicit_output = str(args.get("output_key", "") or "").strip()
        if explicit_output:
            return explicit_output
        candidates = [
            str(field or "").strip().removesuffix("[]")
            for field in required_fields
            if str(field or "").strip()
            and str(field or "").strip() not in {"page_snapshot", "clicked_text", "final_url", "page_title"}
        ]
        if len(candidates) == 1:
            return candidates[0]
        return "items"

    @classmethod
    def _normalize_plan_envelope(
        cls,
        raw_plan: dict,
        user_goal: str,
        *,
        benchmark_context: dict | None = None,
    ) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
        steps = plan.get("steps")
        if not isinstance(steps, list) and isinstance(plan.get("actions"), list):
            steps = plan.pop("actions")
        if not isinstance(steps, list) and isinstance(plan.get("tasks"), list):
            steps = plan.pop("tasks")
        if not isinstance(steps, list) and isinstance(plan.get("order"), list):
            steps = plan.pop("order")
        if not isinstance(steps, list):
            steps = []

        expected_result = plan.get("expected_result")
        if not isinstance(expected_result, dict):
            expected_result = {}
        required_fields = expected_result.get("required_fields")
        if not isinstance(required_fields, list):
            required_fields = []
        required_fields = [str(field).strip() for field in required_fields if str(field).strip()]
        if not required_fields:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                nested_expected = step.get("expected_result")
                if isinstance(nested_expected, dict) and isinstance(nested_expected.get("required_fields"), list):
                    required_fields = [
                        str(field).strip()
                        for field in nested_expected.get("required_fields", [])
                        if str(field).strip()
                    ]
                    if required_fields:
                        break
        if not required_fields and isinstance(benchmark_context, dict):
            context_fields = benchmark_context.get("required_fields")
            if isinstance(context_fields, list):
                required_fields = [str(field).strip() for field in context_fields if str(field).strip()]
        required_fields = normalize_required_field_aliases(required_fields)

        normalized_steps: list[dict] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            current = dict(step)
            args = current.get("args")
            if not isinstance(args, dict) and isinstance(current.get("params"), dict):
                args = current.pop("params")
            if not isinstance(args, dict) and isinstance(current.get("action_params"), dict):
                args = current.pop("action_params")
            if not isinstance(args, dict):
                args = {}
            current["args"] = dict(args)
            if current.get("save_as") is not None and not isinstance(current.get("save_as"), str):
                current.pop("save_as", None)
            action = str(current.get("action", "")).strip()
            reserved_step_keys = {"step_id", "action", "args", "params", "save_as", "expected_result"}
            for key in list(current.keys()):
                if key in reserved_step_keys:
                    continue
                if key not in current["args"]:
                    current["args"][key] = current.pop(key)
            if action == "open_url" and "url" not in current["args"] and current.get("url") is not None:
                current["args"]["url"] = current.pop("url")
            if "output_key" not in current["args"] and current.get("output_key") is not None:
                current["args"]["output_key"] = current.pop("output_key")
            if isinstance(current.get("save_as"), str) and current["save_as"].strip():
                save_aliases = normalize_required_field_aliases([current["save_as"].strip()])
                if save_aliases:
                    current["save_as"] = save_aliases[0]
            if "target" not in current["args"] and "target_text" not in current["args"] and current["args"].get("semantic_target") is not None:
                current["args"]["target"] = current["args"].pop("semantic_target")
            if (
                not str(current.get("save_as", "") or "").strip()
                and str(current["args"].get("output_key", "") or "").strip()
            ):
                current["save_as"] = str(current["args"]["output_key"]).strip()
            current = normalize_schema_fields_step(
                current,
                goal=user_goal,
                required_fields=required_fields,
            )
            action = str(current.get("action", action)).strip()
            if not isinstance(current.get("step_id"), int):
                current["step_id"] = index
            if action == "extract_structured_items" and "pattern" not in current["args"]:
                allowed_actions = set()
                if isinstance(benchmark_context, dict) and isinstance(benchmark_context.get("allowed_actions"), list):
                    allowed_actions = {str(item).strip() for item in benchmark_context.get("allowed_actions", [])}
                extract_by_intent_allowed = not allowed_actions or "extract_by_intent" in allowed_actions
                extract_items_allowed = not allowed_actions or "extract_items" in allowed_actions
                visible_links_allowed = not allowed_actions or "extract_visible_links" in allowed_actions
                row_condition_allowed = not allowed_actions or "find_row_by_condition" in allowed_actions
                explicit_structured_intent = str(
                    current["args"].get("intent", "") or current["args"].get("item_type", "") or ""
                ).strip()
                structured_intent = (
                    canonical_structured_intent(explicit_structured_intent)
                    or semantic_intent_for_structured_step(current)
                )
                if structured_intent and extract_by_intent_allowed:
                    action = "extract_by_intent"
                    current["action"] = action
                    output_key = current["args"].get("output_key") or cls._infer_collection_output_key(
                        user_goal=user_goal,
                        args=current["args"],
                        required_fields=required_fields,
                    )
                    current["args"] = {
                        "intent": structured_intent,
                        **({"fields": current["args"].get("fields")} if isinstance(current["args"].get("fields"), dict) else {}),
                        **({"region_candidates": current["args"].get("region_candidates")} if isinstance(current["args"].get("region_candidates"), list) else {}),
                        **({"region_hint": current["args"].get("region_hint")} if str(current["args"].get("region_hint", "") or "").strip() else {}),
                        **({"numeric_value_required": current["args"].get("numeric_value_required")} if isinstance(current["args"].get("numeric_value_required"), bool) else {}),
                        **({"output_key": output_key} if output_key else {}),
                        "limit": current["args"]["limit"] if isinstance(current["args"].get("limit"), int) and current["args"]["limit"] > 0 else 20,
                    }
                    if output_key and not str(current.get("save_as", "") or "").strip():
                        current["save_as"] = str(output_key)
                elif (
                    item_selector := (current["args"].get("item_selector") or current["args"].get("container_selector"))
                ) and (
                    item_selector
                    and looks_like_css_selector(item_selector)
                    and isinstance(current["args"].get("fields"), dict)
                    and extract_items_allowed
                ):
                    action = "extract_items"
                    current["action"] = action
                    current["args"] = {
                        "container_selector": str(item_selector),
                        "fields": cls._normalize_extract_item_fields(current["args"].get("fields")),
                        "limit": current["args"].get("limit", 5),
                        **({"output_key": current["args"]["output_key"]} if current["args"].get("output_key") else {}),
                    }
                elif (
                    any(key in current["args"] for key in ("filter", "condition", "conditions", "where"))
                    and row_condition_allowed
                ):
                    action = "find_row_by_condition"
                    current["action"] = action
                    condition = (
                        current["args"].get("condition")
                        or current["args"].get("conditions")
                        or current["args"].get("filter")
                        or current["args"].get("where")
                    )
                    if (
                        isinstance(condition, list)
                        and len(condition) == 1
                        and isinstance(condition[0], dict)
                    ):
                        condition = condition[0]
                    output_key = current["args"].get("output_key") or cls._infer_collection_output_key(
                        user_goal=user_goal,
                        args=current["args"],
                        required_fields=required_fields,
                    )
                    current["args"] = {
                        "condition": condition,
                        **({"limit": current["args"]["limit"]} if current["args"].get("limit") else {}),
                        **({"output_key": output_key} if output_key else {}),
                    }
                    if output_key and not str(current.get("save_as", "") or "").strip():
                        current["save_as"] = str(output_key)
                elif "pattern" in current["args"] and (
                    not isinstance(current["args"].get("limit"), int) or current["args"].get("limit") <= 0
                ):
                    current["args"]["limit"] = 20
                elif visible_links_allowed:
                    action = "extract_visible_links"
                    current["action"] = action
                    output_key = current["args"].get("output_key") or "links"
                    current["args"] = {
                        key: value
                        for key, value in current["args"].items()
                        if key in {"output_key", "limit", "min_text_length", "target", "intent"}
                    }
                    if output_key and "output_key" not in current["args"]:
                        current["args"]["output_key"] = output_key
                    if output_key and not str(current.get("save_as", "") or "").strip():
                        current["save_as"] = str(output_key)
            if action == "extract_structured_items" and "fields" in current["args"]:
                current["args"]["fields"] = cls._normalize_structured_fields(
                    current["args"].get("fields"),
                    required_fields,
                )
            if action == "extract_value_near_anchor":
                if "anchor_text" not in current["args"] and current["args"].get("anchor") is not None:
                    current["args"]["anchor_text"] = current["args"].pop("anchor")
                if "anchor_text" not in current["args"] and current["args"].get("target") is not None:
                    current["args"]["anchor_text"] = current["args"].get("target")
                anchor_hint = " ".join(
                    str(current["args"].get(key, "") or "")
                    for key in ("anchor_text", "target", "value_type")
                ).casefold()
                save_as = str(current.get("save_as", "") or "").strip()
                if save_as == "final_url" or "current url" in anchor_hint or "final url" in anchor_hint:
                    action = "extract_by_intent"
                    current["action"] = action
                    current["args"] = {"intent": "current_url"}
            if action == "extract_by_intent" and "intent" not in current["args"] and isinstance(current["args"].get("intents"), list):
                current["args"]["intent"] = "row_fields"
            if action == "extract_by_intent" and str(current["args"].get("intent", "") or "").strip():
                current["args"]["intent"] = normalize_intent_alias(current["args"].get("intent"))
            if action == "extract_text":
                save_as = str(current.get("save_as", "") or "").strip()
                selector_hint = str(current["args"].get("selector", "") or "").strip().casefold()
                target_hint = str(current["args"].get("target", "") or "").strip().casefold()
                if save_as == "page_title" or selector_hint == "title" or target_hint in {"title", "page title"}:
                    action = "extract_by_intent"
                    current["action"] = action
                    current["args"] = {"intent": "page_title"}
            if action == "click_by_semantic_target":
                if "target_text" not in current["args"] and current["args"].get("text") is not None:
                    current["args"]["target_text"] = current["args"].pop("text")
                if "target_text" not in current["args"] and current["args"].get("target") is not None:
                    current["args"]["target_text"] = current["args"].pop("target")
                target_text = str(current["args"].get("target_text", "") or "")
                simplified_target = cls._simplify_semantic_click_target(target_text)
                if simplified_target and simplified_target != target_text:
                    current["args"]["target_text"] = simplified_target
                    target_text = simplified_target
                option_match = re.search(r"\b([A-Za-z][\w-]{1,30})\s+(?:or|или)\s+([A-Za-z][\w-]{1,30})\b", target_text, flags=re.IGNORECASE)
                if option_match and "target_candidates" not in current["args"]:
                    current["args"]["target_candidates"] = [option_match.group(1), option_match.group(2)]
                if "|" in target_text and "target_candidates" not in current["args"]:
                    candidates = [part.strip() for part in target_text.split("|") if part.strip()]
                    if candidates:
                        current["args"]["target_candidates"] = candidates
            if action == "fill_by_semantic_target":
                if "value" not in current["args"] and current["args"].get("query") is not None:
                    current["args"]["value"] = current["args"].pop("query")
            if action == "observe_page" and not str(current.get("save_as", "") or "").strip():
                current["save_as"] = "page_snapshot"
            if (
                action == "click_by_semantic_target"
                and ("clicked_text" in required_fields or cls._goal_requests_clicked_text(user_goal))
                and not str(current.get("save_as", "") or "").strip()
            ):
                current["save_as"] = "clicked_text"
            collection_like_action = action in {
                "extract_items",
                "extract_structured_items",
                "extract_visible_links",
                "find_row_by_condition",
            }
            if action == "extract_by_intent":
                intent = str(current["args"].get("intent", "") or "").strip().casefold()
                collection_like_action = intent in {
                    "",
                    "visible_links",
                    "extract_visible_links",
                    "links",
                    "card_items",
                    "cards",
                    "table_rows",
                    "rows",
                }
            if (
                collection_like_action
                and not str(current.get("save_as", "") or "").strip()
                and not str(current["args"].get("output_key", "") or "").strip()
            ):
                inferred_output_key = (
                    "links"
                    if action == "extract_visible_links"
                    else cls._infer_collection_output_key(
                        user_goal=user_goal,
                        args=current["args"],
                        required_fields=required_fields,
                    )
                )
                if inferred_output_key:
                    current["args"]["output_key"] = inferred_output_key
                    current["save_as"] = inferred_output_key
            if (
                action in {
                    "extract_text",
                    "extract_items",
                    "extract_structured_items",
                    "extract_by_intent",
                    "extract_visible_links",
                    "extract_pattern_from_page_text",
                    "extract_text_near_text",
                    "extract_value_near_anchor",
                    "find_row_by_condition",
                    "visual_observe",
                    "visual_extract_object_count",
                }
                and not str(current.get("save_as", "") or "").strip()
                and not str(current["args"].get("output_key", "") or "").strip()
                and required_fields
            ):
                current["save_as"] = required_fields[0]
            normalized_steps.append(current)

        normalized_steps = coalesce_field_schema_steps(
            normalized_steps,
            goal=user_goal,
            required_fields=required_fields,
        )

        produced_fields = {
            str(step.get("save_as", "") or "").strip()
            for step in normalized_steps
            if str(step.get("save_as", "") or "").strip()
        }

        business_produced_fields = sorted(
            field for field in produced_fields if field not in {"page_snapshot", "clicked_text", "final_url", "page_title"}
        )
        if business_produced_fields and set(required_fields).issubset({"final_url", "page_title"}):
            required_fields = business_produced_fields
        metadata_fields = [field for field in required_fields if field in {"final_url", "page_title"} and field not in produced_fields]
        if metadata_fields:
            insert_index = next((idx for idx, step in enumerate(normalized_steps) if step.get("action") == "finish"), len(normalized_steps))
            metadata_steps = [
                {
                    "step_id": 0,
                    "action": "extract_by_intent",
                    "args": {"intent": "current_url" if field == "final_url" else "page_title"},
                    "save_as": field,
                }
                for field in metadata_fields
            ]
            normalized_steps[insert_index:insert_index] = metadata_steps

        if not any(step.get("action") == "finish" for step in normalized_steps):
            normalized_steps.append({"step_id": len(normalized_steps) + 1, "action": "finish", "args": {}})
        for index, step in enumerate(normalized_steps, start=1):
            step["step_id"] = index

        start_url = Planner._normalize_url_candidate(str(plan.get("start_url") or "").strip())
        if not start_url and isinstance(benchmark_context, dict):
            start_url = Planner._normalize_url_candidate(str(benchmark_context.get("start_url") or "").strip())
        if not start_url:
            start_url = Planner._normalize_url_candidate(user_goal or "")
        if not start_url:
            raise PlannerValidationFailed(
                {
                    "failure_stage": "planner_validation_failed",
                    "reason": "missing_start_url",
                    "message": "Final plan has no start_url, benchmark start_url, or URL in the user goal.",
                }
            )

        allowed_domains = plan.get("allowed_domains")
        netloc = urlparse(start_url).netloc
        if (
            not isinstance(allowed_domains, list)
            or not allowed_domains
            or (netloc and netloc not in {str(domain).strip() for domain in allowed_domains})
        ):
            allowed_domains = [netloc] if netloc else []

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            max_steps = 8
            if isinstance(benchmark_context, dict):
                try:
                    max_steps = int(benchmark_context.get("max_steps") or max_steps)
                except (TypeError, ValueError):
                    max_steps = 8
            constraints = {
                "max_steps": max_steps,
                "max_replans": 1,
                "max_verification_retries": 1,
                "timeout_sec": 30,
            }
        else:
            constraints = {
                "max_steps": constraints.get("max_steps", 8),
                "max_replans": constraints.get("max_replans", 1),
                "max_verification_retries": constraints.get("max_verification_retries", 1),
                "timeout_sec": constraints.get("timeout_sec", 30),
            }

        if not required_fields:
            required_fields = [
                str(step.get("save_as")).strip()
                for step in normalized_steps
                if str(step.get("save_as", "") or "").strip()
                and str(step.get("save_as")).strip() != "page_snapshot"
            ]
        expected_result = {
            "description": expected_result.get("description") or "Complete the requested web automation goal.",
            "required_fields": required_fields,
        }

        return {
            "goal": plan.get("goal") or user_goal,
            "start_url": start_url,
            "allowed_domains": allowed_domains,
            "constraints": constraints,
            "expected_result": expected_result,
            "steps": normalized_steps,
        }
