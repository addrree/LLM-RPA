import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from app.planner.action_vocab import (
    PACKAGE_METADATA_FIELDS,
    build_semantic_region_fields_args,
    coalesce_package_metadata_steps,
    goal_requests_semantic_region_fields,
    default_output_key_for_intent,
    normalize_plan_action_aliases,
    normalize_required_field_alias,
    normalize_intent_alias,
    PlannerValidationFailed,
    raise_for_invalid_plan_actions,
    semantic_intent_for_structured_step,
)
from app.planner.prompts import (
    CORRECTIVE_REPLANNER_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
    build_benchmark_replanner_prompt,
    build_profile_replanner_prompt,
)
from app.planner.task_router import TaskRoute, TaskRouter
from app.schemas.execution import GenerationMetadata, LLMArtifact
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


def _is_valid_url_host(netloc: str) -> bool:
    host = str(netloc or "").split("@")[-1].split(":", 1)[0].strip().strip(".")
    if not host or host != str(netloc or "").split("@")[-1].split(":", 1)[0].strip():
        return False
    labels = host.split(".")
    if len(labels) < 2 or any(not label or label.startswith("-") or label.endswith("-") for label in labels):
        return False
    tld = labels[-1]
    return bool(2 <= len(tld) <= 24 and tld.isalpha())


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
        return candidate if _is_valid_url_host(parsed.netloc) else ""
    domain_like = re.search(
        r"(?<![@\w.-])(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+(?:/[^\s\"'<>]*)?",
        text,
        flags=re.IGNORECASE,
    )
    if not domain_like:
        return ""
    candidate = domain_like.group(0).rstrip(".,);]")
    host = candidate.split("/", 1)[0].lower()
    if not _is_valid_url_host(host):
        return ""
    return f"https://{candidate}"


class Replanner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None
        self.last_action_oov_detected = False
        self.last_normalized_action_aliases: list[dict[str, str]] = []
        self.task_router = TaskRouter()
        self.last_task_route: TaskRoute | None = None
        self.last_profile_diagnostics: dict = {}

    def route_goal(self, user_goal: str, benchmark_context: dict | None = None) -> TaskRoute:
        route = self.task_router.route(user_goal, benchmark_context=benchmark_context)
        self.last_task_route = route
        self.last_profile_diagnostics = route.diagnostics()
        return route

    def revise_plan(
        self,
        user_goal: str,
        page_snapshot: PageSnapshot,
        previous_plan: TaskSpec | None = None,
        validation_error: str | None = None,
        invalid_plan: dict | None = None,
        benchmark_context: dict | None = None,
        task_route: TaskRoute | None = None,
    ) -> TaskSpec:
        route = task_route or self.route_goal(user_goal, benchmark_context=benchmark_context)
        payload = {
            "user_goal": user_goal,
            "page_snapshot": self._compact_page_snapshot_for_prompt(page_snapshot),
            "previous_plan": previous_plan.model_dump(mode="json") if previous_plan else None,
            "planning_profile": route.diagnostics(),
        }
        if validation_error:
            payload["repair_request"] = (
                f"Your previous plan was invalid. Validation error: {validation_error}. "
                "Return corrected JSON only."
            )
            payload["previous_invalid_plan"] = invalid_plan
        if benchmark_context:
            family = str(benchmark_context.get("task_family", "")).strip()
            payload["policy_hints"] = {
                "task_family": family,
                "output_fields": [
                    str(field).strip()
                    for field in (benchmark_context.get("required_top_level_fields") or [])
                    if str(field).strip() and str(field).strip() != "page_snapshot"
                ],
                "schema_contract_source": "context.output_fields",
            }
        system_prompt = build_profile_replanner_prompt(route.profile)
        if benchmark_context:
            system_prompt = build_benchmark_replanner_prompt(
                task_family=str(benchmark_context.get("task_family", "unknown")),
                allowed_actions=list(benchmark_context.get("allowed_actions", [])),
            )
        route.profile.profile_prompt_length = len(system_prompt)
        self.last_task_route = route
        self.last_profile_diagnostics = route.diagnostics()
        try:
            artifact = self.llm_client.generate_planner_artifact(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                stage="replanner",
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Replanner LLM failed; using local fallback plan: %s", exc)
            normalized = self.normalize_final_plan(
                raw_plan={},
                user_goal=user_goal,
                previous_plan=previous_plan,
                page_snapshot=page_snapshot,
                preferred_runtime_intents=route.profile.preferred_runtime_intents,
            )
            self.last_artifact = LLMArtifact(
                raw_response=json.dumps(normalized, ensure_ascii=False),
                parsed_response=normalized,
                generation=GenerationMetadata(
                    backend="local",
                    model="replanner_fallback",
                    source="fallback",
                    fallback_used=True,
                ),
            )
            raise_for_invalid_plan_actions(
                normalized,
                profile_diagnostics=self.last_profile_diagnostics,
                allowed_actions=route.profile.allowed_actions,
            )
            return TaskSpec.model_validate(normalized)
        self.last_artifact = artifact
        canonicalized, action_oov_detected = normalize_plan_action_aliases(artifact.parsed_response)
        self.last_action_oov_detected = action_oov_detected
        self.last_normalized_action_aliases = list(canonicalized.get("_normalized_action_aliases") or [])
        raise_for_invalid_plan_actions(
            canonicalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        normalized = self.normalize_final_plan(
            raw_plan=canonicalized,
            user_goal=user_goal,
            previous_plan=previous_plan,
            page_snapshot=page_snapshot,
            preferred_runtime_intents=route.profile.preferred_runtime_intents,
        )
        raise_for_invalid_plan_actions(
            normalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        return TaskSpec.model_validate(normalized)

    def build_corrective_plan(
        self,
        *,
        user_goal: str,
        page_snapshot: PageSnapshot,
        previous_plan: TaskSpec,
        execution_result: dict,
        verifier_verdict: dict,
        prior_corrective_attempts: list[dict] | None = None,
        failure_type: str | None = None,
        failed_action: str | None = None,
        failed_args: dict | None = None,
        failure_details: dict | None = None,
        error_message: str | None = None,
        verifier_issues: list[str] | None = None,
        previous_attempt_signatures: list[str] | None = None,
        disallowed_next_patterns: list[str] | None = None,
        benchmark_context: dict | None = None,
        task_route: TaskRoute | None = None,
    ) -> TaskSpec:
        route = task_route or self.route_goal(user_goal, benchmark_context=benchmark_context)
        payload = {
            "user_goal": user_goal,
            "page_snapshot": self._compact_page_snapshot_for_prompt(page_snapshot),
            "previous_plan": previous_plan.model_dump(mode="json"),
            "execution_result": execution_result,
            "verifier_verdict": verifier_verdict,
            "extracted_data": execution_result.get("extracted_data", {}),
            "failure_type": failure_type,
            "failed_action": failed_action,
            "failed_args": failed_args or execution_result.get("failed_args", {}),
            "failure_details": failure_details or execution_result.get("failure_details", {}),
            "error_message": error_message or execution_result.get("error_message"),
            "verifier_issues": verifier_issues or verifier_verdict.get("issues", []),
            "verifier_summary": verifier_verdict.get("summary"),
            "previous_attempt_signatures": previous_attempt_signatures or [],
            "prior_corrective_attempts": prior_corrective_attempts or [],
            "disallowed_next_patterns": disallowed_next_patterns or [],
            "planning_profile": route.diagnostics(),
        }
        if benchmark_context:
            family = str(benchmark_context.get("task_family", "")).strip()
            payload["policy_hints"] = {
                "task_family": family,
                "output_fields": [
                    str(field).strip()
                    for field in (benchmark_context.get("required_top_level_fields") or [])
                    if str(field).strip() and str(field).strip() != "page_snapshot"
                ],
                "schema_contract_source": "context.output_fields",
            }
        system_prompt = build_profile_replanner_prompt(route.profile, corrective=True)
        if benchmark_context:
            system_prompt = build_benchmark_replanner_prompt(
                task_family=str(benchmark_context.get("task_family", "unknown")),
                allowed_actions=list(benchmark_context.get("allowed_actions", [])),
            )
        route.profile.profile_prompt_length = len(system_prompt)
        self.last_task_route = route
        self.last_profile_diagnostics = route.diagnostics()
        try:
            artifact = self.llm_client.generate_planner_artifact(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                stage="corrective_replanner",
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Corrective replanner LLM failed; using local fallback plan: %s", exc)
            normalized = self.normalize_final_plan(
                raw_plan={},
                user_goal=user_goal,
                previous_plan=previous_plan,
                page_snapshot=page_snapshot,
            )
            self.last_artifact = LLMArtifact(
                raw_response=json.dumps(normalized, ensure_ascii=False),
                parsed_response=normalized,
                generation=GenerationMetadata(
                    backend="local",
                    model="corrective_replanner_fallback",
                    source="fallback",
                    fallback_used=True,
                ),
            )
            raise_for_invalid_plan_actions(
                normalized,
                profile_diagnostics=self.last_profile_diagnostics,
                allowed_actions=route.profile.allowed_actions,
            )
            return TaskSpec.model_validate(normalized)
        self.last_artifact = artifact
        canonicalized, action_oov_detected = normalize_plan_action_aliases(artifact.parsed_response)
        self.last_action_oov_detected = action_oov_detected
        self.last_normalized_action_aliases = list(canonicalized.get("_normalized_action_aliases") or [])
        raise_for_invalid_plan_actions(
            canonicalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        normalized = self.normalize_final_plan(
            raw_plan=canonicalized,
            user_goal=user_goal,
            previous_plan=previous_plan,
            page_snapshot=page_snapshot,
        )
        normalized = self._repair_unsupported_extract_value_action(normalized)
        normalized = self._repair_empty_section_corrective_plan(
            normalized_plan=normalized,
            page_snapshot=page_snapshot,
            failed_args=failed_args or execution_result.get("failed_args", {}),
            failure_details=failure_details or execution_result.get("failure_details", {}),
            error_message=error_message or execution_result.get("error_message"),
        )
        raise_for_invalid_plan_actions(
            normalized,
            profile_diagnostics=self.last_profile_diagnostics,
            allowed_actions=route.profile.allowed_actions,
        )
        return TaskSpec.model_validate(normalized)

    @staticmethod
    def _compact_page_snapshot_for_prompt(page_snapshot: PageSnapshot) -> dict:
        snapshot = page_snapshot.model_dump(mode="json")

        def truncate(value: object, limit: int) -> object:
            if isinstance(value, str) and len(value) > limit:
                return value[:limit] + "\n...[truncated]"
            return value

        def limit_list(value: object, limit: int) -> object:
            return value[:limit] if isinstance(value, list) else value

        def compact_link(item: object) -> object:
            if not isinstance(item, dict):
                return item
            return {
                key: truncate(item.get(key), 240 if key in {"text", "title"} else 500)
                for key in ("text", "title", "href", "link", "selector")
                if item.get(key) not in (None, "", [], {})
            }

        def compact_control(item: object) -> object:
            if not isinstance(item, dict):
                return item
            return {
                key: truncate(item.get(key), 240)
                for key in ("text", "label", "name", "placeholder", "type", "role", "selector")
                if item.get(key) not in (None, "", [], {})
            }

        def compact_row(item: object) -> object:
            if not isinstance(item, dict):
                return item
            compact = {
                key: truncate(item.get(key), 240 if key == "text" else 120)
                for key in ("row_id", "tag", "role", "text", "selector")
                if item.get(key) not in (None, "", [], {})
            }
            cells = item.get("cells")
            if isinstance(cells, list):
                compact["cells"] = [
                    {
                        cell_key: truncate(cell.get(cell_key), 80)
                        for cell_key in ("text", "value", "header", "column")
                        if isinstance(cell, dict) and cell.get(cell_key) not in (None, "", [], {})
                    }
                    for cell in cells[:8]
                    if isinstance(cell, dict)
                ]
            links = item.get("links")
            if isinstance(links, list):
                compact["links"] = [compact_link(link) for link in links[:4]]
            return compact

        def compact_table(item: object) -> object:
            if not isinstance(item, dict):
                return item
            compact = {
                key: truncate(item.get(key), 240)
                for key in ("caption", "summary", "selector")
                if item.get(key) not in (None, "", [], {})
            }
            headers = item.get("headers")
            if isinstance(headers, list):
                compact["headers"] = [truncate(header, 120) for header in headers[:20]]
            rows = item.get("rows")
            if isinstance(rows, list):
                compact["rows"] = [compact_row(row) for row in rows[:10]]
            return compact

        snapshot["page_text_excerpt"] = truncate(snapshot.get("page_text_excerpt"), 2000)
        snapshot["page_text"] = truncate(snapshot.get("page_text"), 6000)
        for key, limit in {
            "visible_headings": 80,
            "visible_labels": 120,
            "visible_buttons": 80,
            "visible_inputs": 80,
            "visible_links": 50,
            "text_lines": 100,
            "candidates": 50,
            "buttons": 80,
            "links": 50,
            "inputs": 50,
            "rows": 10,
            "tables": 5,
        }.items():
            snapshot[key] = limit_list(snapshot.get(key), limit)
        for key in ("visible_links", "links"):
            if isinstance(snapshot.get(key), list):
                snapshot[key] = [compact_link(item) for item in snapshot[key]]
        for key in ("visible_buttons", "visible_inputs"):
            if isinstance(snapshot.get(key), list):
                snapshot[key] = [truncate(item, 160) for item in snapshot[key]]
        for key in ("candidates", "buttons", "inputs"):
            if isinstance(snapshot.get(key), list):
                snapshot[key] = [compact_control(item) for item in snapshot[key]]
        if isinstance(snapshot.get("rows"), list):
            snapshot["rows"] = [compact_row(item) for item in snapshot["rows"]]
        if isinstance(snapshot.get("tables"), list):
            snapshot["tables"] = [compact_table(item) for item in snapshot["tables"]]
        headings = snapshot.get("headings")
        if isinstance(headings, list):
            compact_headings = []
            for heading in headings[:80]:
                if isinstance(heading, dict):
                    current = dict(heading)
                    current.pop("dom_path", None)
                    current["preview_after"] = limit_list(current.get("preview_after"), 5)
                    current["preview_after"] = [truncate(line, 160) for line in current.get("preview_after", [])]
                    compact_headings.append(current)
                else:
                    compact_headings.append(heading)
            snapshot["headings"] = compact_headings
        return snapshot

    @staticmethod
    def _repair_unsupported_extract_value_action(normalized_plan: dict) -> dict:
        steps = normalized_plan.get("steps")
        if not isinstance(steps, list):
            return normalized_plan
        repairs: list[dict] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict) or step.get("action") != "extract_value":
                continue
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            if any(key in args for key in ("pattern", "regex", "page_text", "source_text")):
                if "pattern" not in args and "regex" in args:
                    args["pattern"] = args.pop("regex")
                step["action"] = "extract_pattern_from_page_text"
                step["args"] = args
                repairs.append({"step_index": idx, "from": "extract_value", "to": "extract_pattern_from_page_text"})
            elif any(key in args for key in ("anchor", "anchor_text", "anchor_candidates")):
                if "anchor_text" not in args and "anchor" in args:
                    args["anchor_text"] = args.pop("anchor")
                step["action"] = "extract_value_near_anchor"
                step["args"] = args
                repairs.append({"step_index": idx, "from": "extract_value", "to": "extract_value_near_anchor"})
            else:
                step["args"] = {**args, "_repair_error": "unsupported action extract_value could not be mapped safely"}
        if repairs:
            normalized_plan.setdefault("_repair_diagnostics", {})["unsupported_action_repairs"] = repairs
        return normalized_plan

    @staticmethod
    def _repair_empty_section_corrective_plan(
        *,
        normalized_plan: dict,
        page_snapshot: PageSnapshot,
        failed_args: dict,
        failure_details: dict,
        error_message: str | None,
    ) -> dict:
        reason = str((failure_details or {}).get("reason", "")).strip().lower()
        error = str(error_message or "").lower()
        if reason != "empty_section" and "extracted zero lines" not in error:
            return normalized_plan
        failed_heading = str((failure_details or {}).get("failed_heading") or (failed_args or {}).get("heading_text") or "").strip().lower()
        details_available = (failure_details or {}).get("available_non_empty_headings", [])
        details_suggested = (failure_details or {}).get("suggested_next_headings", [])
        candidates: list[str] = []
        for pool in (details_suggested, details_available):
            if not isinstance(pool, list):
                continue
            for item in pool:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                line_count = int(item.get("line_count_after", 0) or 0)
                if not text or line_count <= 0 or text.lower() == failed_heading or text in candidates:
                    continue
                candidates.append(text)
        for item in page_snapshot.headings:
            text = str(getattr(item, "text", "") or "").strip()
            region = str(getattr(item, "region", "unknown") or "unknown").lower()
            if (
                not text
                or int(getattr(item, "line_count_after", 0) or 0) <= 0
                or text.lower() == failed_heading
                or text in candidates
                or region in {"nav", "header", "footer", "aside"}
            ):
                continue
            candidates.append(text)
        if not candidates:
            return normalized_plan

        steps = normalized_plan.get("steps")
        if not isinstance(steps, list):
            return normalized_plan
        extract_indices = [idx for idx, step in enumerate(steps) if isinstance(step, dict) and step.get("action") == "extract_section_lines"]
        compare_indices = [idx for idx, step in enumerate(steps) if isinstance(step, dict) and step.get("action") == "compare_structured_values"]
        if len(candidates) < 2 and extract_indices and compare_indices:
            first_extract_idx = extract_indices[0]
            steps[first_extract_idx] = {
                "step_id": steps[first_extract_idx].get("step_id", first_extract_idx + 1),
                "action": "extract_text",
                "args": {"selector": "main"},
                "save_as": "source_a",
            }
            second_idx = extract_indices[1] if len(extract_indices) > 1 else first_extract_idx + 1
            if second_idx >= len(steps):
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "action": "extract_text",
                        "args": {"selector": "article"},
                        "save_as": "source_b",
                    }
                )
            else:
                steps[second_idx] = {
                    "step_id": steps[second_idx].get("step_id", second_idx + 1),
                    "action": "extract_text",
                    "args": {"selector": "article"},
                    "save_as": "source_b",
                }
            return normalized_plan
        if not extract_indices:
            return normalized_plan

        for order, idx in enumerate(extract_indices):
            step = steps[idx]
            args = step.get("args")
            if not isinstance(args, dict):
                args = {}
                step["args"] = args
            replacement = candidates[min(order, len(candidates) - 1)]
            args["heading_text"] = replacement
        return normalized_plan

    @staticmethod
    def normalize_final_plan(
        raw_plan: dict,
        user_goal: str,
        previous_plan: TaskSpec | None,
        page_snapshot: PageSnapshot,
        preferred_runtime_intents: list[str] | None = None,
    ) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
        preferred_intents = {str(intent).strip().casefold() for intent in preferred_runtime_intents or [] if str(intent).strip()}
        context_start_url = ""
        for candidate in (
            plan.get("start_url"),
            str(previous_plan.start_url) if previous_plan else "",
            page_snapshot.url,
        ):
            context_start_url = _normalize_url_candidate(candidate)
            if context_start_url:
                break
        if not context_start_url:
            raise PlannerValidationFailed(
                {
                    "failure_stage": "planner_validation_failed",
                    "reason": "missing_start_url",
                    "message": "Replanner has no start_url from model, previous plan, or page snapshot.",
                }
            )

        steps = plan.get("steps")
        if not isinstance(steps, list):
            steps = []

        raw_expected_result = plan.get("expected_result") if isinstance(plan.get("expected_result"), dict) else {}
        required_fields = [
            str(field).strip()
            for field in (raw_expected_result.get("required_fields") if isinstance(raw_expected_result, dict) else [] or [])
            if str(field).strip()
        ]
        if not required_fields and previous_plan is not None:
            required_fields = list(previous_plan.expected_result.required_fields)

        normalized_steps: list[dict] = []
        logger = logging.getLogger(__name__)
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            current = dict(step)
            args = current.get("args")
            if not isinstance(args, dict) and isinstance(current.get("params"), dict):
                args = current.pop("params")
            if not isinstance(args, dict) and isinstance(current.get("action_params"), dict):
                args = current.pop("action_params")
            current["args"] = dict(args) if isinstance(args, dict) else {}
            action = str(current.get("action", "")).strip()
            reserved_step_keys = {"step_id", "action", "args", "params", "save_as", "expected_result"}
            for key in list(current.keys()):
                if key in reserved_step_keys:
                    continue
                if key not in current["args"]:
                    current["args"][key] = current.pop(key)
            if isinstance(current.get("save_as"), str) and current["save_as"].strip():
                current["save_as"] = normalize_required_field_alias(current["save_as"].strip())

            if action == "open_url" and "url" not in current["args"] and current.get("url") is not None:
                current["args"]["url"] = current.pop("url")
            if action == "open_url":
                raw_url_candidate = current["args"].get("url")
                normalized_url = _normalize_url_candidate(current["args"].get("url"))
                if Replanner._should_rewrite_result_open_url_to_click(
                    goal=user_goal,
                    raw_url=raw_url_candidate,
                    normalized_url=normalized_url,
                    context_start_url=context_start_url,
                    prior_steps=normalized_steps,
                ):
                    current["action"] = "click_by_semantic_target"
                    current["args"] = {
                        "target_text": Replanner._navigation_target_for_empty_fallback(goal=user_goal) or "first result",
                        "role": "link",
                    }
                    current["save_as"] = "clicked_text"
                elif normalized_url:
                    current["args"]["url"] = normalized_url
                else:
                    logger.warning("Malformed open_url from model. Applying start_url normalization.")
                    current["args"]["url"] = context_start_url
            if action == "click_by_semantic_target":
                if "target_text" not in current["args"] and current["args"].get("text") is not None:
                    current["args"]["target_text"] = current["args"].pop("text")
                if "target_text" not in current["args"] and current["args"].get("target") is not None:
                    current["args"]["target_text"] = current["args"].pop("target")
                target_text = str(current["args"].get("target_text", "") or "")
                option_match = re.search(r"\b([A-Za-z][\w-]{1,30})\s+(?:or|или)\s+([A-Za-z][\w-]{1,30})\b", target_text, flags=re.IGNORECASE)
                if option_match and "target_candidates" not in current["args"]:
                    current["args"]["target_candidates"] = [option_match.group(1), option_match.group(2)]
                if "|" in target_text and "target_candidates" not in current["args"]:
                    candidates = [part.strip() for part in target_text.split("|") if part.strip()]
                    if candidates:
                        current["args"]["target_candidates"] = candidates
                if not str(current.get("save_as", "") or "").strip():
                    goal_hint = str(user_goal or "").casefold()
                    if any(token in goal_hint for token in ("clicked", "click", "press", "open link", "нажм", "клик", "перейд")):
                        current["save_as"] = "clicked_text"
                if row_action := Replanner._row_action_for_empty_fallback(goal=user_goal):
                    target_text = str(current["args"].get("target_text", "") or "").strip()
                    condition = dict(row_action["condition"])
                    if target_text and target_text.casefold() not in {"delete", "remove", "trash", "open", "click", "select", "choose"}:
                        condition = {"contains": target_text}
                    current["action"] = "click_row_action"
                    current["args"] = {"action_name": row_action["action_name"], "condition": condition}
                    current["save_as"] = "row_action"
            if action == "fill_by_semantic_target" and "value" not in current["args"] and current["args"].get("query") is not None:
                current["args"]["value"] = current["args"].pop("query")
            if action == "click_row_action":
                action_name = str(current["args"].get("action_name", "") or current["args"].get("action", "") or "").strip().casefold()
                action_aliases = {
                    "remove": "delete",
                    "trash": "trash",
                    "delete": "delete",
                    "close": "delete",
                    "star": "star",
                    "favorite": "star",
                    "favourite": "star",
                    "reply": "reply",
                    "open": "open",
                    "click": "open",
                    "select": "select",
                    "choose": "select",
                }
                normalized_action_name = action_aliases.get(action_name) or Replanner._row_action_name_for_goal(user_goal)
                if normalized_action_name:
                    current["args"]["action_name"] = normalized_action_name
                if not current["args"].get("condition"):
                    condition = Replanner._row_condition_for_goal(user_goal)
                    if condition:
                        current["args"]["condition"] = condition
                if not str(current.get("save_as", "") or "").strip():
                    current["save_as"] = "row_action"
            if (
                action == "extract_structured_items"
                and goal_requests_semantic_region_fields(user_goal, required_fields)
                and ("fields" not in current["args"] or not current["args"].get("fields"))
            ):
                output_key = str(current["args"].get("output_key") or current.get("save_as") or "contact_info").strip()
                current["action"] = "extract_by_intent"
                current["args"] = build_semantic_region_fields_args(user_goal, required_fields, output_key=output_key)
                current["save_as"] = output_key
                action = "extract_by_intent"
            if action == "extract_structured_items":
                semantic_intent = semantic_intent_for_structured_step(current)
                if semantic_intent:
                    output_key = (
                        str(current["args"].get("output_key", "") or "").strip()
                        or (current.get("save_as") if isinstance(current.get("save_as"), str) else "")
                        or default_output_key_for_intent(semantic_intent)
                    )
                    limit = current["args"].get("limit")
                    current["action"] = "extract_by_intent"
                    current["args"] = {
                        "intent": semantic_intent,
                        "output_key": output_key,
                        "limit": limit if isinstance(limit, int) and limit > 0 else 20,
                    }
                    if semantic_intent in {"article_results", "news_items", "paper_results", "repository_results"}:
                        current["args"]["item_type"] = semantic_intent.replace("_results", "").replace("news_items", "news")
                    if output_key and not str(current.get("save_as", "") or "").strip():
                        current["save_as"] = output_key
                elif not isinstance(current["args"].get("limit"), int) or current["args"].get("limit") <= 0:
                    current["args"]["limit"] = 20
            if action == "extract_value_near_anchor":
                anchor_hint = " ".join(
                    str(current["args"].get(key, "") or "")
                    for key in ("anchor_text", "anchor", "target", "value_type")
                ).casefold()
                save_as = str(current.get("save_as", "") or "").strip()
                if save_as == "final_url" or "current url" in anchor_hint or "final url" in anchor_hint:
                    current["action"] = "extract_by_intent"
                    current["args"] = {"intent": "current_url"}
                elif save_as == "page_title" or anchor_hint in {"title", "page title"}:
                    current["action"] = "extract_by_intent"
                    current["args"] = {"intent": "page_title"}
            if action == "extract_text":
                save_as = str(current.get("save_as", "") or "").strip()
                selector_hint = str(current["args"].get("selector", "") or "").strip().casefold()
                target_hint = str(current["args"].get("target", "") or "").strip().casefold()
                if save_as == "page_title" or selector_hint == "title" or target_hint in {"title", "page title"}:
                    current["action"] = "extract_by_intent"
                    current["args"] = {"intent": "page_title"}
            action = str(current.get("action", action)).strip()
            if action == "wait_for":
                if Replanner._should_drop_brittle_result_wait(
                    goal=user_goal,
                    args=current["args"],
                    prior_steps=normalized_steps,
                ) or Replanner._should_drop_brittle_row_wait(goal=user_goal, args=current["args"]):
                    continue
            if action == "extract_by_intent" and str(current["args"].get("intent", "") or "").strip():
                current["args"]["intent"] = normalize_intent_alias(current["args"].get("intent"))
                intent = str(current["args"].get("intent", "") or "").strip().casefold()
                save_as_hint = str(current.get("save_as", "") or "").strip().casefold()
                if (
                    save_as_hint in {"description", "summary", "snippet"}
                    and intent
                    in {
                        "search_results",
                        "results",
                        "result_list",
                        "paper_results",
                        "repository_results",
                        "article_results",
                        "news_items",
                    }
                    and Replanner._navigation_target_for_empty_fallback(goal=user_goal)
                ):
                    current["args"] = {"intent": "package_metadata", "output_key": save_as_hint}
                    intent = "package_metadata"
                if (
                    intent
                    in {
                        "search_results",
                        "paper_results",
                        "repository_results",
                        "article_results",
                        "news_items",
                        "card_items",
                        "product_cards",
                        "table_rows",
                    }
                    and not any(key in current["args"] for key in ("condition", "filter", "where"))
                ):
                    condition = Replanner._condition_for_goal(user_goal)
                    if condition:
                        current["args"]["condition"] = condition
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
                    "search_results",
                    "results",
                    "result_list",
                    "paper_results",
                    "papers",
                    "repository_results",
                    "repositories",
                    "repo_results",
                    "article_results",
                    "articles",
                    "news_items",
                    "news",
                    "product_cards",
                    "products",
                    "card_items",
                    "cards",
                    "table_rows",
                    "rows",
                    "package_metadata",
                    "package_info",
                    "library_metadata",
                }
            if (
                collection_like_action
                and not str(current.get("save_as", "") or "").strip()
                and not str(current["args"].get("output_key", "") or "").strip()
            ):
                intent = str(current["args"].get("intent", "") or "").strip().casefold()
                output_key = default_output_key_for_intent(intent) if intent else "items"
                if action == "extract_visible_links":
                    output_key = "links"
                current["args"]["output_key"] = output_key
                current["save_as"] = output_key
            current["step_id"] = idx
            normalized_steps.append(current)

        fallback_required_fields: list[str] = []
        fallback_description = ""
        fallback_force_required_fields = False
        if not normalized_steps:
            if visual_count := Replanner._visual_count_for_empty_fallback(goal=user_goal):
                output_key = visual_count["output_key"]
                fallback_required_fields = [output_key]
                fallback_description = "Count requested visible objects"
                fallback_force_required_fields = True
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {"step_id": 2, "action": "visual_observe", "args": {}, "save_as": "page_snapshot"},
                    {
                        "step_id": 3,
                        "action": "visual_extract_object_count",
                        "args": {"target": visual_count["target"], **visual_count.get("region_args", {})},
                        "save_as": output_key,
                    },
                    {"step_id": 4, "action": "finish", "args": {}},
                ]
            elif row_action := Replanner._row_action_for_empty_fallback(goal=user_goal):
                fallback_required_fields = ["row_action"]
                fallback_description = "Find matching row and perform requested row action"
                fallback_force_required_fields = True
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": "find_row_by_condition",
                        "args": {"condition": row_action["condition"]},
                        "save_as": "row_ref",
                    },
                    {
                        "step_id": 3,
                        "action": "click_row_action",
                        "args": {"action_name": row_action["action_name"], "condition": row_action["condition"]},
                        "save_as": "row_action",
                    },
                    {"step_id": 4, "action": "finish", "args": {}},
                ]
            elif navigation_target := Replanner._navigation_target_for_empty_fallback(goal=user_goal):
                fallback_required_fields = Replanner._navigation_required_fields_for_goal(user_goal)
                fallback_description = "Navigate and extract requested page metadata"
                if "description" in fallback_required_fields and preferred_intents and "package_metadata" not in preferred_intents:
                    fallback_required_fields = [field for field in fallback_required_fields if field != "description"]
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": "click_by_semantic_target",
                        "args": {"target_text": navigation_target, "role": "link"},
                        "save_as": "clicked_text",
                    },
                    {"step_id": 3, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                ]
                next_step_id = 4
                if "page_title" in fallback_required_fields:
                    normalized_steps.append(
                        {
                            "step_id": next_step_id,
                            "action": "extract_by_intent",
                            "args": {"intent": "page_title"},
                            "save_as": "page_title",
                        }
                    )
                    next_step_id += 1
                if "final_url" in fallback_required_fields:
                    normalized_steps.append(
                        {
                            "step_id": next_step_id,
                            "action": "extract_by_intent",
                            "args": {"intent": "current_url"},
                            "save_as": "final_url",
                        }
                    )
                    next_step_id += 1
                if "description" in fallback_required_fields:
                    normalized_steps.append(
                        {
                            "step_id": next_step_id,
                            "action": "extract_by_intent",
                            "args": {"intent": "package_metadata", "output_key": "page_metadata"},
                            "save_as": "page_metadata",
                        }
                    )
                    next_step_id += 1
                if "visible_links" in fallback_required_fields:
                    normalized_steps.append(
                        {
                            "step_id": next_step_id,
                            "action": "extract_visible_links",
                            "args": {"output_key": "visible_links", "limit": 12},
                            "save_as": "visible_links",
                        }
                    )
                    next_step_id += 1
                normalized_steps.append({"step_id": next_step_id, "action": "finish", "args": {}})
            elif Replanner._looks_like_package_metadata_request(
                goal=user_goal,
                previous_plan=previous_plan,
                plan=plan,
                page_snapshot=page_snapshot,
            ) and (not preferred_intents or "package_metadata" in preferred_intents):
                fallback_required_fields = ["package_metadata"]
                fallback_description = "Extract package metadata"
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": "extract_by_intent",
                        "args": {"intent": "package_metadata", "output_key": "package_metadata"},
                        "save_as": "package_metadata",
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ]
            elif (
                table_intent := Replanner._table_intent_for_empty_fallback(
                    goal=user_goal,
                    previous_plan=previous_plan,
                    plan=plan,
                    page_snapshot=page_snapshot,
                )
            ) and (not preferred_intents or table_intent in preferred_intents):
                output_key = default_output_key_for_intent(table_intent)
                condition = Replanner._condition_for_goal(user_goal)
                fallback_required_fields = [output_key]
                fallback_description = "Extract table rows"
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": "extract_by_intent",
                        "args": {
                            "intent": table_intent,
                            "output_key": output_key,
                            **({"condition": condition} if condition else {}),
                        },
                        "save_as": output_key,
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ]
            elif (
                collection_intent := Replanner._collection_intent_for_empty_fallback(
                    goal=user_goal,
                    previous_plan=previous_plan,
                    plan=plan,
                    page_snapshot=page_snapshot,
                )
            ) and (not preferred_intents or collection_intent in preferred_intents):
                output_key = default_output_key_for_intent(collection_intent)
                condition = Replanner._condition_for_goal(user_goal)
                fallback_required_fields = [output_key]
                fallback_description = "Extract requested collection"
                action = "extract_visible_links" if collection_intent == "visible_links" else "extract_by_intent"
                args = (
                    {"output_key": output_key}
                    if action == "extract_visible_links"
                    else {"intent": collection_intent, "output_key": output_key}
                )
                if condition and action == "extract_by_intent":
                    args["condition"] = condition
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": action,
                        "args": args,
                        "save_as": output_key,
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ]
            else:
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {"step_id": 2, "action": "finish", "args": {}},
                ]
        expected_result = plan.get("expected_result")
        if not isinstance(expected_result, dict):
            expected_result = {}

        if not isinstance(expected_result.get("description"), str) or not expected_result["description"].strip():
            if previous_plan and previous_plan.expected_result.description:
                expected_result["description"] = previous_plan.expected_result.description
            else:
                expected_result["description"] = f"Complete goal: {user_goal}"

        if not isinstance(expected_result.get("required_fields"), list):
            expected_result["required_fields"] = (
                list(previous_plan.expected_result.required_fields) if previous_plan else []
            )
        if fallback_required_fields:
            required_hint = {
                str(field).strip()
                for field in expected_result.get("required_fields", [])
                if str(field).strip()
            }
            if fallback_force_required_fields or not required_hint or required_hint.issubset({"page_snapshot", "final_url", "page_title"}):
                expected_result["required_fields"] = fallback_required_fields
                expected_result["description"] = fallback_description or expected_result["description"]
        expected_result["required_fields"] = Replanner._normalize_required_fields_against_steps(
            required_fields=expected_result.get("required_fields", []),
            steps=normalized_steps,
        )
        normalized_steps = coalesce_package_metadata_steps(
            normalized_steps,
            goal=user_goal,
            required_fields=expected_result["required_fields"],
        )
        produced_fields = {
            str(step.get("save_as", "") or "").strip()
            for step in normalized_steps
            if str(step.get("save_as", "") or "").strip()
        }
        metadata_fields = [
            field
            for field in expected_result["required_fields"]
            if field in {"final_url", "page_title"} and field not in produced_fields
        ]
        if metadata_fields:
            insert_index = next(
                (idx for idx, step in enumerate(normalized_steps) if step.get("action") == "finish"),
                len(normalized_steps),
            )
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
            for idx, step in enumerate(normalized_steps, start=1):
                step["step_id"] = idx
        normalized_steps = Replanner._move_url_extractors_after_navigating_metadata(normalized_steps)

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        if previous_plan:
            constraints = {
                "max_steps": constraints.get("max_steps", previous_plan.constraints.max_steps),
                "max_replans": constraints.get("max_replans", previous_plan.constraints.max_replans),
                "max_verification_retries": constraints.get(
                    "max_verification_retries",
                    previous_plan.constraints.max_verification_retries,
                ),
                "timeout_sec": constraints.get("timeout_sec", previous_plan.constraints.timeout_sec),
            }
        else:
            constraints = {
                "max_steps": constraints.get("max_steps", 10),
                "max_replans": constraints.get("max_replans", 1),
                "max_verification_retries": constraints.get("max_verification_retries", 3),
                "timeout_sec": constraints.get("timeout_sec", 30),
            }

        allowed_domains = plan.get("allowed_domains")
        domain = urlparse(context_start_url).netloc
        if (
            not isinstance(allowed_domains, list)
            or not allowed_domains
            or (domain and domain not in {str(item).strip() for item in allowed_domains})
        ):
            if previous_plan and previous_plan.allowed_domains and domain in previous_plan.allowed_domains:
                allowed_domains = list(previous_plan.allowed_domains)
            else:
                allowed_domains = [domain] if domain else []

        return {
            "goal": plan.get("goal") or (previous_plan.goal if previous_plan else user_goal),
            "start_url": context_start_url,
            "allowed_domains": allowed_domains,
            "constraints": constraints,
            "expected_result": expected_result,
            "steps": normalized_steps,
        }

    @staticmethod
    def _looks_like_package_metadata_request(
        *,
        goal: str,
        previous_plan: TaskSpec | None,
        plan: dict,
        page_snapshot: PageSnapshot,
    ) -> bool:
        hints = [str(goal or ""), str(plan.get("goal", "") or ""), page_snapshot.url, page_snapshot.title]
        required_fields: list[str] = []
        expected_result = plan.get("expected_result")
        if isinstance(expected_result, dict) and isinstance(expected_result.get("required_fields"), list):
            required_fields.extend(str(field).strip() for field in expected_result["required_fields"])
        if previous_plan:
            required_fields.extend(previous_plan.expected_result.required_fields)
            hints.extend([previous_plan.goal, str(previous_plan.start_url)])
        normalized_fields = {normalize_required_field_alias(field).casefold() for field in required_fields if field}
        if len(normalized_fields & PACKAGE_METADATA_FIELDS) >= 2:
            return True
        if len(normalized_fields & {"name", "page_title", "description", "summary", "final_url"}) >= 2:
            return True
        haystack = " ".join(hints).casefold()
        generic_metadata_groups = [
            ("name", "title", "page title"),
            ("description", "summary"),
            ("url", "current url", "final url", "link"),
        ]
        if sum(1 for group in generic_metadata_groups if any(token in haystack for token in group)) >= 2:
            return True
        package_context = any(token in haystack for token in ("package", "library", "module"))
        metadata_context = any(token in haystack for token in ("version", "description", "summary", "metadata"))
        return package_context and metadata_context

    @staticmethod
    def _navigation_target_for_empty_fallback(*, goal: str) -> str:
        text = str(goal or "").strip()
        if not text:
            return ""
        quoted = re.findall(r'["“”«»]([^"“”«»]{2,60})["“”«»]', text)
        if quoted:
            return quoted[0].strip()
        patterns = [
            r"\b(?:click|follow|press)\s+(?:the\s+)?(.{2,60}?)(?:\s+link|\s+button|,|\.|\bthen\b|$)",
            r"\b(?:open|navigate to|go to)\s+(?:the\s+)?(.{2,60}?)(?:\s+link|\s+page|,|\.|\bthen\b|$)",
            r"\b(?:перейди|нажми|кликни|открой)\s+(?:по\s+)?(?:ссылке\s+)?(.{2,60}?)(?:,|\.|\bзатем\b|$)",
        ]
        stop_words = {"the", "a", "an", "link", "button", "page", "website", "site", "ссылку", "ссылка", "страницу"}
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.")
                words = [word for word in candidate.split() if word.casefold() not in stop_words]
                candidate = " ".join(words).strip()
                if candidate and not candidate.startswith(("http://", "https://")):
                    return candidate
        return ""

    @staticmethod
    def _navigation_required_fields_for_goal(goal: str) -> list[str]:
        text = str(goal or "").casefold()
        fields: list[str] = []
        if any(token in text for token in ("title", "page title", "заголов")):
            fields.append("page_title")
        if any(token in text for token in ("url", "current url", "final url", "текущий url")):
            fields.append("final_url")
        if any(token in text for token in ("description", "summary", "snippet", "short description", "РѕРїРёСЃР°РЅ", "РєСЂР°С‚Рє")):
            fields.append("description")
        if any(token in text for token in ("link", "links", "visible links", "ссылк")):
            fields.append("visible_links")
        return fields or ["final_url", "page_title"]

    @staticmethod
    def _row_action_for_empty_fallback(*, goal: str) -> dict[str, Any]:
        text = str(goal or "").strip()
        folded = text.casefold()
        if not re.search(r"\b(?:row|rows|item|items|record|records|list item|table row)\b", folded):
            return {}
        action_name = Replanner._row_action_name_for_goal(goal)
        if not action_name:
            return {}
        condition = Replanner._row_condition_for_goal(goal)
        if not condition:
            return {}
        return {"action_name": action_name, "condition": condition}

    @staticmethod
    def _row_action_name_for_goal(goal: str) -> str:
        text = str(goal or "").casefold()
        checks = [
            ("delete", ("delete", "remove", "trash")),
            ("star", ("star", "favorite", "favourite", "mark important")),
            ("select", ("select", "choose")),
            ("open", ("open", "click")),
        ]
        for action_name, tokens in checks:
            if any(token in text for token in tokens):
                return action_name
        return ""

    @staticmethod
    def _row_condition_for_goal(goal: str) -> dict[str, str]:
        text = str(goal or "").strip()
        quoted = re.findall(r'["вЂњвЂќВ«В»]([^"вЂњвЂќВ«В»]{1,120})["вЂњвЂќВ«В»]', text)
        if quoted:
            return {"contains": quoted[0].strip()}
        patterns = [
            r"\b(?:row|item|record|list item|table row)\s+(?:named|called|labeled|labelled|with text|containing|contains)\s+(.{1,120}?)(?:,|\.|\bthen\b|$)",
            r"\b(?:delete|remove|trash|click|open|select|choose|star)\s+(?:the\s+)?(?:table\s+)?(?:row|item|record|list item)\s+(?:named|called|labeled|labelled|with text|containing|contains)\s+(.{1,120}?)(?:,|\.|\bthen\b|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.\"'")
            if value:
                return {"contains": value}
        return Replanner._condition_for_goal(goal)

    @staticmethod
    def _visual_count_for_empty_fallback(*, goal: str) -> dict[str, Any]:
        text = str(goal or "").casefold()
        if not any(token in text for token in ("count", "how many", "visual", "visible", "screenshot")):
            return {}
        target_checks = [
            ("link", ("link", "links", "anchor", "anchors")),
            ("button", ("button", "buttons")),
            ("input", ("input", "field", "fields")),
            ("item", ("item", "items", "card", "cards", "row", "rows")),
        ]
        target = ""
        for candidate, tokens in target_checks:
            if any(token in text for token in tokens):
                target = candidate
                break
        if not target:
            return {}
        output_key = "count"
        if "link" in target:
            output_key = "link_count"
        elif "button" in target:
            output_key = "button_count"
        elif "input" in target:
            output_key = "input_count"
        elif "item" in target:
            output_key = "item_count"
        if any(token in text for token in ("language", "languages")) and target == "link":
            output_key = "language_link_count"
        region_args: dict[str, Any] = {}
        if any(token in text for token in ("center", "centre", "middle")):
            region_args["region"] = {"x": 0.25, "y": 0.15, "width": 0.5, "height": 0.55}
        return {"target": target, "output_key": output_key, "region_args": region_args}

    @staticmethod
    def _should_rewrite_result_open_url_to_click(
        *,
        goal: str,
        raw_url: Any,
        normalized_url: str,
        context_start_url: str,
        prior_steps: list[dict],
    ) -> bool:
        target = Replanner._navigation_target_for_empty_fallback(goal=goal)
        target_text = target.casefold()
        goal_text = str(goal or "").casefold()
        if "result" not in target_text and "result" not in goal_text:
            return False
        prior_result_extraction = False
        for step in prior_steps:
            if not isinstance(step, dict):
                continue
            save_as = str(step.get("save_as", "") or "").casefold()
            action = str(step.get("action", "") or "").casefold()
            intent = str((step.get("args") or {}).get("intent", "") or "").casefold() if isinstance(step.get("args"), dict) else ""
            if action == "extract_by_intent" and intent in {"search_results", "results", "result_list", "repository_results", "paper_results", "article_results", "news_items"}:
                prior_result_extraction = True
                break
            if "result" in save_as:
                prior_result_extraction = True
                break
        if not prior_result_extraction:
            return False
        raw_text = str(raw_url or "")
        dynamic_markers = ("result", "href", "link", "{{", "}}", "[0]", "$", ".")
        if any(marker in raw_text.casefold() for marker in dynamic_markers) and not normalized_url:
            return True
        if normalized_url and context_start_url and normalized_url.rstrip("/") == context_start_url.rstrip("/"):
            return True
        return False

    @staticmethod
    def _move_url_extractors_after_navigating_metadata(steps: list[dict]) -> list[dict]:
        package_indices = [
            idx
            for idx, step in enumerate(steps)
            if isinstance(step, dict)
            and step.get("action") == "extract_by_intent"
            and isinstance(step.get("args"), dict)
            and str(step["args"].get("intent", "") or "").strip().casefold() == "package_metadata"
        ]
        if not package_indices:
            return steps
        last_package_idx = max(package_indices)
        move_indices = [
            idx
            for idx, step in enumerate(steps)
            if idx < last_package_idx
            and isinstance(step, dict)
            and step.get("action") == "extract_by_intent"
            and isinstance(step.get("args"), dict)
            and str(step["args"].get("intent", "") or "").strip().casefold() == "current_url"
            and str(step.get("save_as", "") or "").strip() == "final_url"
        ]
        if not move_indices:
            return steps
        moving = [steps[idx] for idx in move_indices]
        remaining = [step for idx, step in enumerate(steps) if idx not in set(move_indices)]
        package_step = steps[last_package_idx]
        insert_index = next((idx for idx, step in enumerate(remaining) if step is package_step), len(remaining) - 1) + 1
        remaining[insert_index:insert_index] = moving
        for idx, step in enumerate(remaining, start=1):
            if isinstance(step, dict):
                step["step_id"] = idx
        return remaining

    @staticmethod
    def _should_drop_brittle_result_wait(*, goal: str, args: dict, prior_steps: list[dict]) -> bool:
        url_contains = str(args.get("url_contains", "") or "").strip()
        if not url_contains:
            return False
        goal_text = str(goal or "").casefold()
        if url_contains.casefold() in goal_text:
            return False
        target = Replanner._navigation_target_for_empty_fallback(goal=goal).casefold()
        if "result" not in target and "result" not in goal_text:
            return False
        for step in reversed(prior_steps[-3:]):
            if not isinstance(step, dict) or step.get("action") != "click_by_semantic_target":
                continue
            args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
            target_text = str(args_obj.get("target_text") or args_obj.get("target") or args_obj.get("text") or "").casefold()
            if "result" in target_text:
                return True
        return False

    @staticmethod
    def _should_drop_brittle_row_wait(*, goal: str, args: dict) -> bool:
        if not Replanner._row_action_for_empty_fallback(goal=goal):
            return False
        if str(args.get("selector", "") or "").strip() or str(args.get("url_contains", "") or "").strip():
            return False
        return bool(str(args.get("text", "") or "").strip())

    @staticmethod
    def _condition_for_goal(goal: str) -> dict[str, Any]:
        text = str(goal or "").strip()
        if not text:
            return {}
        patterns = [
            ("title", r"\b(?:whose|where|with)\s+title\s+(?:contains?|includes?|has)\s+(.{1,80}?)(?:,|\.|\bwith\b|\bthen\b|$)"),
            ("title", r"\btitle\s+(?:contains?|includes?|has)\s+(.{1,80}?)(?:,|\.|\bwith\b|\bthen\b|$)"),
            ("title", r"\b(?:в\s+заголовке|заголовок)\s+(?:есть|содержит)\s+(.{1,80}?)(?:,|\.|\bс\b|\bзатем\b|$)"),
            ("contains", r"\b(?:contains?|includes?|matching)\s+(.{1,80}?)(?:,|\.|\bwith\b|\bthen\b|$)"),
            ("contains", r"\b(?:содержит|содержащие)\s+(.{1,80}?)(?:,|\.|\bс\b|\bзатем\b|$)"),
        ]
        for key, pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.")
            if not raw:
                continue
            terms = [
                item.strip(" '\"“”«»")
                for item in re.split(r"\s+(?:or|или)\s+|\|", raw, flags=re.IGNORECASE)
                if item.strip(" '\"“”«»")
            ]
            if not terms:
                continue
            return {key: terms if len(terms) > 1 else terms[0]}
        return {}

    @staticmethod
    def _table_intent_for_empty_fallback(
        *,
        goal: str,
        previous_plan: TaskSpec | None,
        plan: dict,
        page_snapshot: PageSnapshot,
    ) -> str:
        hints = [
            str(goal or ""),
            str(plan.get("goal", "") or ""),
            page_snapshot.title,
            page_snapshot.page_text_excerpt or "",
            page_snapshot.page_text or "",
        ]
        required_fields: list[str] = []
        expected_result = plan.get("expected_result")
        if isinstance(expected_result, dict) and isinstance(expected_result.get("required_fields"), list):
            required_fields.extend(str(field).strip() for field in expected_result["required_fields"])
        if previous_plan:
            hints.append(previous_plan.goal)
            required_fields.extend(previous_plan.expected_result.required_fields)
        normalized_required = " ".join(normalize_required_field_alias(field) for field in required_fields).casefold()
        haystack = " ".join(hints).casefold()
        table_markers = ("table", "row", "rows", "таблиц", "строк")
        if any(marker in normalized_required for marker in ("row", "rows", "table")):
            return "table_rows"
        if any(marker in haystack for marker in table_markers):
            return "table_rows"
        return ""

    @staticmethod
    def _collection_intent_for_empty_fallback(
        *,
        goal: str,
        previous_plan: TaskSpec | None,
        plan: dict,
        page_snapshot: PageSnapshot,
    ) -> str:
        goal_hints = [
            str(goal or ""),
            str(plan.get("goal", "") or ""),
        ]
        page_hints = [
            page_snapshot.title,
            page_snapshot.page_text_excerpt or "",
        ]
        required_fields: list[str] = []
        expected_result = plan.get("expected_result")
        if isinstance(expected_result, dict) and isinstance(expected_result.get("required_fields"), list):
            required_fields.extend(str(field).strip() for field in expected_result["required_fields"])
        if previous_plan:
            goal_hints.append(previous_plan.goal)
            required_fields.extend(previous_plan.expected_result.required_fields)
        normalized_required = " ".join(normalize_required_field_alias(field) for field in required_fields).casefold()
        goal_haystack = " ".join(goal_hints).casefold()
        page_haystack = " ".join(page_hints).casefold()
        checks = [
            ("product_cards", ("product", "products", "product_card", "product_cards", "товар")),
            ("card_items", ("card", "cards", "card_items", "catalog", "listing", "listings", "карточ", "каталог")),
            ("repository_results", ("repository", "repositories", "repo", "repos", "репозитор")),
            ("paper_results", ("paper", "papers", "preprint", "publication", "научн", "стат")),
            ("article_results", ("article", "articles", "news", "post", "posts", "новост", "публик")),
            ("search_results", ("search result", "search_results", "results", "result list", "результат")),
            ("visible_links", ("visible link", "visible_links", "links", "link list", "ссылк")),
        ]
        for intent, markers in checks:
            if any(marker in normalized_required for marker in markers):
                return intent
        for intent, markers in checks:
            if any(marker in goal_haystack for marker in markers):
                return intent
        for intent, markers in checks:
            if any(marker in page_haystack for marker in markers):
                return intent
        return ""

    @staticmethod
    def _normalize_required_fields_against_steps(required_fields: list, steps: list[dict]) -> list[str]:
        top_level_fields = {
            step.get("save_as")
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str) and step.get("save_as").strip()
        }
        business_top_level_fields = sorted(
            str(field).strip()
            for field in top_level_fields
            if str(field).strip() and str(field).strip() not in {"page_snapshot", "clicked_text", "final_url", "page_title"}
        )
        single_business_parent = business_top_level_fields[0] if len(business_top_level_fields) == 1 else ""
        structured_field_to_parent: dict[str, str] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("action") not in {"extract_items", "extract_structured_items"}:
                continue
            save_as = step.get("save_as")
            if not isinstance(save_as, str) or not save_as.strip():
                continue
            fields = step.get("args", {}).get("fields")
            if not isinstance(fields, dict):
                continue
            for field_name in fields.keys():
                structured_field_to_parent[str(field_name)] = save_as

        normalized_required_fields: list[str] = []
        for field in required_fields:
            name = normalize_required_field_alias(str(field).strip())
            if not name:
                continue
            mapped = structured_field_to_parent.get(name, name)
            if mapped == name and single_business_parent and name not in top_level_fields:
                mapped = single_business_parent
            if mapped in top_level_fields and mapped not in normalized_required_fields:
                normalized_required_fields.append(mapped)
                continue
            if mapped == name and mapped not in normalized_required_fields:
                normalized_required_fields.append(mapped)
        return normalized_required_fields
