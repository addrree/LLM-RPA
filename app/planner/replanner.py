import json
import logging
import re
from urllib.parse import urlparse

from app.planner.action_vocab import (
    PACKAGE_METADATA_FIELDS,
    coalesce_package_metadata_steps,
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
    ) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
        context_start_url = (
            str(plan.get("start_url") or "")
            or (str(previous_plan.start_url) if previous_plan else "")
            or page_snapshot.url
        )
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
            if action == "open_url" and not str(current["args"].get("url", "")).strip():
                logger.warning("Malformed open_url from model: missing args.url. Applying start_url normalization.")
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
            if action == "fill_by_semantic_target" and "value" not in current["args"] and current["args"].get("query") is not None:
                current["args"]["value"] = current["args"].pop("query")
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
            if action == "extract_by_intent" and str(current["args"].get("intent", "") or "").strip():
                current["args"]["intent"] = normalize_intent_alias(current["args"].get("intent"))
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
        if not normalized_steps:
            if Replanner._looks_like_package_metadata_request(
                goal=user_goal,
                previous_plan=previous_plan,
                plan=plan,
                page_snapshot=page_snapshot,
            ):
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
            elif table_intent := Replanner._table_intent_for_empty_fallback(
                goal=user_goal,
                previous_plan=previous_plan,
                plan=plan,
                page_snapshot=page_snapshot,
            ):
                output_key = default_output_key_for_intent(table_intent)
                fallback_required_fields = [output_key]
                fallback_description = "Extract table rows"
                normalized_steps = [
                    {"step_id": 1, "action": "open_url", "args": {"url": context_start_url}},
                    {
                        "step_id": 2,
                        "action": "extract_by_intent",
                        "args": {"intent": table_intent, "output_key": output_key},
                        "save_as": output_key,
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ]
            elif collection_intent := Replanner._collection_intent_for_empty_fallback(
                goal=user_goal,
                previous_plan=previous_plan,
                plan=plan,
                page_snapshot=page_snapshot,
            ):
                output_key = default_output_key_for_intent(collection_intent)
                fallback_required_fields = [output_key]
                fallback_description = "Extract requested collection"
                action = "extract_visible_links" if collection_intent == "visible_links" else "extract_by_intent"
                args = (
                    {"output_key": output_key}
                    if action == "extract_visible_links"
                    else {"intent": collection_intent, "output_key": output_key}
                )
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
            if not required_hint or required_hint.issubset({"page_snapshot", "final_url", "page_title"}):
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
        haystack = " ".join(hints).casefold()
        package_context = any(token in haystack for token in ("package", "library", "module"))
        metadata_context = any(token in haystack for token in ("version", "description", "summary", "metadata"))
        return package_context and metadata_context

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
            ("product_cards", ("product", "products", "product_card", "product_cards", "товар", "карточ")),
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
