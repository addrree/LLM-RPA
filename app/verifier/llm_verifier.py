import json
import os
import re
from copy import deepcopy

from app.schemas.execution import ExecutionResult, LLMArtifact
from app.schemas.task_spec import TaskSpec
from app.schemas.verification import VerificationPackage, VerificationVerdict
from app.utils.llm_client import LLMClient, LLMClientError


VERIFIER_SYSTEM_PROMPT = """
Ты независимый модуль верификации результата веб-автоматизации.

Верни строго JSON-объект (без markdown и без пояснений) вида:
{
  "task_completed": true/false,
  "confidence": 0.0,
  "verdict": "accept" | "reject" | "uncertain",
  "issues": ["..."],
  "summary": "..."
}

Правила:
1) confidence обязательно от 0 до 1.
2) verdict=accept только если semantic required_fields заполнены и цель достигнута.
3) screenshot_path — технический артефакт, не бизнес-поле цели.
4) Если данных не хватает — verdict=uncertain или reject и пояснение в issues.
"""


TECHNICAL_REQUIRED_FIELDS = {"screenshot_path", "screenshot", "artifact_screenshot"}
NEGATIVE_PROBE_ACTIONS = {
    "extract_text",
    "extract_by_intent",
    "extract_visible_links",
    "extract_pattern_from_page_text",
    "extract_value_near_anchor",
    "find_row_by_condition",
    "extract_structured_items",
    "extract_section_lines",
    "extract_value_from_section",
    "extract_html",
}


class LLMVerifier:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None

    def verify(
        self,
        plan: TaskSpec,
        result: ExecutionResult,
        benchmark_context: dict | None = None,
    ) -> VerificationVerdict:
        if result.status != "success":
            self.last_artifact = None
            return VerificationVerdict(
                task_completed=False,
                confidence=0.0,
                verdict="reject",
                issues=["Execution status is not success."],
                summary="Verifier rejected the run because execution did not finish successfully.",
            )

        semantic_required_fields = [
            field for field in plan.expected_result.required_fields if field not in TECHNICAL_REQUIRED_FIELDS
        ]
        verifier_extracted_data = self._normalized_extracted_data_for_verifier(
            required_fields=semantic_required_fields,
            result=result,
        )
        deterministic_issues = self._validate_structured_compare_contract(result.extracted_data)
        deterministic_issues.extend(
            self._validate_semantic_content_presence(
                required_fields=semantic_required_fields,
                extracted_data=verifier_extracted_data,
            )
        )
        deterministic_issues.extend(
            self._validate_semantic_value_quality(
                required_fields=semantic_required_fields,
                extracted_data=verifier_extracted_data,
                goal=plan.goal,
            )
        )
        deterministic_issues.extend(
            self._validate_single_value_key_alignment(
                required_fields=semantic_required_fields,
                extracted_data=result.extracted_data,
                benchmark_context=benchmark_context or {},
            )
        )
        deterministic_issues.extend(
            self._validate_collection_projection_quality(
                plan=plan,
                extracted_data=verifier_extracted_data,
            )
        )
        deterministic_issues.extend(
            self._validate_negative_probe_policy(
                plan=plan,
                result=result,
                benchmark_context=benchmark_context or {},
            )
        )
        if deterministic_issues:
            self.last_artifact = None
            return VerificationVerdict(
                task_completed=False,
                confidence=0.0,
                verdict="reject",
                issues=deterministic_issues,
                summary="Verifier rejected due to deterministic contract issues.",
            )

        fast_path_verdict = self._deterministic_fast_path_verdict(
            plan=plan,
            result=result,
            required_fields=semantic_required_fields,
            extracted_data=verifier_extracted_data,
            benchmark_context=benchmark_context or {},
        )
        if fast_path_verdict is not None:
            self.last_artifact = None
            return fast_path_verdict

        package = VerificationPackage(
            user_goal=plan.goal,
            expected_result_description=plan.expected_result.description,
            required_fields=semantic_required_fields,
            extracted_data=verifier_extracted_data,
            final_url=result.final_url,
            page_title=result.page_title,
            page_text_excerpt=result.page_text_excerpt,
            screenshot_path=result.screenshot_path,
            logs=[log.model_dump() for log in result.logs],
        )

        user_prompt = json.dumps(package.model_dump(), ensure_ascii=False, indent=2)
        verifier_image_path = result.screenshot_path if self._should_send_image_to_verifier(
            plan=plan,
            result=result,
            benchmark_context=benchmark_context or {},
        ) else None
        try:
            artifact = self.llm_client.generate_verifier_artifact(
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=verifier_image_path,
                stage="verifier",
            )
        except LLMClientError as exc:
            self.last_artifact = None
            return VerificationVerdict(
                task_completed=False,
                confidence=0.0,
                verdict="uncertain",
                issues=[f"Verifier LLM response could not be parsed: {exc}"],
                summary="Verifier returned invalid JSON; execution result was preserved for diagnostics.",
            )
        self.last_artifact = artifact
        return VerificationVerdict.model_validate(self._normalize_verdict_payload(artifact.parsed_response))

    @staticmethod
    def _should_send_image_to_verifier(
        *,
        plan: TaskSpec,
        result: ExecutionResult,
        benchmark_context: dict,
    ) -> bool:
        if not result.screenshot_path:
            return False
        mode = os.getenv("OLLAMA_VERIFIER_VISION_MODE", "auto").strip().casefold()
        if mode in {"always", "on", "true", "1", "vision"}:
            return True
        if mode in {"never", "off", "false", "0", "text"}:
            return False

        visual_actions = {"visual_observe", "visual_extract_object_count", "visual_click_by_geometry", "screenshot"}
        if any(step.action in visual_actions for step in plan.steps):
            return True

        task_family = str((benchmark_context or {}).get("task_family", "") or "").casefold()
        if any(token in task_family for token in ("visual", "spatial", "image", "screenshot")):
            return True

        goal = str(plan.goal or "").casefold()
        return any(
            token in goal
            for token in (
                "visual",
                "spatial",
                "image",
                "screenshot",
                "canvas",
                "coordinate",
                "визуал",
                "скрин",
                "изображ",
                "координат",
            )
        )

    @staticmethod
    def _normalize_verdict_payload(payload: object) -> dict:
        data = dict(payload) if isinstance(payload, dict) else {}
        verdict = str(data.get("verdict") or "").strip().lower()
        if verdict not in {"accept", "reject", "uncertain"}:
            verdict = "uncertain"
        data["verdict"] = verdict
        data["task_completed"] = bool(data.get("task_completed", verdict == "accept"))
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        data["confidence"] = max(0.0, min(1.0, confidence))
        issues = data.get("issues")
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        data["issues"] = [str(issue) for issue in issues if str(issue).strip()]
        if not str(data.get("summary", "") or "").strip():
            data["summary"] = "; ".join(data["issues"]) if data["issues"] else f"Verifier returned {verdict}."
        return data

    @staticmethod
    def _validate_structured_compare_contract(extracted_data: dict) -> list[str]:
        if not isinstance(extracted_data, dict):
            return []
        if "section_a_data" not in extracted_data and "section_b_data" not in extracted_data:
            return []
        if "structured_comparison" not in extracted_data:
            return ["Structured comparison is missing for section_a_data/section_b_data compare pipeline."]
        comparison = extracted_data.get("structured_comparison")
        if not isinstance(comparison, dict):
            return ["structured_comparison must be an object."]
        if "exact_match" not in comparison or "status" not in comparison:
            return ["structured_comparison must include exact_match and status fields."]
        compare_status = extracted_data.get("compare_status")
        if compare_status is None:
            return ["compare_status is missing for structured comparison result."]
        if str(compare_status) != str(comparison.get("status")):
            return ["compare_status must match structured_comparison.status."]
        return []

    @staticmethod
    def _validate_semantic_content_presence(*, required_fields: list[str], extracted_data: dict) -> list[str]:
        required = [str(field).strip() for field in required_fields if str(field).strip()]
        if not required or not isinstance(extracted_data, dict):
            return []
        technical = {
            "page_snapshot",
            "screenshot_path",
            "screenshot",
            "artifact_screenshot",
            "final_url",
            "current_url",
            "url",
            "page_title",
            "current_title",
            "title",
        }
        content_required = [field for field in required if field not in technical]
        if not content_required:
            return []
        present_content = [
            field for field in content_required
            if LLMVerifier._has_meaningful_value(LLMVerifier._first_meaningful_value_for_aliases((field.casefold(),), extracted_data))
        ]
        if present_content:
            return []
        return ["Semantic required fields are missing; URL/title/page_snapshot alone are not content success."]

    @staticmethod
    def _validate_semantic_value_quality(*, required_fields: list[str], extracted_data: dict, goal: str) -> list[str]:
        if not isinstance(extracted_data, dict):
            return []
        issues: list[str] = []
        normalized_required = [str(field).strip() for field in required_fields if str(field).strip()]
        normalized_goal = str(goal).lower()
        negative_like_goal = any(token in normalized_goal for token in ["absent", "ambiguous", "uncertainty", "not-found"])

        for field in normalized_required:
            if field not in extracted_data:
                continue
            value = extracted_data.get(field)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            if field == "status":
                status_token = text.lower().replace(" ", "_")
                allowed_prefixes = (
                    "not_found",
                    "ambiguous",
                    "uncertain",
                    "unknown",
                    "missing",
                    "no_match",
                    "not_available",
                )
                if len(status_token) > 32 or not status_token.startswith(allowed_prefixes):
                    issues.append(
                        "status field must be a compact uncertainty token (e.g., not_found/ambiguous), not broad prose."
                    )
            if len(text) >= 120 and LLMVerifier._looks_like_sentence_prose(text):
                issues.append(
                    f"Field '{field}' looks like broad page prose, not a concrete extracted value."
                )
            field_hint = field.casefold()
            if (
                ("name" in field_hint or "title" in field_hint)
                and LLMVerifier._looks_like_countish_value(text)
            ):
                issues.append(
                    f"Field '{field}' looks like a count/value, not a name/title."
                )
            if ("count" in field_hint or "number" in field_hint or field_hint.endswith("_total")) and not LLMVerifier._looks_like_countish_value(text):
                issues.append(
                    f"Field '{field}' must look like a numeric/count value."
                )
            if "email" in field_hint and not LLMVerifier._looks_like_email(text):
                issues.append(f"Field '{field}' must look like an email address.")
            if ("phone" in field_hint or "tel" == field_hint) and not LLMVerifier._looks_like_phone(text):
                issues.append(f"Field '{field}' must look like a phone number.")
            if (
                ("url" in field_hint or field_hint in {"href", "link", "final_url", "current_url"})
                and not LLMVerifier._looks_like_url_or_path(text)
            ):
                issues.append(f"Field '{field}' must look like a URL or path.")
            if negative_like_goal and field in {"value", "status"} and len(text) >= 80 and LLMVerifier._looks_like_sentence_prose(text):
                issues.append(
                    f"Negative/ambiguous goal requires explicit uncertainty semantics; field '{field}' contains broad prose."
                )
        return issues

    @staticmethod
    def _looks_like_sentence_prose(text: str) -> bool:
        token_count = len([token for token in text.split() if token])
        punctuation_count = sum(text.count(mark) for mark in [".", ";", ":"])
        return token_count >= 14 and punctuation_count >= 1

    @staticmethod
    def _looks_like_countish_value(text: str) -> bool:
        compact = str(text or "").strip().casefold()
        if not re.search(r"\d", compact):
            return False
        unit_tokens = ("article", "articles", "item", "items", "result", "results", "row", "rows", "count", "total")
        if any(token in compact for token in unit_tokens):
            return True
        alnum_chars = [char for char in compact if char.isalnum()]
        digit_chars = [char for char in alnum_chars if char.isdigit()]
        return bool(alnum_chars) and len(digit_chars) / len(alnum_chars) >= 0.45

    @staticmethod
    def _looks_like_email(text: str) -> bool:
        return bool(re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(text or "").strip()))

    @staticmethod
    def _looks_like_phone(text: str) -> bool:
        value = str(text or "").strip()
        digits = re.sub(r"\D+", "", value)
        return len(digits) >= 7 and bool(re.search(r"^\+?[\d\s().-]+$", value))

    @staticmethod
    def _looks_like_url_or_path(text: str) -> bool:
        value = str(text or "").strip()
        return bool(re.match(r"^(https?://|/|#|mailto:|tel:)", value))

    @classmethod
    def _validate_collection_projection_quality(
        cls,
        *,
        plan: TaskSpec,
        extracted_data: dict,
    ) -> list[str]:
        if not isinstance(extracted_data, dict):
            return []
        collection_keys: set[str] = set()
        expected_fields_by_key: dict[str, set[str]] = {}
        for step in plan.steps:
            action = str(step.action or "").strip()
            args = step.args if isinstance(step.args, dict) else {}
            intent = str(args.get("intent", "") or "").strip().casefold()
            is_collection_step = action in {"extract_items", "extract_structured_items"} or (
                action == "extract_by_intent" and intent in {"card_items", "cards", "table_rows", "rows"}
            )
            if not is_collection_step:
                continue
            output_key = str(args.get("output_key", "") or step.save_as or "").strip()
            if not output_key:
                continue
            collection_keys.add(output_key)
            fields = args.get("fields")
            if isinstance(fields, dict):
                expected = {str(field).strip() for field in fields if str(field).strip()}
                if expected:
                    expected_fields_by_key.setdefault(output_key, set()).update(expected)

        issues: list[str] = []
        technical_item_fields = {
            "raw_text",
            "selector",
            "dom_path",
            "bbox",
            "confidence",
            "source",
            "match_scope",
        }
        semantic_fallback_fields = {
            "title",
            "name",
            "description",
            "summary",
            "snippet",
            "href",
            "url",
            "link",
        }
        for key in sorted(collection_keys):
            value = extracted_data.get(key)
            if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
                continue
            expected_fields = expected_fields_by_key.get(key) or semantic_fallback_fields
            has_projected_value = any(
                any(cls._has_meaningful_value(item.get(field)) for field in expected_fields)
                for item in value
            )
            has_nontechnical_value = any(
                any(
                    str(field).strip().casefold() not in technical_item_fields
                    and cls._has_meaningful_value(item_value)
                    for field, item_value in item.items()
                )
                for item in value
            )
            if not has_projected_value or not has_nontechnical_value:
                issues.append(
                    f"Collection '{key}' did not populate meaningful projected item fields; raw container text alone is insufficient."
                )
        return issues

    @staticmethod
    def _validate_single_value_key_alignment(
        *,
        required_fields: list[str],
        extracted_data: dict,
        benchmark_context: dict,
    ) -> list[str]:
        if not isinstance(extracted_data, dict):
            return []
        task_family = str(benchmark_context.get("task_family", "")).strip().lower()
        normalized_required = [str(field).strip() for field in required_fields if str(field).strip()]
        if task_family != "single_value_extraction" or normalized_required != ["value"]:
            return []
        if "value" in extracted_data:
            return []
        scalar_aliases = [
            key
            for key, value in extracted_data.items()
            if key != "value" and isinstance(value, (str, int, float, bool))
        ]
        if scalar_aliases:
            aliases = ", ".join(sorted(scalar_aliases))
            return [
                "single_value_extraction expects extracted_data.value, but scalar output was saved under "
                f"different key(s): {aliases}."
            ]
        return []

    @classmethod
    def _validate_negative_probe_policy(
        cls,
        *,
        plan: TaskSpec,
        result: ExecutionResult,
        benchmark_context: dict,
    ) -> list[str]:
        task_family = str(benchmark_context.get("task_family", "")).strip().lower()
        if task_family != "negative_or_ambiguous_case":
            return []

        attempted_actions = {
            str(log.action).strip()
            for log in result.logs
            if str(log.status).strip().lower() in {"success", "failed"}
        }
        probe_attempted = bool(attempted_actions.intersection(NEGATIVE_PROBE_ACTIONS))
        if probe_attempted:
            return []

        observe_attempted = "observe_page" in attempted_actions
        if observe_attempted and cls._has_explicit_negative_reasoning(result.extracted_data):
            return []

        if attempted_actions == {"open_url", "finish"} or not attempted_actions:
            return [
                "Negative benchmark scenario cannot be accepted with open_url -> finish only; at least one probe/extraction attempt is required."
            ]
        return [
            "Negative benchmark scenario requires probe/extraction action (or observe_page with explicit missing-field evidence)."
        ]

    @staticmethod
    def _has_explicit_negative_reasoning(extracted_data: dict) -> bool:
        if not isinstance(extracted_data, dict):
            return False
        status = str(extracted_data.get("status", "")).strip().lower().replace(" ", "_")
        if not status:
            return False
        allowed_prefixes = ("not_found", "ambiguous", "uncertain", "missing", "unknown", "not_available")
        if not status.startswith(allowed_prefixes):
            return False
        evidence_keys = ("reason", "explanation", "evidence", "missing_field_evidence", "probe_result")
        return any(str(extracted_data.get(key, "")).strip() for key in evidence_keys)

    @classmethod
    def _deterministic_fast_path_verdict(
        cls,
        *,
        plan: TaskSpec,
        result: ExecutionResult,
        required_fields: list[str],
        extracted_data: dict | None = None,
        benchmark_context: dict,
    ) -> VerificationVerdict | None:
        extracted_data = extracted_data if isinstance(extracted_data, dict) else (
            result.extracted_data if isinstance(result.extracted_data, dict) else {}
        )
        normalized_required = [str(field).strip() for field in required_fields if str(field).strip()]
        if not normalized_required:
            return None

        goal_lower = str(plan.goal or "").lower()
        negative_like_goal = any(token in goal_lower for token in ["absent", "ambiguous", "uncertainty", "not-found"])
        if negative_like_goal:
            return None

        generic_fields = [field for field in normalized_required if field != "count" and not field.endswith("_count")]
        if generic_fields:
            present_fields = [
                field
                for field in generic_fields
                if cls._has_meaningful_value(cls._value_for_required_field(field, result, extracted_data))
            ]
            missing_fields = [field for field in generic_fields if field not in present_fields]
            metadata_aliases = {
                alias
                for aliases in cls._field_alias_groups()
                for alias in aliases
            }
            deterministic_metadata = all(field.casefold() in metadata_aliases for field in generic_fields)
            if not missing_fields and len(generic_fields) == len(normalized_required) and deterministic_metadata:
                return VerificationVerdict(
                    task_completed=True,
                    confidence=0.91,
                    verdict="accept",
                    issues=[],
                    summary="Deterministic verifier accepted populated required fields.",
                )
            if present_fields and missing_fields:
                return VerificationVerdict(
                    task_completed=False,
                    confidence=0.58,
                    verdict="uncertain",
                    issues=["Partial required fields satisfied; missing: " + ", ".join(missing_fields)],
                    summary="Verifier saw partial success for required fields.",
                )

        anchor_object_verdict = cls._deterministic_anchor_object_fast_path(
            normalized_required=normalized_required,
            result=result,
            extracted_data=extracted_data,
        )
        if anchor_object_verdict is not None:
            return anchor_object_verdict

        count_fields = [field for field in normalized_required if field == "count" or field.endswith("_count")]
        if count_fields and len(count_fields) == len(normalized_required):
            missing_or_invalid = [
                field
                for field in count_fields
                if not isinstance(extracted_data.get(field), (int, float)) or extracted_data.get(field) < 0
            ]
            if not missing_or_invalid:
                return VerificationVerdict(
                    task_completed=True,
                    confidence=0.93,
                    verdict="accept",
                    issues=[],
                    summary="Deterministic verifier accepted populated count fields.",
                )

        task_family = str(benchmark_context.get("task_family", "")).strip().lower()
        if task_family == "single_value_extraction" and len(normalized_required) == 1:
            field = normalized_required[0]
            value = extracted_data.get(field)
            if isinstance(value, (int, float)) and field != "status":
                return VerificationVerdict(
                    task_completed=True,
                    confidence=0.95,
                    verdict="accept",
                    issues=[],
                    summary="Deterministic verifier accepted scalar extraction without LLM call.",
                )
            if isinstance(value, str):
                text = value.strip()
                if text and len(text) <= 160 and not cls._looks_like_sentence_prose(text):
                    return VerificationVerdict(
                        task_completed=True,
                        confidence=0.94,
                        verdict="accept",
                        issues=[],
                        summary="Deterministic verifier accepted compact scalar value without LLM call.",
                    )

        if task_family == "repeated_structured_items" and all(field in extracted_data for field in normalized_required):
            evaluator_metadata = benchmark_context.get("evaluator_metadata", {}) if isinstance(benchmark_context, dict) else {}
            expected_min_items = int(evaluator_metadata.get("expected_min_items", 0) or 0)
            expected_item_fields = [
                str(field).strip()
                for field in (evaluator_metadata.get("expected_item_fields") or [])
                if str(field).strip()
            ]
            for field in normalized_required:
                value = extracted_data.get(field)
                if not (isinstance(value, list) and value and all(isinstance(item, dict) for item in value)):
                    return None
                if expected_min_items > 0 and len(value) < expected_min_items:
                    return None
                if expected_item_fields and not all(
                    all(str(item.get(item_field, "")).strip() for item_field in expected_item_fields)
                    for item in value
                ):
                    return None
            return VerificationVerdict(
                task_completed=True,
                confidence=0.92,
                verdict="accept",
                issues=[],
                summary="Deterministic verifier accepted repeated structured extraction without LLM call.",
            )
        return None

    @classmethod
    def _deterministic_anchor_object_fast_path(
        cls,
        *,
        normalized_required: list[str],
        result: ExecutionResult,
        extracted_data: dict,
    ) -> VerificationVerdict | None:
        label_fields: list[str] = []
        count_fields: list[str] = []
        for field in normalized_required:
            field_hint = str(field or "").strip().casefold()
            if any(token in field_hint for token in ("name", "title", "label", "language")):
                label_fields.append(field)
            elif any(token in field_hint for token in ("count", "number", "total", "value", "article")):
                count_fields.append(field)
        if not label_fields or not count_fields:
            return cls._deterministic_anchor_parent_fast_path(
                normalized_required=normalized_required,
                result=result,
                extracted_data=extracted_data,
            )

        recognized = set(label_fields) | set(count_fields)
        for field in normalized_required:
            if field in recognized:
                continue
            value = cls._value_for_required_field(field, result, extracted_data)
            if cls._has_meaningful_value(value) and cls._looks_like_url_or_path(str(value)):
                continue
            return None

        missing: list[str] = []
        invalid: list[str] = []
        for field in label_fields:
            value = cls._value_for_required_field(field, result, extracted_data)
            text = str(value or "").strip()
            if not text:
                missing.append(field)
            elif cls._looks_like_countish_value(text) or len(text) > 160:
                invalid.append(field)
        for field in count_fields:
            value = cls._value_for_required_field(field, result, extracted_data)
            text = str(value or "").strip()
            if not text:
                missing.append(field)
            elif not cls._looks_like_countish_value(text):
                invalid.append(field)
        if missing:
            return VerificationVerdict(
                task_completed=False,
                confidence=0.58,
                verdict="uncertain",
                issues=["Partial anchor object fields satisfied; missing: " + ", ".join(missing)],
                summary="Verifier saw partial anchor object extraction.",
            )
        if invalid:
            return None
        return VerificationVerdict(
            task_completed=True,
            confidence=0.93,
            verdict="accept",
            issues=[],
            summary="Deterministic verifier accepted populated anchor object fields.",
        )

    @classmethod
    def _deterministic_anchor_parent_fast_path(
        cls,
        *,
        normalized_required: list[str],
        result: ExecutionResult,
        extracted_data: dict,
    ) -> VerificationVerdict | None:
        if not normalized_required:
            return None
        saw_anchor_parent = False
        for field in normalized_required:
            value = cls._value_for_required_field(field, result, extracted_data)
            if isinstance(value, dict) and cls._dict_has_anchor_object_shape(value):
                saw_anchor_parent = True
                continue
            if cls._has_meaningful_value(value) and cls._looks_like_url_or_path(str(value)):
                continue
            return None
        if not saw_anchor_parent:
            return None
        return VerificationVerdict(
            task_completed=True,
            confidence=0.93,
            verdict="accept",
            issues=[],
            summary="Deterministic verifier accepted populated nested anchor object.",
        )

    @classmethod
    def _dict_has_anchor_object_shape(cls, value: dict) -> bool:
        if not isinstance(value, dict):
            return False
        label_values: list[str] = []
        count_values: list[str] = []
        for key, item in value.items():
            key_hint = str(key or "").strip().casefold()
            text = str(item or "").strip() if item is not None else ""
            if not text:
                continue
            if any(token in key_hint for token in ("name", "title", "label", "language")):
                label_values.append(text)
            elif any(token in key_hint for token in ("count", "number", "total", "value", "article")):
                count_values.append(text)
        has_label = any(text and not cls._looks_like_countish_value(text) and len(text) <= 160 for text in label_values)
        has_count = any(cls._looks_like_countish_value(text) for text in count_values)
        if has_label and has_count:
            return True
        for item in value.values():
            if isinstance(item, dict) and cls._dict_has_anchor_object_shape(item):
                return True
        return False

    @classmethod
    def _normalized_extracted_data_for_verifier(
        cls,
        *,
        required_fields: list[str],
        result: ExecutionResult,
    ) -> dict:
        source = result.extracted_data if isinstance(result.extracted_data, dict) else {}
        normalized = deepcopy(source)
        if not isinstance(normalized, dict):
            normalized = {}

        if result.final_url:
            normalized.setdefault("final_url", result.final_url)
        if result.page_title:
            normalized.setdefault("page_title", result.page_title)

        for field in required_fields:
            field_name = str(field or "").strip()
            if not field_name:
                continue
            value = cls._value_for_required_field(field_name, result, normalized)
            if cls._has_meaningful_value(value):
                normalized.setdefault(field_name, value)

        for aliases in cls._field_alias_groups():
            value = cls._first_meaningful_value_for_aliases(aliases, normalized)
            if not cls._has_meaningful_value(value):
                continue
            for alias in aliases:
                normalized.setdefault(alias, value)
        return normalized

    @staticmethod
    def _value_for_required_field(field: str, result: ExecutionResult, extracted_data: dict) -> object:
        normalized = str(field or "").strip().lower()
        aliases = LLMVerifier._aliases_for_field(normalized)
        value = LLMVerifier._first_meaningful_value_for_aliases(aliases, extracted_data)
        if LLMVerifier._has_meaningful_value(value):
            return value
        if normalized in {"final_url", "current_url", "url"}:
            return result.final_url
        if normalized in {"page_title", "current_title", "title", "name"}:
            return result.page_title
        return extracted_data.get(field)

    @staticmethod
    def _aliases_for_field(field: str) -> tuple[str, ...]:
        normalized = str(field or "").strip().lower()
        for aliases in LLMVerifier._field_alias_groups():
            if normalized in aliases:
                return aliases
        return (normalized,)

    @staticmethod
    def _field_alias_groups() -> tuple[tuple[str, ...], ...]:
        return (
            ("title", "name", "page_title", "current_title"),
            ("url", "final_url", "current_url"),
            ("description", "summary", "snippet"),
        )

    @classmethod
    def _first_meaningful_value_for_aliases(cls, aliases: tuple[str, ...], data: object) -> object:
        if not isinstance(data, dict):
            return None
        lower_aliases = {alias.lower() for alias in aliases}
        for key, value in data.items():
            if str(key).strip().lower() in lower_aliases and cls._has_meaningful_value(value):
                return value
        for value in data.values():
            if isinstance(value, dict):
                nested = cls._first_meaningful_value_for_aliases(aliases, value)
                if cls._has_meaningful_value(nested):
                    return nested
            elif isinstance(value, list):
                for item in value:
                    nested = cls._first_meaningful_value_for_aliases(aliases, item)
                    if cls._has_meaningful_value(nested):
                        return nested
        return None

    @staticmethod
    def _has_meaningful_value(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        if isinstance(value, dict):
            return len(value) > 0
        return True
