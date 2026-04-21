import json

from app.schemas.execution import ExecutionResult, LLMArtifact
from app.schemas.task_spec import TaskSpec
from app.schemas.verification import VerificationPackage, VerificationVerdict
from app.utils.llm_client import LLMClient


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
        deterministic_issues = self._validate_structured_compare_contract(result.extracted_data)
        deterministic_issues.extend(
            self._validate_semantic_value_quality(
                required_fields=semantic_required_fields,
                extracted_data=result.extracted_data,
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
        if deterministic_issues:
            self.last_artifact = None
            return VerificationVerdict(
                task_completed=False,
                confidence=0.0,
                verdict="reject",
                issues=deterministic_issues,
                summary="Verifier rejected due to invalid structured comparison contract.",
            )

        fast_path_verdict = self._deterministic_fast_path_verdict(
            plan=plan,
            result=result,
            required_fields=semantic_required_fields,
            benchmark_context=benchmark_context or {},
        )
        if fast_path_verdict is not None:
            self.last_artifact = None
            return fast_path_verdict

        package = VerificationPackage(
            user_goal=plan.goal,
            expected_result_description=plan.expected_result.description,
            required_fields=semantic_required_fields,
            extracted_data=result.extracted_data,
            final_url=result.final_url,
            page_title=result.page_title,
            page_text_excerpt=result.page_text_excerpt,
            screenshot_path=result.screenshot_path,
            logs=[log.model_dump() for log in result.logs],
        )

        user_prompt = json.dumps(package.model_dump(), ensure_ascii=False, indent=2)
        artifact = self.llm_client.generate_verifier_artifact(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_path=result.screenshot_path,
            stage="verifier",
        )
        self.last_artifact = artifact
        return VerificationVerdict.model_validate(artifact.parsed_response)

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
    def _deterministic_fast_path_verdict(
        cls,
        *,
        plan: TaskSpec,
        result: ExecutionResult,
        required_fields: list[str],
        benchmark_context: dict,
    ) -> VerificationVerdict | None:
        extracted_data = result.extracted_data if isinstance(result.extracted_data, dict) else {}
        normalized_required = [str(field).strip() for field in required_fields if str(field).strip()]
        if not normalized_required:
            return None

        goal_lower = str(plan.goal or "").lower()
        negative_like_goal = any(token in goal_lower for token in ["absent", "ambiguous", "uncertainty", "not-found"])
        if negative_like_goal:
            return None

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
            expected_min_items = int(benchmark_context.get("expected_min_items", 0) or 0)
            expected_item_fields = [
                str(field).strip()
                for field in (benchmark_context.get("expected_item_fields") or [])
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
