from __future__ import annotations

from typing import Any

BENCHMARK_CONTRACT_FIELDS_BY_TASK_FAMILY: dict[str, list[str]] = {
    "single_value_extraction": ["value"],
    "anchored_value_extraction": ["value"],
    "repeated_structured_items": ["items"],
    "navigation_then_extraction": ["value"],
    "multi_step_information_retrieval": ["combined_result"],
    "negative_or_ambiguous_case": [],
}

EXTRACTION_ACTIONS = {
    "extract_text",
    "extract_html",
    "extract_items",
    "extract_structured_items",
    "extract_value_from_section",
    "extract_structured_items_from_region",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
}
def required_contract_fields(*, task_family: str, scenario_required_fields: list[str] | None = None) -> list[str]:
    contract = BENCHMARK_CONTRACT_FIELDS_BY_TASK_FAMILY.get(str(task_family).strip())
    if contract:
        return list(contract)
    return [str(field).strip() for field in (scenario_required_fields or []) if str(field).strip()]


def normalize_payload_for_task_family_contract(payload: dict[str, Any], *, task_family: str) -> dict[str, Any]:
    normalized = dict(payload)
    steps = [dict(step) for step in normalized.get("steps", []) if isinstance(step, dict)]

    # Minimal safe mapping: keep scalar extraction stable for benchmark verifiers.
    if task_family in {"single_value_extraction", "anchored_value_extraction", "navigation_then_extraction"}:
        for step in steps:
            if step.get("action") in EXTRACTION_ACTIONS:
                step["save_as"] = "value"
    elif task_family == "repeated_structured_items":
        for step in steps:
            if step.get("action") in EXTRACTION_ACTIONS:
                step["save_as"] = "items"
                break
    elif task_family == "multi_step_information_retrieval":
        for step in steps:
            if step.get("action") == "compare_structured_values":
                step["save_as"] = "combined_result"
                break

    normalized["steps"] = steps
    return normalized
