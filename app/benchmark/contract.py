from __future__ import annotations

from typing import Any

BENCHMARK_CONTRACT_FIELDS_BY_TASK_FAMILY: dict[str, list[str]] = {
    "single_value_extraction": ["value"],
    "anchored_value_extraction": ["anchor", "value"],
    "repeated_structured_items": ["items"],
    "navigation_then_extraction": ["source_page", "target_page", "value"],
    "multi_step_information_retrieval": ["source_a", "source_b", "combined_result"],
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
    if isinstance(scenario_required_fields, list):
        return [str(field).strip() for field in scenario_required_fields if str(field).strip()]
    return []


def normalize_payload_for_task_family_contract(payload: dict[str, Any], *, task_family: str) -> dict[str, Any]:
    normalized = dict(payload)
    steps = [dict(step) for step in normalized.get("steps", []) if isinstance(step, dict)]

    # Temporary controlled rollback:
    # keep only minimal, safe rewrites required by current benchmark consumers.
    if task_family == "single_value_extraction":
        for step in steps:
            if step.get("action") in EXTRACTION_ACTIONS:
                step["save_as"] = "value"

    elif task_family == "multi_step_information_retrieval":
        compare_idx = _find_first_index(steps, lambda step: step.get("action") == "compare_structured_values")
        if compare_idx is not None:
            steps[compare_idx]["save_as"] = "combined_result"

    normalized["steps"] = steps
    return normalized


def _find_first_index(steps: list[dict[str, Any]], predicate) -> int | None:
    for idx, step in enumerate(steps):
        if predicate(step):
            return idx
    return None
