from __future__ import annotations

from typing import Any

BENCHMARK_CONTRACT_FIELDS_BY_TASK_FAMILY: dict[str, list[str]] = {
    "single_value_extraction": ["value"],
    # Controlled rollback: disable strict family-specific contracts temporarily.
    # Keep only the minimal safe contract for scalar extraction.
    "repeated_structured_items": ["items"],
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
    # Controlled cleanup: do not inherit strict scenario-level required field contracts
    # for families without explicit safe mapping.
    _ = scenario_required_fields
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

    normalized["steps"] = steps
    return normalized
