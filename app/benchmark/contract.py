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
NAVIGATION_ACTIONS = {"click", "navigate_to_relevant_section"}


def required_contract_fields(*, task_family: str, scenario_required_fields: list[str] | None = None) -> list[str]:
    del scenario_required_fields
    contract = BENCHMARK_CONTRACT_FIELDS_BY_TASK_FAMILY.get(str(task_family).strip())
    return list(contract) if contract else []


def normalize_payload_for_task_family_contract(payload: dict[str, Any], *, task_family: str) -> dict[str, Any]:
    normalized = dict(payload)
    steps = [dict(step) for step in normalized.get("steps", []) if isinstance(step, dict)]
    expected = normalized.get("expected_result") if isinstance(normalized.get("expected_result"), dict) else {}

    if task_family == "single_value_extraction":
        for step in steps:
            if step.get("action") in EXTRACTION_ACTIONS:
                step["save_as"] = "value"

    elif task_family == "anchored_value_extraction":
        first_extract_idx = _find_first_index(steps, lambda step: step.get("action") in EXTRACTION_ACTIONS)
        if first_extract_idx is not None:
            steps[first_extract_idx]["save_as"] = "value"
            _ensure_observe_step(steps, index=first_extract_idx, save_as="anchor")

    elif task_family == "repeated_structured_items":
        for step in steps:
            if step.get("action") in EXTRACTION_ACTIONS:
                step["save_as"] = "items"

    elif task_family == "navigation_then_extraction":
        nav_idx = _find_first_index(steps, lambda step: step.get("action") in NAVIGATION_ACTIONS)
        if nav_idx is not None:
            _ensure_observe_step(steps, index=nav_idx, save_as="source_page")
            nav_idx = _find_first_index(steps, lambda step: step.get("action") in NAVIGATION_ACTIONS)

        target_extract_idx = None
        for idx, step in enumerate(steps):
            if step.get("action") not in EXTRACTION_ACTIONS:
                continue
            if nav_idx is None or idx > nav_idx:
                target_extract_idx = idx
                break
        if target_extract_idx is None:
            finish_idx = _find_first_index(steps, lambda step: step.get("action") == "finish")
            insert_idx = finish_idx if finish_idx is not None else len(steps)
            steps.insert(insert_idx, {"action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"})
            target_extract_idx = insert_idx

        if nav_idx is not None:
            has_target_observe = False
            for idx in range(nav_idx + 1, min(target_extract_idx + 1, len(steps))):
                step = steps[idx]
                if step.get("action") == "observe_page" and str(step.get("save_as", "")).strip() == "target_page":
                    has_target_observe = True
                    break
            if not has_target_observe:
                steps.insert(target_extract_idx, {"action": "observe_page", "args": {}, "save_as": "target_page"})
                target_extract_idx += 1

        steps[target_extract_idx]["save_as"] = "value"

    elif task_family == "multi_step_information_retrieval":
        extraction_indices = [
            idx
            for idx, step in enumerate(steps)
            if step.get("action") in {"extract_value_from_section", "extract_structured_items_from_region"}
        ]
        if extraction_indices:
            steps[extraction_indices[0]]["save_as"] = "source_a"
        if len(extraction_indices) > 1:
            steps[extraction_indices[1]]["save_as"] = "source_b"
        compare_idx = _find_first_index(steps, lambda step: step.get("action") == "compare_structured_values")
        if compare_idx is not None:
            compare_step = steps[compare_idx]
            args = compare_step.get("args") if isinstance(compare_step.get("args"), dict) else {}
            args["left_key"] = "source_a"
            args["right_key"] = "source_b"
            compare_step["args"] = args
            compare_step["save_as"] = "combined_result"

    contract_fields = required_contract_fields(task_family=task_family, scenario_required_fields=expected.get("required_fields"))
    if contract_fields:
        expected["required_fields"] = contract_fields
    normalized["expected_result"] = expected
    normalized["steps"] = steps
    return normalized


def _find_first_index(steps: list[dict[str, Any]], predicate) -> int | None:
    for idx, step in enumerate(steps):
        if predicate(step):
            return idx
    return None


def _ensure_observe_step(
    steps: list[dict[str, Any]],
    *,
    index: int,
    save_as: str,
    start_after: int | None = None,
) -> None:
    lower_bound = 0 if start_after is None else start_after + 1
    for idx in range(lower_bound, min(index + 1, len(steps))):
        step = steps[idx]
        if step.get("action") == "observe_page" and str(step.get("save_as", "")).strip() == save_as:
            return
    steps.insert(index, {"action": "observe_page", "args": {}, "save_as": save_as})
