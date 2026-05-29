from __future__ import annotations

from typing import Any

ACTION_ALIASES: dict[str, str] = {
    "enter_text": "fill",
    "input_text": "fill",
    "select": "select_option",
    "choose_list": "select_option",
    "autocomplete": "select_autocomplete",
    "choose_autocomplete": "choose_autocomplete_suggestion",
    "click_semantic": "click_by_semantic_target",
    "semantic_click": "click_by_semantic_target",
    "fill_semantic": "fill_by_semantic_target",
    "semantic_fill": "fill_by_semantic_target",
    "select_semantic": "select_by_semantic_target",
    "visible_links": "extract_visible_links",
}

CANONICAL_ACTIONS = {
    "open_url",
    "click",
    "navigate_to_relevant_section",
    "type",
    "fill",
    "focus",
    "clear",
    "press",
    "hover",
    "select_option",
    "check",
    "uncheck",
    "select_autocomplete",
    "choose_autocomplete_suggestion",
    "choose_date",
    "click_by_semantic_target",
    "fill_by_semantic_target",
    "select_by_semantic_target",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_items",
    "extract_structured_items",
    "extract_section_lines",
    "extract_value_from_section",
    "extract_structured_items_from_region",
    "compare_structured_values",
    "assert_page_contains",
    "screenshot",
    "observe_page",
    "extract_by_intent",
    "extract_visible_links",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
    "find_row_by_condition",
    "click_row_action",
    "visual_observe",
    "visual_extract_object_count",
    "visual_click_by_geometry",
    "finish",
}

def normalize_plan_action_aliases(payload: dict[str, Any] | list[Any]) -> tuple[dict[str, Any], bool]:
    if isinstance(payload, list):
        plan = {"steps": payload}
    else:
        plan = dict(payload) if isinstance(payload, dict) else {}
    steps = plan.get("steps")
    if not isinstance(steps, list) and isinstance(plan.get("actions"), list):
        steps = plan.pop("actions")
        plan["steps"] = steps
    if not isinstance(steps, list) and isinstance(plan.get("tasks"), list):
        steps = plan.pop("tasks")
        plan["steps"] = steps
    if not isinstance(steps, list) and isinstance(plan.get("order"), list):
        steps = plan.pop("order")
        plan["steps"] = steps
    if not isinstance(steps, list):
        return plan, False

    action_oov_detected = False
    normalized_steps: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            normalized_steps.append(raw_step)
            continue
        step = dict(raw_step)
        action = step.get("action")
        if isinstance(action, str):
            canonical = ACTION_ALIASES.get(action, action)
            step["action"] = canonical
            if canonical not in CANONICAL_ACTIONS:
                action_oov_detected = True
        normalized_steps.append(step)

    plan["steps"] = normalized_steps
    return plan, action_oov_detected
