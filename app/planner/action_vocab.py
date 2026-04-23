from __future__ import annotations

from typing import Any

ACTION_ALIASES: dict[str, str] = {}

CANONICAL_ACTIONS = {
    "open_url",
    "click",
    "navigate_to_relevant_section",
    "type",
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
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
    "finish",
}

def normalize_plan_action_aliases(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    plan = dict(payload) if isinstance(payload, dict) else {}
    steps = plan.get("steps")
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
