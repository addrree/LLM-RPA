from __future__ import annotations

from typing import Any

CANONICAL_ACTIONS = {
    "open_url",
    "click",
    "type",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_items",
    "extract_structured_items",
    "screenshot",
    "observe_page",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
    "finish",
}

# Safe 1:1 aliases only (no semantic remapping).
ACTION_ALIASES = {
    "click_element": "click",
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
        if isinstance(action, str) and action in ACTION_ALIASES:
            step["action"] = ACTION_ALIASES[action]
            action_oov_detected = True
        normalized_steps.append(step)

    plan["steps"] = normalized_steps
    return plan, action_oov_detected
