from __future__ import annotations

from app.benchmark.scenario_loader import ScenarioCategory

BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY: dict[ScenarioCategory, list[str]] = {
    "single_value_extraction": [
        "open_url",
        "extract_text",
        "extract_pattern_from_page_text",
        "finish",
    ],
    "anchored_value_extraction": [
        "open_url",
        "observe_page",
        "extract_value_near_anchor",
        "finish",
    ],
    "repeated_structured_items": [
        "open_url",
        "observe_page",
        "extract_structured_items",
        "finish",
    ],
    "navigation_then_extraction": [
        "open_url",
        "click",
        "wait_for",
        "observe_page",
        "extract_text",
        "extract_structured_items",
        "finish",
    ],
    "multi_step_information_retrieval": [
        "open_url",
        "observe_page",
        "extract_structured_items",
        "extract_text",
        "compare_structured_values",
        "finish",
    ],
    "negative_or_ambiguous_case": [
        "open_url",
        "observe_page",
        "extract_text",
        "extract_pattern_from_page_text",
        "finish",
    ],
}


def build_benchmark_context(
    *,
    category: ScenarioCategory,
    task_family: str | None = None,
    required_top_level_fields: list[str] | None = None,
    expected_item_fields: list[str] | None = None,
    expected_min_items: int | None = None,
    anchor_candidates: list[str] | None = None,
) -> dict:
    family = task_family or category
    return {
        "is_benchmark": True,
        "task_family": family,
        "allowed_actions": list(BENCHMARK_ALLOWED_ACTIONS_BY_CATEGORY[category]),
        "required_top_level_fields": list(required_top_level_fields or []),
        "expected_item_fields": list(expected_item_fields or []),
        "expected_min_items": int(expected_min_items or 0),
        "anchor_candidates": list(anchor_candidates or []),
    }
