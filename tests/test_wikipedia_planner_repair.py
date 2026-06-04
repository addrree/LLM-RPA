from app.planner.prompts import PLANNER_SYSTEM_PROMPT, REPLANNER_SYSTEM_PROMPT, CORRECTIVE_REPLANNER_SYSTEM_PROMPT
from app.planner.replanner import Replanner


def test_planner_prompts_prefer_generic_anchor_extraction_with_regex_fallback():
    assert "anchor/value" in PLANNER_SYSTEM_PROMPT
    assert "<observed label>" in PLANNER_SYSTEM_PROMPT
    assert "extract_by_intent" in REPLANNER_SYSTEM_PROMPT
    assert "extract_pattern_from_page_text" in REPLANNER_SYSTEM_PROMPT
    assert "fallback" in REPLANNER_SYSTEM_PROMPT
    assert "Wikipedia English article count" not in PLANNER_SYSTEM_PROMPT


def test_corrective_prompt_forbids_unsupported_extract_value():
    assert "Wrong: extract_value" in CORRECTIVE_REPLANNER_SYSTEM_PROMPT
    assert "Never invent action names" in CORRECTIVE_REPLANNER_SYSTEM_PROMPT


def test_corrective_repair_rewrites_extract_value_pattern_intent():
    plan = {"steps": [{"action": "extract_value", "args": {"pattern": "Metric\\s+(\\d+)"}, "save_as": "value"}]}
    repaired = Replanner._repair_unsupported_extract_value_action(plan)
    assert repaired["steps"][0]["action"] == "extract_pattern_from_page_text"


def test_corrective_repair_rewrites_extract_value_anchor_intent():
    plan = {"steps": [{"action": "extract_value", "args": {"anchor_candidates": ["Metric"], "value_pattern": "(\\d+)"}, "save_as": "value"}]}
    repaired = Replanner._repair_unsupported_extract_value_action(plan)
    assert repaired["steps"][0]["action"] == "extract_value_near_anchor"
