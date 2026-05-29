from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BrowserGymStepRecord(BaseModel):
    step_idx: int
    url: str = ""
    action: str = ""
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    info_summary: dict = Field(default_factory=dict)
    internal_plan: dict | None = None
    selected_step: dict | None = None
    extracted_value: str | None = None
    rationale: str | None = None
    action_rationale: str | None = None
    action_string: str | None = None
    miniwob_instruction: str | None = None
    mapping_error: str | None = None
    action_string_before_mapping: str | None = None
    action_string_after_mapping: str | None = None
    selected_candidate: dict | None = None
    selected_candidate_bid: str | None = None
    bid_source: str | None = None
    selected_candidate_verbose: dict | None = None
    clickable_candidates_count: int | None = None
    page_candidate_extraction_failed: bool | None = None
    mapping_strategy: str | None = None
    mapping_diagnostics: dict | None = None
    fallback_used: bool = False
    fallback_type: str | None = None
    fallback_reward: float | None = None
    fallback_terminated: bool | None = None
    vision_used: bool = False
    vision_image_present: bool = False
    llm_used: bool = False
    planner_used: bool = False
    verifier_used: bool = False
    policy_used: bool = False
    extraction_controller_used: bool = False
    visual_controller_used: bool = False
    skill_name: str | None = None
    controller_name: str | None = None
    raw_llm_output_present: bool = False
    error: str | None = None


class BrowserGymRunReport(BaseModel):
    env_id: str
    goal: str
    status: str
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    steps: list[BrowserGymStepRecord] = Field(default_factory=list)
    runtime_sec: float = 0.0
    failure_stage: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    final_answer: str | None = None
    output_path: str | None = None
    steps_count: int | None = None
    success: bool | None = None
    benchmark: str | None = None
    task_name: str | None = None
    llm_used: bool = False
    planner_used: bool = False
    verifier_used: bool = False
    policy_used: bool = False
    extraction_controller_used: bool = False
    visual_controller_used: bool = False
    skill_name: str | None = None
    controller_name: str | None = None
    raw_llm_output_present: bool = False


def classify_miniwob_step(step: BrowserGymStepRecord) -> dict[str, Any]:
    strategy = str(step.mapping_strategy or "").strip()
    diagnostics = step.mapping_diagnostics if isinstance(step.mapping_diagnostics, dict) else {}
    action_before = str(step.action_string_before_mapping or "")
    rationale = str(step.rationale or step.action_rationale or "")

    extraction_used = strategy.startswith("extraction_") or bool(diagnostics.get("extraction_intent"))
    policy_used = strategy.startswith("policy_") or bool(diagnostics.get("policy_pre_llm_used"))
    visual_used = (
        strategy.startswith("policy_count_shape")
        or strategy.startswith("policy_grid_coordinate")
        or strategy.startswith("visual_")
        or strategy == "visual_spatial_controller_required"
        or str(diagnostics.get("failure_stage", "")).startswith("visual_spatial")
    )
    direct_llm_like = not (extraction_used or policy_used or visual_used)
    raw_llm_output_present = bool(step.internal_plan or (direct_llm_like and action_before) or "llm" in rationale.lower())
    llm_used = raw_llm_output_present and direct_llm_like

    if extraction_used:
        controller_name = "extraction_controller"
    elif policy_used:
        controller_name = "miniwob_policy"
    elif visual_used:
        controller_name = "visual_spatial_controller"
    elif llm_used:
        controller_name = "llm_direct_action"
    else:
        controller_name = "browsergym_adapter"

    skill_name = _skill_name_for_step(strategy=strategy, diagnostics=diagnostics)
    return {
        "llm_used": bool(llm_used),
        "planner_used": False,
        "verifier_used": False,
        "policy_used": bool(policy_used),
        "extraction_controller_used": bool(extraction_used),
        "visual_controller_used": bool(visual_used),
        "skill_name": skill_name,
        "controller_name": controller_name,
        "raw_llm_output_present": bool(raw_llm_output_present),
    }


def enrich_miniwob_report(report: BrowserGymRunReport) -> BrowserGymRunReport:
    step_flags = []
    for step in report.steps:
        flags = classify_miniwob_step(step)
        for key, value in flags.items():
            setattr(step, key, value)
        step_flags.append(flags)

    report.planner_used = False
    report.verifier_used = False
    report.llm_used = any(flag["llm_used"] for flag in step_flags)
    report.policy_used = any(flag["policy_used"] for flag in step_flags)
    report.extraction_controller_used = any(flag["extraction_controller_used"] for flag in step_flags)
    report.visual_controller_used = any(flag["visual_controller_used"] for flag in step_flags)
    report.raw_llm_output_present = any(flag["raw_llm_output_present"] for flag in step_flags)
    report.skill_name = _skill_name_for_task(report.task_name or report.env_id) or next((flag["skill_name"] for flag in step_flags if flag.get("skill_name")), None)
    report.controller_name = next((flag["controller_name"] for flag in step_flags if flag.get("controller_name")), None)
    return report


def _skill_name_for_task(task_name: str | None) -> str | None:
    value = str(task_name or "").lower()
    if any(token in value for token in ["find-midpoint", "circle-center", "bisect-angle"]):
        return "canvas_geometry"
    if any(token in value for token in ["count-shape", "identify-shape", "count-sides", "grid-coordinate"]):
        return "visual_svg_recognition"
    return None


def _skill_name_for_step(*, strategy: str, diagnostics: dict[str, Any]) -> str | None:
    extraction_intent = str(diagnostics.get("extraction_intent") or "").lower()
    combined = f"{strategy} {extraction_intent}".lower()
    if any(token in combined for token in ["find_tree", "tree_node", "navigate_tree"]):
        return "tree_navigation"
    if any(token in combined for token in ["email", "row", "inbox"]):
        return "row_list_email_action"
    if any(token in combined for token in ["max_numeric", "parity", "numeric", "odd", "even"]):
        return "numeric_extraction"
    if any(token in combined for token in ["ordinal_word", "fill", "textbox", "enter_text", "focus_text"]):
        return "semantic_fill"
    if any(token in combined for token in ["autocomplete", "list", "option", "checkbox", "radio", "tab", "collapsible"]):
        return "select_list_autocomplete"
    if any(token in combined for token in ["shape", "grid", "svg", "visual"]):
        return "visual_svg_recognition"
    if any(token in combined for token in ["midpoint", "circle_center", "bisect"]):
        return "canvas_geometry"
    if any(token in combined for token in ["click", "button", "link", "menu"]):
        return "semantic_click"
    return None
