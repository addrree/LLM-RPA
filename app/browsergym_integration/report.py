from __future__ import annotations

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
