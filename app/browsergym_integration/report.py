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
