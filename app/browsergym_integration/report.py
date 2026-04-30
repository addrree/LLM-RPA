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
