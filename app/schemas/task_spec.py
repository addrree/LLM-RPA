from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl


ActionType = Literal[
    "open_url",
    "click",
    "type",
    "wait_for",
    "extract_text",
    "extract_html",
    "screenshot",
    "finish"
]


class Constraints(BaseModel):
    max_steps: int = 10
    max_replans: int = 1
    timeout_sec: int = 30


class ExpectedResult(BaseModel):
    description: str
    required_fields: List[str] = Field(default_factory=list)


class ActionStep(BaseModel):
    step_id: int
    action: ActionType
    args: Dict[str, Any] = Field(default_factory=dict)
    save_as: Optional[str] = None


class TaskSpec(BaseModel):
    goal: str
    start_url: HttpUrl
    allowed_domains: List[str] = Field(default_factory=list)
    constraints: Constraints
    expected_result: ExpectedResult
    steps: List[ActionStep]