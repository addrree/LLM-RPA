from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class VerificationPackage(BaseModel):
    user_goal: str
    expected_result_description: str
    required_fields: List[str] = Field(default_factory=list)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    final_url: Optional[str] = None
    page_title: Optional[str] = None
    page_text_excerpt: Optional[str] = None
    screenshot_path: Optional[str] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)


class VerificationVerdict(BaseModel):
    task_completed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Literal["accept", "reject", "uncertain"]
    issues: List[str] = Field(default_factory=list)
    summary: str
