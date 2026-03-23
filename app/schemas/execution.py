from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StepLog(BaseModel):
    step_id: int
    action: str
    status: str
    message: Optional[str] = None


class ExecutionResult(BaseModel):
    status: str
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    final_url: Optional[str] = None
    page_title: Optional[str] = None
    page_text_excerpt: Optional[str] = None
    screenshot_path: Optional[str] = None
    logs: List[StepLog] = Field(default_factory=list)
    error_message: Optional[str] = None