from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepLog(BaseModel):
    step_id: int
    action: str
    status: str
    message: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class GenerationMetadata(BaseModel):
    backend: str
    model: str
    source: str
    fallback_used: bool = False


class LLMArtifact(BaseModel):
    raw_response: str
    parsed_response: Dict[str, Any]
    generation: GenerationMetadata


class ExecutionResult(BaseModel):
    status: str
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    final_url: Optional[str] = None
    page_title: Optional[str] = None
    page_text_excerpt: Optional[str] = None
    screenshot_path: Optional[str] = None
    logs: List[StepLog] = Field(default_factory=list)
    error_message: Optional[str] = None
    failure_type: Optional[str] = None
    failed_action: Optional[str] = None
    failed_args: Dict[str, Any] = Field(default_factory=dict)
    technical_failure: bool = False
    retry_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
