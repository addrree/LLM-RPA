from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class HeadingSnapshot(BaseModel):
    text: str
    level: str
    index: int
    visible: bool
    dom_path: str = ""
    region: str = "unknown"
    preview_after: List[str] = Field(default_factory=list)
    line_count_after: int = 0
    is_content_heading: bool = False


class PageSnapshot(BaseModel):
    url: str
    title: str
    screenshot_path: str
    page_text_excerpt: str
    visible_headings: List[str] = Field(default_factory=list)
    headings: List[HeadingSnapshot] = Field(default_factory=list)
    visible_labels: List[str] = Field(default_factory=list)
    visible_buttons: List[str] = Field(default_factory=list)
    visible_inputs: List[str] = Field(default_factory=list)
    visible_links: List[dict[str, Any]] = Field(default_factory=list)
    text_lines: List[str] = Field(default_factory=list)
    candidates: List[dict[str, Any]] = Field(default_factory=list)
    buttons: List[dict[str, Any]] = Field(default_factory=list)
    links: List[dict[str, Any]] = Field(default_factory=list)
    inputs: List[dict[str, Any]] = Field(default_factory=list)
    rows: List[dict[str, Any]] = Field(default_factory=list)
    tables: List[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime
    page_text: Optional[str] = None
