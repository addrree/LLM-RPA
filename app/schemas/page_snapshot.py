from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HeadingSnapshot(BaseModel):
    text: str
    level: str
    index: int
    visible: bool
    preview_after: List[str] = Field(default_factory=list)
    line_count_after: int = 0


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
    timestamp: datetime
    page_text: Optional[str] = None
