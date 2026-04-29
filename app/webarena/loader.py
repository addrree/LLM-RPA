from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class WebArenaTaskConfig(BaseModel):
    task_id: str
    objective: str
    start_url: str
    allowed_domains: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    split: str = "custom"
    site: str = ""
    metadata: dict = Field(default_factory=dict)


class WebArenaTaskCollection(BaseModel):
    tasks: list[WebArenaTaskConfig]


def load_webarena_tasks(path: Path) -> list[WebArenaTaskConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [WebArenaTaskConfig.model_validate(item) for item in payload]
    collection = WebArenaTaskCollection.model_validate(payload)
    return collection.tasks
