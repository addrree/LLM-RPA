from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ScenarioCategory = Literal[
    "single_value_extraction",
    "anchored_value_extraction",
    "repeated_structured_items",
    "navigation_then_extraction",
    "multi_step_information_retrieval",
    "negative_or_ambiguous_case",
]

ExpectedOutputType = Literal["scalar", "object", "list", "mixed", "none"]
AnchorMatchingMode = Literal["auto", "exact", "contains"]


class BenchmarkScenario(BaseModel):
    scenario_id: str
    goal: str
    start_url: str
    target_page_hint: str = ""
    anchor_candidates: list[str] = Field(default_factory=list)
    anchor_matching_mode: AnchorMatchingMode = "auto"
    expected_navigation: list[str] = Field(default_factory=list)
    task_family: ScenarioCategory | None = None
    preconditions: list[str] = Field(default_factory=list)
    page_expectations: list[str] = Field(default_factory=list)
    category: ScenarioCategory
    description: str
    expected_output_type: ExpectedOutputType
    required_top_level_fields: list[str] = Field(default_factory=list)
    expected_min_items: int = 0
    expected_item_fields: list[str] = Field(default_factory=list)
    should_succeed: bool = True
    notes: str = ""


class ScenarioSuite(BaseModel):
    suite_id: str
    description: str
    scenarios: list[BenchmarkScenario]


def load_scenario_suite(path: Path) -> ScenarioSuite:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScenarioSuite.model_validate(payload)
