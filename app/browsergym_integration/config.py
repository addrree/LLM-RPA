from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BrowserGymRunConfig:
    env_id: str
    goal: str | None = None
    max_steps: int = 15
    backend: str = "ollama_cloud"
    two_stage_planning: bool = True
    headless: bool = False
    save_artifacts: bool = True
    output_dir: Path = field(default_factory=lambda: Path("artifacts/browsergym"))
    use_llm_verifier: bool = True
    stop_on_agent_finish: bool = True
    use_vision: bool = False
    action_mode: Literal["browsergym_text", "taskspec_step"] = "browsergym_text"
    task_kwargs: dict | None = None
    benchmark: str | None = None
    task_name: str | None = None
    allow_playwright_fallback: bool = False
