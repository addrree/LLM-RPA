from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

WEBARENA_REQUIRED_ENV_VARS = [
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
]


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


def validate_webarena_env_vars(env_id: str) -> dict:
    normalized = (env_id or "").lower()
    if "webarena" not in normalized:
        return {"ok": True, "missing": [], "message": ""}

    missing = [name for name in WEBARENA_REQUIRED_ENV_VARS if not os.getenv(name)]
    if not missing:
        return {"ok": True, "missing": [], "message": ""}

    message = (
        "BrowserGym WebArena requires self-hosted WebArena URLs via WA_* env vars. "
        "Run openended smoke first or configure WebArena server."
    )
    return {"ok": False, "missing": missing, "message": message}
