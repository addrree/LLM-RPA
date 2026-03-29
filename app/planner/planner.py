import json
import re

from app.planner.prompts import INITIAL_PLANNER_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from app.schemas.execution import LLMArtifact
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClient


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.last_artifact: LLMArtifact | None = None
        self.last_initial_artifact: LLMArtifact | None = None

    def build_plan(self, user_goal: str) -> TaskSpec:
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_goal,
        )
        self.last_artifact = artifact
        return TaskSpec.model_validate(artifact.parsed_response)

    def build_initial_plan(self, user_goal: str) -> TaskSpec:
        artifact = self.llm_client.generate_planner_artifact(
            system_prompt=INITIAL_PLANNER_SYSTEM_PROMPT,
            user_prompt=user_goal,
        )
        self.last_initial_artifact = artifact

        parsed = artifact.parsed_response
        # Fail-safe for non-compliant outputs.
        if not self._is_valid_initial_shape(parsed):
            fallback = self._build_initial_fallback(user_goal)
            self.last_initial_artifact = LLMArtifact(
                raw_response=json.dumps(fallback, ensure_ascii=False),
                parsed_response=fallback,
                generation=artifact.generation,
            )
            parsed = fallback
        return TaskSpec.model_validate(parsed)

    @staticmethod
    def _is_valid_initial_shape(payload: dict) -> bool:
        try:
            steps = payload.get("steps", [])
            actions = [s.get("action") for s in steps]
            return actions == ["open_url", "observe_page", "finish"]
        except Exception:
            return False

    @staticmethod
    def _build_initial_fallback(user_goal: str) -> dict:
        match = re.search(r"https?://[^\s\"'<>]+", user_goal)
        url = (match.group(0).rstrip(".,)") if match else "https://www.wikipedia.org")
        domain = re.sub(r"^https?://", "", url).split("/")[0]
        return {
            "goal": user_goal,
            "start_url": url,
            "allowed_domains": [domain],
            "constraints": {"max_steps": 4, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {
                "description": "Observe landing page context",
                "required_fields": ["page_snapshot"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": url}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
