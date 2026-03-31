import json
import re
from urllib.parse import urlparse

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
        normalized = self._normalize_initial_plan(parsed, user_goal)
        return TaskSpec.model_validate(normalized)

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

    @staticmethod
    def _normalize_initial_plan(raw_plan: dict, user_goal: str) -> dict:
        plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}

        steps = plan.get("steps")
        if not isinstance(steps, list):
            steps = []

        normalized_steps: list[dict] = []
        has_observe_page = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            current = dict(step)
            args = current.get("args")
            if not isinstance(args, dict):
                args = {}
            current["args"] = dict(args)

            if current.get("action") == "open_url" and "url" not in current["args"] and "url" in current:
                current["args"]["url"] = current["url"]
                current.pop("url", None)
            if current.get("action") == "observe_page":
                has_observe_page = True
                save_as = current.get("save_as")
                if not isinstance(save_as, str) or not save_as.strip():
                    current["save_as"] = "page_snapshot"

            normalized_steps.append(current)

        has_finish = any(step.get("action") == "finish" for step in normalized_steps)
        if not has_finish:
            normalized_steps.append({"action": "finish", "args": {}})

        for idx, step in enumerate(normalized_steps, start=1):
            if not isinstance(step.get("step_id"), int):
                step["step_id"] = idx

        expected_result = plan.get("expected_result")
        if not isinstance(expected_result, dict):
            expected_result = {}
        if not expected_result.get("description"):
            expected_result["description"] = "Collect page snapshot for replanning"
        if not isinstance(expected_result.get("required_fields"), list):
            expected_result["required_fields"] = ["page_snapshot"]
        elif has_observe_page and "page_snapshot" not in expected_result["required_fields"]:
            expected_result["required_fields"] = [*expected_result["required_fields"], "page_snapshot"]

        start_url = plan.get("start_url")
        if not start_url:
            for step in normalized_steps:
                if step.get("action") == "open_url":
                    candidate_url = step.get("args", {}).get("url")
                    if candidate_url:
                        start_url = candidate_url
                        break

        if not start_url:
            start_url = "https://www.wikipedia.org"

        allowed_domains = plan.get("allowed_domains")
        if not isinstance(allowed_domains, list) or not allowed_domains:
            netloc = urlparse(str(start_url)).netloc
            allowed_domains = [netloc] if netloc else []

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {"max_steps": 4, "max_replans": 1, "timeout_sec": 30}
        else:
            constraints = {
                "max_steps": constraints.get("max_steps", 4),
                "max_replans": constraints.get("max_replans", 1),
                "timeout_sec": constraints.get("timeout_sec", 30),
            }

        return {
            "goal": plan.get("goal") or user_goal,
            "start_url": start_url,
            "allowed_domains": allowed_domains,
            "constraints": constraints,
            "expected_result": expected_result,
            "steps": normalized_steps,
        }
