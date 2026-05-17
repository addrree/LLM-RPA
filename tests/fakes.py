import json
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from app.schemas.execution import GenerationMetadata, LLMArtifact
from app.utils.llm_client import LLMClient


class DummyLLMClient(LLMClient):
    def __init__(self):
        self.backend = "dummy"
        self.planner_model = "dummy-template"
        self.verifier_model = "dummy-template"

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        lower_prompt = system_prompt.lower()
        if "модуль верификации" in lower_prompt or "verification" in lower_prompt:
            return self._build_dummy_verdict(user_prompt)
        if "initial planner" in lower_prompt or "двухэтапного режима" in lower_prompt:
            return self._build_dummy_initial_plan(user_prompt)
        if "context-aware replanner" in lower_prompt or "replanner" in lower_prompt:
            return self._build_dummy_replan(user_prompt)
        return self._build_dummy_plan(user_prompt)

    def _build_dummy_initial_plan(self, user_goal: str) -> Dict[str, Any]:
        target_url = self._extract_first_url(user_goal) or "https://www.wikipedia.org"
        domain = urlparse(target_url).netloc or "www.wikipedia.org"
        return {
            "goal": user_goal,
            "start_url": target_url,
            "allowed_domains": [domain],
            "constraints": {"max_steps": 4, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Observe page", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": target_url}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }

    def _build_dummy_replan(self, user_prompt: str) -> Dict[str, Any]:
        try:
            payload = json.loads(user_prompt)
        except json.JSONDecodeError:
            payload = {"user_goal": user_prompt, "page_snapshot": {}}

        user_goal = payload.get("user_goal", "Extract value")
        page_snapshot = payload.get("page_snapshot", {}) or {}
        url = page_snapshot.get("url") or self._extract_first_url(user_goal) or "https://www.wikipedia.org"
        domain = urlparse(url).netloc or "www.wikipedia.org"

        return {
            "goal": user_goal,
            "start_url": url,
            "allowed_domains": [domain],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {
                "description": "Extract value using observed page text",
                "required_fields": ["ru_articles_count"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": url}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "step_id": 3,
                    "action": "extract_pattern_from_page_text",
                    "args": {
                        "pattern": "Русский\\s*[—-]\\s*([0-9\\s,\\.]+)",
                        "flags": "IGNORECASE",
                        "occurrence": 1,
                    },
                    "save_as": "ru_articles_count",
                },
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }

    def _build_dummy_plan(self, user_goal: str) -> Dict[str, Any]:
        target_url = self._extract_first_url(user_goal) or "https://www.wikipedia.org"
        domain = urlparse(target_url).netloc or "www.wikipedia.org"

        return {
            "goal": user_goal,
            "start_url": target_url,
            "allowed_domains": [domain],
            "constraints": {
                "max_steps": 6,
                "max_replans": 1,
                "timeout_sec": 20,
            },
            "expected_result": {
                "description": f"Extract the page heading from {domain}",
                "required_fields": ["heading"],
            },
            "steps": [
                {
                    "step_id": 1,
                    "action": "open_url",
                    "args": {"url": target_url},
                },
                {
                    "step_id": 2,
                    "action": "extract_text",
                    "args": {"selector": "h1"},
                    "save_as": "heading",
                },
                {
                    "step_id": 3,
                    "action": "screenshot",
                    "args": {},
                },
                {
                    "step_id": 4,
                    "action": "finish",
                    "args": {},
                },
            ],
        }

    def _build_dummy_verdict(self, verification_package: str) -> Dict[str, Any]:
        try:
            payload = json.loads(verification_package)
        except json.JSONDecodeError:
            return {
                "task_completed": False,
                "confidence": 0.0,
                "verdict": "reject",
                "issues": ["Verifier input is not a valid JSON package."],
                "summary": "Cannot verify results because verification payload is invalid.",
            }

        required_fields = payload.get("required_fields", []) or []
        extracted_data = payload.get("extracted_data", {}) or {}
        logs = payload.get("logs", []) or []

        missing_fields = [field for field in required_fields if not extracted_data.get(field)]
        has_failed_step = any(log.get("status") == "failed" for log in logs if isinstance(log, dict))

        if has_failed_step:
            return {
                "task_completed": False,
                "confidence": 0.15,
                "verdict": "reject",
                "issues": ["Execution contains failed steps."],
                "summary": "Execution failed before completing required actions.",
            }

        if missing_fields:
            return {
                "task_completed": False,
                "confidence": 0.3,
                "verdict": "uncertain",
                "issues": [f"Missing required fields: {', '.join(missing_fields)}"],
                "summary": "Execution finished but required data is incomplete.",
            }

        return {
            "task_completed": True,
            "confidence": 0.9,
            "verdict": "accept",
            "issues": [],
            "summary": "The extracted data contains all required fields.",
        }

    @staticmethod
    def _extract_first_url(user_goal: str) -> Optional[str]:
        match = re.search(r"https?://[^\s\"'<>]+", user_goal)
        if not match:
            return None
        return match.group(0).rstrip(".,)")

    def generate_planner_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_verifier_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        *,
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_planner_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "planner",
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> LLMArtifact:
        parsed = self.generate_json(system_prompt, user_prompt)
        return LLMArtifact(
            raw_response=json.dumps(parsed, ensure_ascii=False),
            parsed_response=parsed,
            generation=GenerationMetadata(
                backend=self.backend,
                model=self.planner_model,
                source="dummy",
                fallback_used=False,
            ),
        )

    def generate_verifier_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        *,
        stage: str = "verifier",
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> LLMArtifact:
        parsed = self.generate_json(system_prompt, user_prompt)
        return LLMArtifact(
            raw_response=json.dumps(parsed, ensure_ascii=False),
            parsed_response=parsed,
            generation=GenerationMetadata(
                backend=self.backend,
                model=self.verifier_model,
                source="dummy",
                fallback_used=False,
            ),
        )
