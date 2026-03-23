import json
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        planner_model: str = "gemini-2.0-flash",
        verifier_model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMClientError("GEMINI_API_KEY is not set.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMClientError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

        self._types = types
        self.client = genai.Client(api_key=self.api_key)
        self.planner_model = planner_model
        self.verifier_model = verifier_model
        self.temperature = temperature

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self._generate_json_with_model(
            model=self.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self._generate_json_with_model(
            model=self.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def generate_verifier_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self._generate_json_with_model(
            model=self.verifier_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _generate_json_with_model(self, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            system_instruction=system_prompt,
        )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:
            raise LLMClientError(f"Gemini API request failed: {exc}") from exc

        text = self._extract_response_text(response)
        return self._safe_parse_json(text)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    candidates.append(part_text)

        if candidates:
            return "\n".join(candidates).strip()

        raise LLMClientError("Gemini API returned empty response.")

    @staticmethod
    def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"Failed to parse JSON response. Raw response: {raw_text[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMClientError("JSON response must be an object.")

        return data


class DummyLLMClient(LLMClient):
    def __init__(self):
        pass

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if "модуль верификации" in system_prompt.lower() or "verification" in system_prompt.lower():
            return self._build_dummy_verdict(user_prompt)
        return self._build_dummy_plan(user_prompt)

    def _build_dummy_plan(self, user_goal: str) -> Dict[str, Any]:
        target_url = self._extract_first_url(user_goal) or "https://example.com"
        domain = urlparse(target_url).netloc or "example.com"

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

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_verifier_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)
