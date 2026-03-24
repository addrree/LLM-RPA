import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        planner_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        temperature: float = 0.1,
        timeout_sec: int = 60,
    ):
        self.backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
        if self.backend != "ollama":
            raise LLMClientError(
                f"Unsupported backend '{self.backend}'. Only 'ollama' and DummyLLMClient are available."
            )

        self.ollama_base_url = (ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        default_model = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")
        self.planner_model = planner_model or os.getenv("OLLAMA_PLANNER_MODEL", default_model)
        self.verifier_model = verifier_model or os.getenv("OLLAMA_VERIFIER_MODEL", default_model)
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_planner_json(system_prompt, user_prompt)

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw_text = self._ollama_chat(
            model=self.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
        )
        return self._safe_parse_json(raw_text)

    def generate_verifier_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw_text = self._ollama_chat(
            model=self.verifier_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
        )
        return self._safe_parse_json(raw_text)

    def _ollama_chat(self, model: str, system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
        url = f"{self.ollama_base_url}/api/chat"

        user_message: Dict[str, Any] = {"role": "user", "content": user_prompt}
        if image_path:
            user_message["images"] = [self._encode_image_base64(image_path)]

        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_sec)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError(f"Ollama returned non-JSON response: {response.text[:300]}") from exc

        message_content = (
            (data.get("message") or {}).get("content")
            or data.get("response")
            or ""
        )
        cleaned_content = str(message_content).strip()
        if not cleaned_content:
            raise LLMClientError(f"Ollama returned empty content. Full payload: {data}")

        return cleaned_content

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        candidate = Path(image_path).expanduser()
        if not candidate.is_file():
            raise LLMClientError(f"Screenshot does not exist or is not a file: {candidate}")

        try:
            image_bytes = candidate.read_bytes()
        except OSError as exc:
            raise LLMClientError(f"Failed to read screenshot '{candidate}': {exc}") from exc

        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                "Failed to parse JSON response from LLM. "
                f"Raw response (first 500 chars): {raw_text[:500]}"
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

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_verifier_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)
