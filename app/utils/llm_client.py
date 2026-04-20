import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from app.schemas.execution import GenerationMetadata, LLMArtifact


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
        timeout_sec: Optional[int] = None,
    ):
        self.backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
        if self.backend not in {"ollama", "ollama_cloud"}:
            raise LLMClientError(
                f"Unsupported backend '{self.backend}'. Supported backends: 'ollama', 'ollama_cloud', 'dummy'."
            )

        default_base_url = "https://ollama.com" if self.backend == "ollama_cloud" else "http://localhost:11434"
        self.ollama_base_url = (ollama_base_url or os.getenv("OLLAMA_BASE_URL", default_base_url)).rstrip("/")
        default_model = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")
        self.planner_model = planner_model or os.getenv("OLLAMA_PLANNER_MODEL", default_model)
        self.verifier_model = verifier_model or os.getenv("OLLAMA_VERIFIER_MODEL", default_model)
        self.temperature = temperature
        self.timeout_sec = timeout_sec if timeout_sec is not None else self._resolve_timeout_sec()
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "").strip()

        self.session = requests.Session()
        self.session.trust_env = False
        self.last_chat_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def _resolve_timeout_sec() -> int:
        raw_timeout = os.getenv("OLLAMA_TIMEOUT_SEC", "300")
        try:
            return int(raw_timeout)
        except ValueError as exc:
            raise LLMClientError(
                f"Invalid OLLAMA_TIMEOUT_SEC='{raw_timeout}'. Expected integer seconds."
            ) from exc

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_planner_json(system_prompt, user_prompt)

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_planner_artifact(system_prompt, user_prompt).parsed_response

    def generate_planner_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "planner",
    ) -> LLMArtifact:
        raw_text = self._ollama_chat(
            model=self.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
        )
        parsed = self._safe_parse_json(raw_text, stage=stage)
        fallback_used = bool(self.last_chat_diagnostics.get("used_thinking_fallback", False))
        return LLMArtifact(
            raw_response=raw_text,
            parsed_response=parsed,
            generation=GenerationMetadata(
                backend=self.backend,
                model=self.planner_model,
                source="llm",
                fallback_used=fallback_used,
            ),
        )

    def generate_verifier_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_verifier_artifact(system_prompt, user_prompt, image_path=image_path).parsed_response

    def generate_verifier_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        *,
        stage: str = "verifier",
    ) -> LLMArtifact:
        raw_text = self._ollama_chat(
            model=self.verifier_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
        )
        parsed = self._safe_parse_json(raw_text, stage=stage)
        fallback_used = bool(self.last_chat_diagnostics.get("used_thinking_fallback", False))
        return LLMArtifact(
            raw_response=raw_text,
            parsed_response=parsed,
            generation=GenerationMetadata(
                backend=self.backend,
                model=self.verifier_model,
                source="llm",
                fallback_used=fallback_used,
            ),
        )

    def _ollama_chat(self, model: str, system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        headers: Dict[str, str] = {}
        if self.backend == "ollama_cloud":
            if not self.ollama_api_key:
                raise LLMClientError(
                    "OLLAMA_API_KEY is required for backend=ollama_cloud. "
                    "Set OLLAMA_API_KEY and retry."
                )
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

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
            response = self.session.post(url, json=payload, headers=headers or None, timeout=self.timeout_sec)
            if not response.ok:
                details = response.text[:800]
                if response.status_code in {401, 403}:
                    raise LLMClientError(
                        f"Ollama Cloud authentication failed (status_code={response.status_code}, url={url}). "
                        "Check OLLAMA_API_KEY."
                    )
                if response.status_code == 404:
                    raise LLMClientError(
                        f"Ollama model not found (status_code=404, url={url}, model={model}). "
                        "Verify OLLAMA_PLANNER_MODEL / OLLAMA_VERIFIER_MODEL / OLLAMA_MODEL."
                    )
                raise LLMClientError(
                    "Ollama request failed "
                    f"(status_code={response.status_code}, url={url}, model={model}). "
                    f"Response: {details}"
                )
        except requests.Timeout as exc:
            backend_hint = "Ollama Cloud" if self.backend == "ollama_cloud" else "Ollama local"
            raise LLMClientError(
                f"{backend_hint} request timed out after {self.timeout_sec}s "
                f"(url={url}, model={model}). Increase OLLAMA_TIMEOUT_SEC or simplify the prompt."
            ) from exc
        except requests.RequestException as exc:
            raise LLMClientError(
                f"Ollama request failed (backend={self.backend}, url={url}, model={model}): {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError(f"Ollama returned non-JSON response: {response.text[:300]}") from exc

        message = data.get("message") or {}
        message_content_raw = str(message.get("content") or "")
        response_content_raw = str(data.get("response") or "")
        message_content = message_content_raw.strip()
        thinking_content = str(message.get("thinking") or "").strip()
        content_source = "message.content"
        cleaned_content = str(message_content).strip()
        used_thinking_fallback = False
        if not cleaned_content and response_content_raw.strip():
            cleaned_content = response_content_raw.strip()
            content_source = "response"
        if not cleaned_content and thinking_content:
            extracted_thinking = self._sanitize_llm_json_text(thinking_content)
            if extracted_thinking:
                cleaned_content = extracted_thinking
                content_source = "message.thinking"
                used_thinking_fallback = True

        self.last_chat_diagnostics = {
            "content_source": content_source,
            "used_thinking_fallback": used_thinking_fallback,
            "response_keys": sorted(list(data.keys())),
        }
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
    def _safe_parse_json(raw_text: str, *, stage: str = "unknown_stage") -> Dict[str, Any]:
        cleaned = LLMClient._sanitize_llm_json_text(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            snippet_start = max(0, exc.pos - 60)
            snippet_end = min(len(cleaned), exc.pos + 60)
            snippet = cleaned[snippet_start:snippet_end]
            raise LLMClientError(
                f"Failed to parse JSON response from LLM at stage={stage}. "
                f"line={exc.lineno}, col={exc.colno}, pos={exc.pos}. "
                f"Context snippet: {snippet!r}. "
                f"Raw response (first 500 chars): {raw_text[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMClientError("JSON response must be an object.")

        return data

    @staticmethod
    def _sanitize_llm_json_text(raw_text: str) -> str:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
        candidate = LLMClient._extract_first_json_block(cleaned)
        return candidate.strip() if candidate else cleaned

    @staticmethod
    def _extract_first_json_block(text: str) -> str | None:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == "\"":
                        in_string = False
                    continue
                if char == "\"":
                    in_string = True
                    continue
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start: idx + 1]
        return None


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

    def generate_planner_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_verifier_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_planner_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "planner",
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
