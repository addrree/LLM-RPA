import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

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
                f"Unsupported backend '{self.backend}'. Supported backends: 'ollama', 'ollama_cloud'."
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
        self.last_prompt_chars: int = 0

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

    def generate_planner_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_planner_artifact(
            system_prompt,
            user_prompt,
            images_base64=images_base64,
            image_base64=image_base64,
        ).parsed_response

    def generate_planner_artifact(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "planner",
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> LLMArtifact:
        try:
            raw_text = self._ollama_chat(
                model=self.planner_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                images_base64=self._normalize_images_base64(images_base64, image_base64),
            )
            parsed = self._safe_parse_json(raw_text, stage=stage, diagnostics=self.last_chat_diagnostics)
        except LLMClientError:
            logger.exception(
                "Planner generation failed at stage=%s (system_prompt_len=%d, user_prompt_len=%d, response_mode=%s)",
                stage,
                len(system_prompt or ""),
                len(user_prompt or ""),
                self.last_chat_diagnostics.get("content_source", "unknown"),
            )
            raise
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
        *,
        images_base64: Optional[list[str]] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_verifier_artifact(
            system_prompt,
            user_prompt,
            image_path=image_path,
            images_base64=images_base64,
            image_base64=image_base64,
        ).parsed_response

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
        raw_text = self._ollama_chat(
            model=self.verifier_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            images_base64=self._normalize_images_base64(images_base64, image_base64),
        )
        parsed = self._safe_parse_json(raw_text, stage=stage, diagnostics=self.last_chat_diagnostics)
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

    def _ollama_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        images_base64: Optional[list[str]] = None,
    ) -> str:
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
        message_images: list[str] = []
        if image_path:
            message_images.append(self._encode_image_base64(image_path))
        if images_base64:
            message_images.extend(str(image) for image in images_base64 if image is not None)
        if message_images:
            user_message["images"] = message_images

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

        last_error: LLMClientError | None = None
        retry_used = False
        retry_reason: str | None = None
        self.last_prompt_chars = len(system_prompt or "") + len(user_prompt or "")
        for attempt in range(2):
            try:
                response = self.session.post(url, json=payload, headers=headers or None, timeout=self.timeout_sec)
            except requests.Timeout as exc:
                backend_hint = "Ollama Cloud" if self.backend == "ollama_cloud" else "Ollama local"
                last_error = LLMClientError(
                    f"{backend_hint} request timed out after {self.timeout_sec}s "
                    f"(url={url}, model={model}). Increase OLLAMA_TIMEOUT_SEC or simplify the prompt."
                )
                if attempt == 0:
                    retry_used = True
                    retry_reason = "timeout"
                    logger.warning("LLM transport retry scheduled due to timeout (model=%s, url=%s)", model, url)
                    time.sleep(0.5)
                    continue
                raise last_error from exc
            except requests.RequestException as exc:
                raise LLMClientError(
                    f"Ollama request failed (backend={self.backend}, url={url}, model={model}): {exc}"
                ) from exc

            if not response.ok:
                details = response.text[:800]
                if response.status_code in {401, 403}:
                    if response.status_code == 403 and "requires a subscription" in details.lower():
                        raise LLMClientError(
                            "Ollama Cloud model access denied: model requires subscription/access. "
                            f"Check model name or subscription. (status_code=403, url={url}, model={model})"
                        )
                    raise LLMClientError(
                        f"Ollama Cloud authentication failed (status_code={response.status_code}, url={url}, model={model}). "
                        "Check OLLAMA_API_KEY or model access."
                    )
                if response.status_code == 404:
                    raise LLMClientError(
                        f"Ollama model not found (status_code=404, url={url}, model={model}). "
                        "Verify OLLAMA_PLANNER_MODEL / OLLAMA_VERIFIER_MODEL / OLLAMA_MODEL."
                    )
                if response.status_code in {500, 502} and attempt == 0:
                    retry_used = True
                    retry_reason = f"http_{response.status_code}"
                    logger.warning(
                        "LLM transport retry scheduled due to transient status (status=%s, model=%s, url=%s)",
                        response.status_code,
                        model,
                        url,
                    )
                    time.sleep(0.5)
                    continue
                raise LLMClientError(
                    "Ollama request failed "
                    f"(status_code={response.status_code}, url={url}, model={model}). "
                    f"Response: {details}"
                )

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
                "transport_retry_used": retry_used,
                "transport_retry_reason": retry_reason,
                "transport_attempt_count": attempt + 1,
                "prompt_chars": self.last_prompt_chars,
            }
            if cleaned_content:
                return cleaned_content

            last_error = LLMClientError(f"Ollama returned empty content. Full payload: {data}")
            if attempt == 0:
                retry_used = True
                retry_reason = "empty_content"
                logger.warning("LLM transport retry scheduled due to empty content (model=%s, url=%s)", model, url)
                time.sleep(0.5)
                continue
            raise last_error

        raise last_error or LLMClientError("Ollama request failed without diagnostics.")


    @staticmethod
    def _normalize_images_base64(images_base64: Optional[list[str]] = None, image_base64: Optional[str] = None) -> list[str] | None:
        images: list[str] = []
        if image_base64 is not None:
            images.append(image_base64)
        if images_base64 is not None:
            images.extend(str(image) for image in images_base64 if image is not None)
        return images or None

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
    def _safe_parse_json(
        raw_text: str,
        *,
        stage: str = "unknown_stage",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cleaned = LLMClient._sanitize_llm_json_text(raw_text)
        if diagnostics is not None:
            diagnostics["json_escape_repair_applied"] = False
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if LLMClient._is_invalid_json_escape_error(exc):
                repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    LLMClient._raise_json_parse_error(raw_text, cleaned, stage, exc)
                else:
                    if diagnostics is not None:
                        diagnostics["json_escape_repair_applied"] = True
                    logger.warning(
                        "Applied JSON invalid-escape repair to LLM response at stage=%s",
                        stage,
                    )
            else:
                LLMClient._raise_json_parse_error(raw_text, cleaned, stage, exc)

        if not isinstance(data, dict):
            raise LLMClientError("JSON response must be an object.")

        return data

    @staticmethod
    def _is_invalid_json_escape_error(exc: json.JSONDecodeError) -> bool:
        return "Invalid \\escape" in exc.msg

    @staticmethod
    def _raise_json_parse_error(
        raw_text: str,
        cleaned: str,
        stage: str,
        exc: json.JSONDecodeError,
    ) -> None:
        snippet_start = max(0, exc.pos - 60)
        snippet_end = min(len(cleaned), exc.pos + 60)
        snippet = cleaned[snippet_start:snippet_end]
        raise LLMClientError(
            f"Failed to parse JSON response from LLM at stage={stage}. "
            f"line={exc.lineno}, col={exc.colno}, pos={exc.pos}. "
            f"Context snippet: {snippet!r}. "
            f"Raw response (first 500 chars): {raw_text[:500]}"
        ) from exc

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
