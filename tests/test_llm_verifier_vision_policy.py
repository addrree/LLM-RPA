from app.schemas.execution import ExecutionResult, GenerationMetadata, LLMArtifact, StepLog
from app.schemas.task_spec import TaskSpec
from app.utils.llm_client import LLMClientError
from app.verifier.llm_verifier import LLMVerifier


class _CaptureLLMClient:
    def __init__(self):
        self.image_paths: list[str | None] = []

    def generate_verifier_artifact(self, system_prompt, user_prompt, image_path=None, *, stage="verifier"):
        self.image_paths.append(image_path)
        return LLMArtifact(
            raw_response='{"task_completed":true,"confidence":0.9,"verdict":"accept","issues":[],"summary":"ok"}',
            parsed_response={
                "task_completed": True,
                "confidence": 0.9,
                "verdict": "accept",
                "issues": [],
                "summary": "ok",
            },
            generation=GenerationMetadata(
                backend="dummy",
                model="dummy",
                source="dummy",
                fallback_used=False,
            ),
        )


class _BrokenVerifierClient:
    def generate_verifier_artifact(self, system_prompt, user_prompt, image_path=None, *, stage="verifier"):
        raise LLMClientError("Failed to parse JSON response from LLM at stage=verifier.")


class _FailIfCalledClient:
    def generate_verifier_artifact(self, system_prompt, user_prompt, image_path=None, *, stage="verifier"):
        raise AssertionError("Verifier LLM should not be called for deterministic metadata fast path.")


def _plan(action: str = "extract_text", goal: str = "Extract answer") -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": goal,
            "start_url": "https://example.org",
            "allowed_domains": ["example.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Answer", "required_fields": ["answer"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
                {"step_id": 2, "action": action, "args": {"selector": "main"} if action == "extract_text" else {}, "save_as": "answer"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )


def _result() -> ExecutionResult:
    return ExecutionResult(
        status="success",
        extracted_data={"answer": "ok"},
        final_url="https://example.org",
        page_title="Example",
        page_text_excerpt="answer ok",
        screenshot_path="artifacts/screenshots/example.png",
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )


def test_verifier_auto_mode_uses_text_only_for_non_visual_tasks(monkeypatch):
    monkeypatch.delenv("OLLAMA_VERIFIER_VISION_MODE", raising=False)
    client = _CaptureLLMClient()

    verdict = LLMVerifier(client).verify(_plan(), _result())

    assert verdict.verdict == "accept"
    assert client.image_paths == [None]


def test_verifier_auto_mode_sends_image_for_visual_tasks(monkeypatch):
    monkeypatch.delenv("OLLAMA_VERIFIER_VISION_MODE", raising=False)
    client = _CaptureLLMClient()

    verdict = LLMVerifier(client).verify(_plan(action="visual_observe", goal="Visually inspect page"), _result())

    assert verdict.verdict == "accept"
    assert client.image_paths == ["artifacts/screenshots/example.png"]


def test_verifier_vision_mode_env_overrides_auto(monkeypatch):
    client = _CaptureLLMClient()
    monkeypatch.setenv("OLLAMA_VERIFIER_VISION_MODE", "always")
    LLMVerifier(client).verify(_plan(), _result())

    monkeypatch.setenv("OLLAMA_VERIFIER_VISION_MODE", "never")
    LLMVerifier(client).verify(_plan(action="visual_observe", goal="Visually inspect page"), _result())

    assert client.image_paths == ["artifacts/screenshots/example.png", None]


def test_verifier_parse_error_returns_uncertain_verdict_without_exception():
    verdict = LLMVerifier(_BrokenVerifierClient()).verify(_plan(), _result())

    assert verdict.verdict == "uncertain"
    assert verdict.task_completed is False
    assert "could not be parsed" in verdict.issues[0]


def test_verifier_accepts_populated_page_metadata_without_llm():
    plan = TaskSpec.model_validate(
        {
            "goal": "Open Wikipedia English and return current URL and page title.",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["www.wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Page metadata", "required_fields": ["final_url", "page_title"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"step_id": 2, "action": "click_by_semantic_target", "args": {"target": "English"}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    result = ExecutionResult(
        status="success",
        extracted_data={},
        final_url="https://en.wikipedia.org/wiki/Main_Page",
        page_title="Wikipedia, the free encyclopedia",
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(plan, result)

    assert verdict.verdict == "accept"
