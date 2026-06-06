import asyncio
import re

import pytest

from app.executor.action_handlers import ActionHandlers, StructuredExtractionError
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec
from app.verifier.llm_verifier import LLMVerifier


class _FailIfCalledClient:
    def generate_verifier_artifact(self, *args, **kwargs):
        raise AssertionError("Verifier LLM should not be called for deterministic rejection.")


class _TextBlockPage:
    url = "https://docs.sample.test/reference"

    def __init__(self, html: str, title: str = "Reference"):
        self.html = html
        self._title = title

    async def title(self):
        return self._title

    async def evaluate(self, _script, payload=None):
        requested = list((payload or {}).get("requested") or [])
        main_match = re.search(r"<main[^>]*>(.*?)</main>", self.html, flags=re.IGNORECASE | re.DOTALL)
        main_html = main_match.group(1) if main_match else self.html
        heading_match = re.search(r"<h[12][^>]*>(.*?)</h[12]>", main_html, flags=re.IGNORECASE | re.DOTALL)
        paragraph_match = re.search(r"<p[^>]*>(.*?)</p>", main_html, flags=re.IGNORECASE | re.DOTALL)
        strip_tags = lambda value: re.sub(r"<[^>]+>", "", value or "").strip()
        title = strip_tags(heading_match.group(1) if heading_match else self._title)
        description = " ".join(strip_tags(paragraph_match.group(1) if paragraph_match else "").split())
        first_sentence = description.split(". ", 1)[0] + "." if ". " in description else description
        out = {
            "title": title,
            "heading": title,
            "page_title": self._title,
            "description": description,
            "summary": description,
            "snippet": description,
            "first_sentence": first_sentence,
            "paragraph": description,
            "current_url": self.url,
            "final_url": self.url,
        }
        projected = {}
        for field in requested:
            lower = str(field).casefold()
            if "title" in lower or "heading" in lower:
                projected[field] = title
            elif "sentence" in lower:
                projected[field] = first_sentence
            elif "description" in lower or "summary" in lower or "paragraph" in lower:
                projected[field] = description
            elif "url" in lower:
                projected[field] = self.url
        return {**out, "projected": projected}


async def _not_blocked(*_args, **_kwargs):
    return None


def _description_plan() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "Extract the page description.",
            "start_url": "https://docs.sample.test/reference",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Description", "required_fields": ["description"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://docs.sample.test/reference"}},
                {
                    "step_id": 2,
                    "action": "extract_by_intent",
                    "args": {"intent": "text_block", "fields": {"description": {"type": "description"}}},
                    "save_as": "description",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )


def test_text_block_extraction_uses_main_heading_and_first_meaningful_paragraph():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    page = _TextBlockPage(
        """
        <aside><p>Navigation teaser that should not become the description.</p></aside>
        <main>
          <h1>WebSocket API</h1>
          <p>The WebSocket API makes it possible to open a two-way interactive communication session.</p>
        </main>
        """,
        title="WebSocket API",
    )

    result = asyncio.run(
        handler.extract_by_intent(
            page,
            {
                "intent": "text_block",
                "fields": {
                    "heading": {"type": "page_title"},
                    "description": {"type": "description"},
                    "first_sentence": {"type": "description"},
                },
            },
            {},
        )
    )

    assert result["heading"] == "WebSocket API"
    assert result["description"].startswith("The WebSocket API makes")
    assert "Navigation teaser" not in result["description"]
    assert result["status"] == "success"


def test_text_block_description_missing_is_controlled_failure():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    page = _TextBlockPage("<main><h1>Empty Reference</h1></main>", title="Empty Reference")

    with pytest.raises(StructuredExtractionError) as exc_info:
        asyncio.run(
            handler.extract_by_intent(
                page,
                {"intent": "text_block", "fields": {"description": {"type": "description"}}},
                {},
            )
        )

    assert exc_info.value.code == "description_not_found"


def test_text_block_source_text_fallback_finds_first_readable_paragraph():
    result = ActionHandlers._text_block_from_source_text(
        source_text="""
        Skip to main content
        Web APIs
        WebSocket API
        The WebSocket API makes it possible to open a two-way interactive communication session between a browser and a server.
        In this article
        Interfaces
        """,
        title="WebSocket API",
    )

    assert result["title"] == "WebSocket API"
    assert result["description"].startswith("The WebSocket API makes")
    assert result["first_sentence"].endswith("server.")


def test_verifier_rejects_page_snapshot_only_for_description_goal():
    result = ExecutionResult(
        status="success",
        extracted_data={"page_snapshot": {"url": "https://docs.sample.test/reference", "title": "Reference"}},
        logs=[StepLog(step_id=2, action="observe_page", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_description_plan(), result)

    assert verdict.verdict == "reject"
    assert "page_snapshot alone" in verdict.issues[0]
