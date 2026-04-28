import asyncio

from app.executor.action_handlers import ActionHandlers, StructuredExtractionError


class _NeverVisibleLocator:
    first = None

    def __init__(self):
        self.first = self

    async def count(self):
        return 0

    async def wait_for(self, **_kwargs):
        raise TimeoutError("not visible")

    async def inner_text(self):
        raise TimeoutError("not visible")


class _FallbackPage:
    def __init__(self):
        self._locator = _NeverVisibleLocator()

    def locator(self, _selector):
        return self._locator

    async def evaluate(self, _script):
        return {"value": "СПбПУ Петра Великого", "fallback": "meta_og_title"}

    async def title(self):
        return "SPbSTU"


def test_extract_text_fallback_without_h1():
    handlers = ActionHandlers()
    page = _FallbackPage()
    args = {"selector": "h1"}
    value = asyncio.run(handlers.extract_text(page=page, args=args, runtime_state={}))
    assert value == "СПбПУ Петра Великого"
    assert "fallback=meta_og_title" in args.get("_executor_note", "")


def test_extract_section_lines_empty_is_not_success():
    handlers = ActionHandlers()
    args = {"heading_text": "Statuses", "limit": 3}
    runtime_state = {
        "last_page_snapshot": {
            "visible_headings": ["Statuses"],
            "page_text": "Statuses",
            "url": "https://example.org",
        }
    }

    async def _fake_source_text(**_kwargs):
        return "Statuses\n\n\n"

    handlers._load_source_text = _fake_source_text  # type: ignore[method-assign]

    try:
        asyncio.run(handlers.extract_section_lines(page=None, args=args, runtime_state=runtime_state))
        assert False, "expected insufficient section data failure"
    except StructuredExtractionError as exc:
        assert exc.code == "insufficient_section_data"


def test_extract_section_lines_empty_heading_returns_actionable_diagnostics():
    handlers = ActionHandlers()
    args = {"heading_text": "Introduction", "limit": 4}
    runtime_state = {
        "last_page_snapshot": {
            "visible_headings": ["Introduction", "The RFC Series", "RFC Editor"],
            "headings": [
                {"text": "Introduction", "line_count_after": 0, "visible": True},
                {"text": "The RFC Series", "line_count_after": 5, "visible": True},
                {"text": "RFC Editor", "line_count_after": 4, "visible": True},
            ],
            "page_text": "Introduction",
            "url": "https://example.org",
        }
    }

    async def _fake_source_text(**_kwargs):
        return "Introduction\n\n"

    handlers._load_source_text = _fake_source_text  # type: ignore[method-assign]

    try:
        asyncio.run(handlers.extract_section_lines(page=None, args=args, runtime_state=runtime_state))
        assert False, "expected insufficient section data failure"
    except StructuredExtractionError as exc:
        assert exc.code == "insufficient_section_data"
        assert exc.details["reason"] == "empty_section"
        assert exc.details["failed_heading"] == "Introduction"
        assert exc.details["available_non_empty_headings"][0]["text"] == "The RFC Series"
        assert any(item["text"] == "RFC Editor" for item in exc.details["suggested_next_headings"])
