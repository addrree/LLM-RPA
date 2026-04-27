import asyncio

from app.executor.action_handlers import ActionHandlers


class _FakePage:
    def __init__(self, table_rows=None, list_rows=None, links=None):
        self._table_rows = table_rows or []
        self._list_rows = list_rows or []
        self._links = links or []

    async def evaluate(self, script, payload=None):
        text = str(script)
        if "table tr" in text:
            return self._table_rows[: int((payload or {}).get("limit", len(self._table_rows)))]
        if "main a[href], article a[href], ul li, ol li" in text:
            return self._list_rows[: int((payload or {}).get("limit", len(self._list_rows)))]
        if "querySelectorAll(\"a[href]\")" in text:
            return self._links
        return []


def test_extract_structured_items_falls_back_to_table_rows_when_regex_fails():
    handler = ActionHandlers()
    page = _FakePage(table_rows=[[".aaa", "aaa", "Alpha"], [".aarp", "aarp", "AARP"]])

    async def _raise(*_args, **_kwargs):
        raise ValueError("Pattern not found")

    handler.extract_pattern_from_page_text = _raise  # type: ignore[method-assign]
    result = asyncio.run(
        handler.extract_structured_items(
            page,
            {"pattern": r"^\.(\w+)\t([\w-]+)\t(.+)$", "limit": 2, "fields": {"name": 1, "detail": 3}},
            runtime_state={},
        )
    )
    assert result == [{"name": ".aaa", "detail": "aaa"}, {"name": ".aarp", "detail": "aarp"}]


def test_extract_structured_items_falls_back_to_list_rows_when_table_absent():
    handler = ActionHandlers()
    page = _FakePage(list_rows=[["Software", "/software/"], ["Licenses", "/licenses/"]])

    async def _raise(*_args, **_kwargs):
        raise ValueError("Pattern not found")

    handler.extract_pattern_from_page_text = _raise  # type: ignore[method-assign]
    result = asyncio.run(
        handler.extract_structured_items(
            page,
            {"pattern": r"(never-matches)", "limit": 2, "fields": {"name": 1, "detail": 2}},
            runtime_state={},
        )
    )
    assert result == [{"name": "Software", "detail": "/software/"}, {"name": "Licenses", "detail": "/licenses/"}]


def test_contact_value_type_supports_email_or_phone():
    assert ActionHandlers._resolve_value_pattern("email_or_phone")
    assert ActionHandlers._is_valid_typed_contact_value(value="support@example.org", value_type="email_or_phone")
    assert ActionHandlers._is_valid_typed_contact_value(value="+1 415 555 0100", value_type="email_or_phone")


def test_click_helpers_infer_slug_and_href_from_visible_links():
    handler = ActionHandlers()
    page = _FakePage(links=[{"text": "Software", "href": "/software/"}, {"text": "About GNU", "href": "/gnu/"}])
    slug = handler._infer_href_slug_from_text("Software")
    assert slug == "software"
    href = asyncio.run(handler._discover_href_from_visible_links(page=page, text="Software"))
    assert href == "/software/"
