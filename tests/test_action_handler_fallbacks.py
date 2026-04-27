import asyncio

from app.executor.action_handlers import ActionHandlers


class _FakePage:
    def __init__(self, table_rows=None, list_rows=None, entity_rows=None, links=None):
        self._table_rows = table_rows or []
        self._list_rows = list_rows or []
        self._entity_rows = entity_rows or []
        self._links = links or []

    async def evaluate(self, script, payload=None):
        text = str(script)
        if "table tr" in text:
            return self._table_rows[: int((payload or {}).get("limit", len(self._table_rows)))]
        if "main a[href], article a[href], ul li, ol li" in text:
            return self._list_rows[: int((payload or {}).get("limit", len(self._list_rows)))]
        if "main li, article li, main article" in text:
            return self._entity_rows[: int((payload or {}).get("limit", len(self._entity_rows)))]
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


def test_extract_structured_items_rejects_broad_pattern_and_prefers_dom_fallback():
    handler = ActionHandlers()
    page = _FakePage(table_rows=[["Protocol Registries", "/protocols"], ["Time Zones", "/tz"]])

    async def _raise(*_args, **_kwargs):
        raise AssertionError("regex extractor must not be called for broad repeated pattern")

    handler.extract_pattern_from_page_text = _raise  # type: ignore[method-assign]
    args = {"pattern": "(.+)", "limit": 2, "fields": {"name": 1, "href": 2}}
    result = asyncio.run(
        handler.extract_structured_items(
            page,
            args,
            runtime_state={"benchmark_context": {"task_family": "repeated_structured_items"}},
        )
    )
    assert result == [{"name": "Protocol Registries", "href": "/protocols"}, {"name": "Time Zones", "href": "/tz"}]
    assert "fallback=table_rows" in args.get("_executor_note", "")


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


def test_broad_repeated_pattern_rejects_low_quality_list_items_fallback():
    handler = ActionHandlers()
    page = _FakePage(list_rows=[["Home"], ["About"], ["Contact"], ["Privacy"], ["Terms"]])

    try:
        asyncio.run(
            handler.extract_structured_items(
                page,
                {"pattern": "(.+)", "limit": 5, "fields": {"name": 1, "detail": 2}},
                runtime_state={"benchmark_context": {"task_family": "repeated_structured_items"}},
            )
        )
        assert False, "expected low-quality list fallback rejection"
    except Exception as exc:  # noqa: BLE001
        assert "high-quality DOM fallback" in str(exc) or "low-quality" in str(exc)


def test_repeated_entity_blocks_extract_python_release_like_items_generically():
    handler = ActionHandlers()
    page = _FakePage(
        entity_rows=[
            {"text": "Python 3.14.0 Oct. 7, 2025 Download", "raw_text": "Python 3.14.0 Oct. 7, 2025 Download", "href": "/downloads/release/python-3140/"},
            {"text": "Python 3.13.7 September 10, 2025 Notes", "raw_text": "Python 3.13.7 September 10, 2025 Notes", "href": "/downloads/release/python-3137/"},
        ]
    )
    args = {"pattern": "(.+)", "limit": 2, "fields": {"title": 1, "date": 2, "href": 3}}
    result = asyncio.run(
        handler.extract_structured_items(
            page,
            args,
            runtime_state={"benchmark_context": {"task_family": "repeated_structured_items"}},
        )
    )
    assert len(result) == 2
    assert any("Python 3.14.0" in item.get("title", "") for item in result)
    assert all(item.get("date") for item in result)
    assert all(str(item.get("href", "")).startswith("/downloads/release/") for item in result)
    assert "fallback=repeated_entity_blocks" in args.get("_executor_note", "")


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


def test_click_selector_plain_text_is_canonicalized_before_locator_resolution():
    handler = ActionHandlers()

    class _FakeLocator:
        def __init__(self, count):
            self._count = count
            self.first = self

        async def count(self):
            return self._count

    class _FakePageForClick(_FakePage):
        def get_by_role(self, role, name=None, exact=False):
            if role == "link" and name == "Protocol Registries":
                return _FakeLocator(1)
            return _FakeLocator(0)

        def get_by_text(self, *_args, **_kwargs):
            return _FakeLocator(0)

        def locator(self, _selector):
            return _FakeLocator(0)

    args = {"selector": "Protocol Registries"}
    locator, meta = asyncio.run(
        handler._resolve_ranked_click_locator(page=_FakePageForClick(), args=args, runtime_state={})
    )
    assert locator is not None
    assert meta["strategy"] == "role_link_name"
    assert args.get("text") == "Protocol Registries"
    assert "selector" not in args


def test_click_meta_text_uses_anchor_fallback_for_locator_resolution():
    handler = ActionHandlers()

    class _FakeLocator:
        def __init__(self, count):
            self._count = count
            self.first = self

        async def count(self):
            return self._count

    class _FakePageForClick(_FakePage):
        def get_by_role(self, role, name=None, exact=False):
            if role == "link" and name == "Tutorial":
                return _FakeLocator(1)
            return _FakeLocator(0)

        def get_by_text(self, *_args, **_kwargs):
            return _FakeLocator(0)

        def locator(self, _selector):
            return _FakeLocator(0)

    args = {"text": "Scenario ID", "anchor": "Tutorial", "exact": True}
    locator, meta = asyncio.run(
        handler._resolve_ranked_click_locator(page=_FakePageForClick(), args=args, runtime_state={})
    )
    assert locator is not None
    assert meta["strategy"] in {"role_link_name", "role_link_anchor"}
    assert args.get("text") == "Tutorial"
    assert "anchor" not in args


def test_click_meta_text_without_fallback_returns_validation_style_error():
    handler = ActionHandlers()
    try:
        asyncio.run(
            handler._resolve_ranked_click_locator(
                page=_FakePage(),
                args={"text": "Scenario ID"},
                runtime_state={},
            )
        )
        assert False, "expected invalid click target error"
    except Exception as exc:  # noqa: BLE001
        assert "meta label" in str(exc)


def test_extract_section_lines_rejects_ungrounded_heading_before_execution():
    handler = ActionHandlers()
    args = {"heading_text": "Wikipedia\nСвободная энциклопедия", "limit": 5}
    runtime_state = {
        "last_page_snapshot": {
            "url": "https://www.python.org/",
            "visible_headings": ["Downloads", "Documentation", "Success Stories"],
            "page_text": "Welcome to Python.org\nDownloads\nDocumentation",
        }
    }
    try:
        asyncio.run(handler.extract_section_lines(page=_FakePage(), args=args, runtime_state=runtime_state))
        assert False, "expected ungrounded heading rejection"
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        assert "section_heading_not_grounded_in_current_snapshot" in message
        diagnostic = args.get("_grounding_diagnostic", {})
        assert diagnostic.get("current_url") == "https://www.python.org/"
        assert diagnostic.get("heading_text") == "Wikipedia\nСвободная энциклопедия"
