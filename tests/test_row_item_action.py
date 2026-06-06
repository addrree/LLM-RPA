import asyncio
from datetime import datetime, timezone

import pytest

from app.executor.action_handlers import ActionHandlers, StructuredExtractionError
from app.observer.page_observer import PageSnapshot
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator


class _FakeControl:
    def __init__(self, *, text="", attrs=None):
        self.text = text
        self.attrs = attrs or {}
        self.clicked = False

    async def inner_text(self, timeout=500):
        return self.text

    async def get_attribute(self, name):
        return self.attrs.get(name)

    async def click(self):
        self.clicked = True


class _FakeLocator:
    def __init__(self, items):
        self.items = list(items)
        self.first = self.items[0] if self.items else self

    async def count(self):
        return len(self.items)

    def nth(self, index):
        return self.items[index]


class _FakeRow:
    def __init__(self, controls):
        self.controls = controls
        self.clicked = False
        self.first = self

    async def click(self):
        self.clicked = True

    def locator(self, _selector):
        return _FakeLocator(self.controls)


class _FakePage:
    def __init__(self, row):
        self.row = row

    def locator(self, selector):
        assert selector == "#todo > li:nth-of-type(2)"
        return self.row


async def _not_blocked(*_args, **_kwargs):
    return None


def test_fieldless_value_condition_uses_row_text_not_value_field():
    rows = [
        {"text": "Tutorial container Pay bills close Next", "selector": "main"},
        {"text": "Pay bills", "selector": "#todo > li:nth-of-type(2)", "tag": "li"},
    ]

    matched = ActionHandlers._filter_structured_rows_by_condition(
        rows=rows,
        condition={"field": None, "operator": "contains", "value": "Pay bills"},
    )

    assert matched == rows
    assert ActionHandlers._condition_field_operator_value(
        {"field": None, "operator": "contains", "value": "Pay bills"}
    ) is None


def test_find_row_missing_condition_is_controlled_failure():
    handler = ActionHandlers()

    async def _table_rows(*, page, limit):
        return []

    async def _generic_rows(*, page, limit):
        return []

    handler._extract_table_rows_as_dicts = _table_rows  # type: ignore[method-assign]
    handler._collect_row_candidates_generic = _generic_rows  # type: ignore[method-assign]

    with pytest.raises(StructuredExtractionError) as exc_info:
        asyncio.run(handler.find_row_by_condition(object(), {"condition": None}, {}))

    assert exc_info.value.code == "missing_row_condition"


def test_click_row_action_clicks_control_inside_selected_row():
    handler = ActionHandlers()
    close_control = _FakeControl(attrs={"class": "close"})
    row = _FakeRow([_FakeControl(text="Details"), close_control])
    page = _FakePage(row)

    result = asyncio.run(
        handler.click_row_action(
            page,
            {
                "action_name": "delete",
                "row_ref": {"selector": "#todo > li:nth-of-type(2)", "text": "Pay bills"},
            },
            {},
        )
    )

    assert result["clicked"] is True
    assert close_control.clicked is True
    assert row.clicked is False


def test_click_row_action_normalizes_control_phrase_condition_before_row_lookup():
    handler = ActionHandlers()
    close_control = _FakeControl(attrs={"class": "close"})
    row = _FakeRow([_FakeControl(text="Details"), close_control])
    page = _FakePage(row)
    captured = {}

    async def _find_row(_page, args, _runtime_state):
        captured["condition"] = args["condition"]
        return {"selector": "#todo > li:nth-of-type(2)", "text": "Pay bills"}

    handler.find_row_by_condition = _find_row  # type: ignore[method-assign]

    result = asyncio.run(
        handler.click_row_action(
            page,
            {
                "action_name": "close",
                "condition": {"contains": "close button for list item containing text Pay bills"},
            },
            {},
        )
    )

    assert captured["condition"] == {"field": None, "operator": "contains", "value": "Pay bills"}
    assert result["clicked"] is True
    assert close_control.clicked is True


def test_row_action_condition_normalizes_subject_without_relation_word():
    assert ActionHandlers._normalize_row_action_condition(
        {"contains": "close button for list item Pay bills"}
    ) == {"field": None, "operator": "contains", "value": "Pay bills"}


def test_find_row_by_condition_accepts_control_phrase_condition():
    handler = ActionHandlers()
    rows = [{"text": "Pay bills", "selector": "#todo > li:nth-of-type(2)", "tag": "li"}]

    async def _table_rows(*, page, limit):
        return []

    async def _generic_rows(*, page, limit):
        return rows

    handler._extract_table_rows_as_dicts = _table_rows  # type: ignore[method-assign]
    handler._collect_row_candidates_generic = _generic_rows  # type: ignore[method-assign]

    result = asyncio.run(
        handler.find_row_by_condition(
            object(),
            {"condition": {"contains": "close button for list item containing text Pay bills"}},
            {},
        )
    )

    assert result["selector"] == "#todo > li:nth-of-type(2)"


def test_row_candidate_collection_includes_accessible_frames():
    handler = ActionHandlers()

    class _EvalContext:
        def __init__(self, rows, url="https://frame.sample.test"):
            self.rows = rows
            self.url = url

        async def evaluate(self, *_args, **_kwargs):
            return list(self.rows)

    class _FramedPage(_EvalContext):
        def __init__(self):
            super().__init__([])
            self.main_frame = _EvalContext([])
            self.child_frame = _EvalContext(
                [{"text": "Pay bills", "selector": "#myUL > li:nth-of-type(2)", "tag": "li"}],
                url="https://frame.sample.test/demo",
            )
            self.frames = [self.main_frame, self.child_frame]

    rows = asyncio.run(handler._collect_row_candidates_generic(page=_FramedPage(), limit=20))

    assert rows == [
        {
            "text": "Pay bills",
            "selector": "#myUL > li:nth-of-type(2)",
            "tag": "li",
            "frame_index": 1,
            "frame_url": "https://frame.sample.test/demo",
        }
    ]


def test_card_items_from_row_candidates_preserves_frame_items_and_strips_close_symbol():
    rows = [
        {"text": "Buy eggs Г—", "selector": "#todo > li:nth-of-type(1)", "frame_index": 1, "frame_url": "https://frame.sample.test/demo"},
        {"text": "Read a book Г—", "selector": "#todo > li:nth-of-type(2)", "frame_index": 1, "frame_url": "https://frame.sample.test/demo"},
    ]

    items = ActionHandlers._card_items_from_row_candidates(rows=rows, limit=10)

    assert [item["title"] for item in items] == ["Buy eggs", "Read a book"]
    assert items[0]["frame_index"] == 1
    assert items[0]["frame_url"] == "https://frame.sample.test/demo"


def test_click_row_action_uses_frame_context_from_row_ref():
    handler = ActionHandlers()
    close_control = _FakeControl(attrs={"class": "close"})
    row = _FakeRow([close_control])

    class _Frame:
        def locator(self, selector):
            assert selector == "#myUL > li:nth-of-type(2)"
            return row

    class _Page:
        frames = [object(), _Frame()]

        def locator(self, _selector):
            raise AssertionError("top-level page locator should not be used for framed row")

    result = asyncio.run(
        handler.click_row_action(
            _Page(),
            {
                "action_name": "delete",
                "row_ref": {
                    "selector": "#myUL > li:nth-of-type(2)",
                    "text": "Pay bills",
                    "frame_index": 1,
                },
            },
            {},
        )
    )

    assert result["clicked"] is True
    assert close_control.clicked is True


def test_click_row_action_uses_force_click_for_offscreen_control():
    handler = ActionHandlers()

    class _OffscreenControl(_FakeControl):
        async def click(self, *args, **kwargs):
            if kwargs.get("force"):
                self.clicked = True
                return
            raise RuntimeError("element is outside of the viewport")

    close_control = _OffscreenControl(attrs={"class": "close"})
    row = _FakeRow([close_control])
    page = _FakePage(row)

    result = asyncio.run(
        handler.click_row_action(
            page,
            {
                "action_name": "delete",
                "row_ref": {"selector": "#todo > li:nth-of-type(2)", "text": "Pay bills"},
            },
            {},
        )
    )

    assert result["clicked"] is True
    assert close_control.clicked is True


def test_click_row_action_filters_non_unique_row_selector_by_row_text():
    handler = ActionHandlers()
    wrong_control = _FakeControl(text="Details")
    close_control = _FakeControl(attrs={"class": "close"})

    class _TextRow(_FakeRow):
        def __init__(self, text, controls):
            super().__init__(controls)
            self.text = text

    class _RowLocator:
        def __init__(self, rows):
            self.rows = list(rows)
            self.first = self.rows[0]

        async def count(self):
            return len(self.rows)

        def filter(self, *, has_text):
            return _RowLocator([row for row in self.rows if has_text in row.text])

    class _Page:
        def __init__(self, rows):
            self.rows = rows

        def locator(self, selector):
            assert selector == "ul > li.checked:nth-of-type(2)"
            return _RowLocator(self.rows)

    page = _Page(
        [
            _TextRow("Other item", [wrong_control]),
            _TextRow("Pay bills", [close_control]),
        ]
    )

    result = asyncio.run(
        handler.click_row_action(
            page,
            {
                "action_name": "delete",
                "row_ref": {
                    "selector": "ul > li.checked:nth-of-type(2)",
                    "text": "Pay bills",
                },
            },
            {},
        )
    )

    assert result["clicked"] is True
    assert close_control.clicked is True
    assert wrong_control.clicked is False


def test_click_row_action_retries_inside_runnable_example_when_source_row_has_no_control():
    handler = ActionHandlers()
    source_row = _FakeRow([])
    close_control = _FakeControl(attrs={"class": "close"})
    live_row = _FakeRow([close_control])

    class _RunnableExamplePage:
        def __init__(self):
            self.navigated_to = ""

        def locator(self, selector):
            if selector == "#source-row":
                return source_row
            if selector == "#live-row":
                return live_row
            raise AssertionError(f"unexpected selector: {selector}")

        async def evaluate(self, *_args, **_kwargs):
            return {"href": "https://docs.sample.test/live-example", "text": "Try it", "score": 8}

        async def goto(self, href, **_kwargs):
            self.navigated_to = href

    page = _RunnableExamplePage()

    async def _find_row(_page, args, _runtime_state):
        assert page.navigated_to == "https://docs.sample.test/live-example"
        assert args["condition"] == {"field": None, "operator": "contains", "value": "Pay bills"}
        return {"selector": "#live-row", "text": "Pay bills"}

    handler.find_row_by_condition = _find_row  # type: ignore[method-assign]

    result = asyncio.run(
        handler.click_row_action(
            page,
            {
                "action_name": "delete",
                "condition": {"field": None, "operator": "contains", "value": "Pay bills"},
                "row_ref": {"selector": "#source-row", "text": "Pay bills"},
            },
            {},
        )
    )

    assert result["clicked"] is True
    assert page.navigated_to == "https://docs.sample.test/live-example"
    assert close_control.clicked is True


def test_planner_normalizes_row_action_condition_and_remaining_items():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://todo.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://todo.sample.test"}},
                {"action": "find_row_by_condition", "args": {}, "save_as": "target_row"},
                {"action": "click_row_action", "args": {"action_name": "delete"}, "save_as": "row_action"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["remaining_items"]},
        },
        "Open the todo page, delete the item with text Pay bills. After the action return remaining items.",
    )

    find_step = normalized["steps"][1]
    remaining_step = normalized["steps"][3]
    assert find_step["args"]["condition"] == {"field": None, "operator": "contains", "value": "Pay bills"}
    assert remaining_step["action"] == "extract_by_intent"
    assert remaining_step["args"]["intent"] == "card_items"
    assert remaining_step["save_as"] == "remaining_items"
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_malformed_remaining_extract_items_becomes_generic_card_items():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://todo.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://todo.sample.test"}},
                {
                    "action": "click_row_action",
                    "args": {
                        "action_name": "delete",
                        "condition": {"field": None, "operator": "contains", "value": "Pay bills"},
                    },
                    "save_as": "row_action",
                },
                {"action": "extract_items", "args": {}, "save_as": "remaining_items"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["remaining_items"]},
        },
        "Open the todo page, delete the item with text Pay bills. After the action return remaining items.",
    )

    remaining_step = next(step for step in normalized["steps"] if step.get("save_as") == "remaining_items")
    assert remaining_step["action"] == "extract_by_intent"
    assert remaining_step["args"]["intent"] == "card_items"
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_empty_pattern_structured_items_becomes_generic_card_items():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://todo.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://todo.sample.test"}},
                {
                    "action": "extract_structured_items",
                    "args": {"pattern": "", "output_key": "remaining_items"},
                    "save_as": "remaining_items",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["remaining_items"]},
        },
        "Open the todo page and return remaining items.",
    )

    remaining_step = next(step for step in normalized["steps"] if step.get("save_as") == "remaining_items")
    assert remaining_step["action"] == "extract_by_intent"
    assert remaining_step["args"]["intent"] == "card_items"
    assert remaining_step["args"]["output_key"] == "remaining_items"
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_replanner_empty_pattern_structured_items_becomes_generic_card_items():
    snapshot = PageSnapshot(
        url="https://todo.sample.test",
        title="Todo",
        screenshot_path="",
        page_text_excerpt="Pay bills",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://todo.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://todo.sample.test"}},
                {
                    "action": "extract_structured_items",
                    "args": {"pattern": "", "output_key": "remaining_items"},
                    "save_as": "remaining_items",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["remaining_items"]},
        },
        user_goal="Open the todo page and return remaining items.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    remaining_step = next(step for step in normalized["steps"] if step.get("save_as") == "remaining_items")
    assert remaining_step["action"] == "extract_by_intent"
    assert remaining_step["args"]["intent"] == "card_items"
    assert remaining_step["args"]["output_key"] == "remaining_items"
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_replanner_normalizes_cyrillic_row_action_condition():
    snapshot = PageSnapshot(
        url="https://todo.sample.test",
        title="Todo",
        screenshot_path="",
        page_text_excerpt="Pay bills",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://todo.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://todo.sample.test"}},
                {"action": "find_row_by_condition", "args": {}, "save_as": "target_row"},
                {"action": "click_row_action", "args": {}, "save_as": "row_action"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["remaining_items"]},
        },
        user_goal="Открой список, удали строку с текстом Pay bills. После действия верни оставшиеся элементы.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    find_step = normalized["steps"][1]
    click_step = normalized["steps"][2]
    assert find_step["args"]["condition"] == {"field": None, "operator": "contains", "value": "Pay bills"}
    assert click_step["args"]["action_name"] == "delete"
    PlanValidator().validate(TaskSpec.model_validate(normalized))
