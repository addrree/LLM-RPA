import asyncio
from datetime import datetime, timezone

import pytest

from app.executor.action_handlers import ActionHandlers, StructuredExtractionError
from app.observer.page_observer import PageSnapshot
from app.planner.action_vocab import collection_condition_for_goal
from app.planner.planner import Planner
from app.planner.replanner import Replanner


def test_planner_repairs_malformed_extract_items_to_search_results_intent():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://search.sample.test/?q=websocket",
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=websocket"}},
                {
                    "action": "extract_items",
                    "args": {"fields": {"title": {"type": "text"}, "href": {"type": "url"}}},
                    "save_as": "search_results",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["search_results"]},
        },
        "Open a search page and return search_results with title and href.",
    )

    step = normalized["steps"][1]
    assert step["action"] == "extract_by_intent"
    assert step["args"]["intent"] == "search_results"
    assert step["args"]["output_key"] == "search_results"


def test_replanner_repairs_malformed_extract_items_to_search_results_intent():
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=websocket",
        title="Search",
        screenshot_path="",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://search.sample.test/?q=websocket",
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=websocket"}},
                {
                    "action": "extract_items",
                    "args": {"fields": {"title": {"type": "text"}, "href": {"type": "url"}}},
                    "save_as": "search_results",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["search_results"]},
        },
        user_goal="Open a search page and return search_results with title and href.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    step = normalized["steps"][1]
    assert step["action"] == "extract_by_intent"
    assert step["args"]["intent"] == "search_results"
    assert step["args"]["output_key"] == "search_results"


def test_malformed_extract_items_without_output_hint_uses_generic_card_items():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://list.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://list.sample.test"}},
                {"action": "extract_items", "args": {}, "save_as": "after_action"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["after_action"]},
        },
        "Open the page and return the visible list items after the action.",
    )

    step = normalized["steps"][1]
    assert step["action"] == "extract_by_intent"
    assert step["args"]["intent"] == "card_items"
    assert step["args"]["output_key"] == "after_action"


def test_collect_search_results_filters_ui_links_and_prefers_content_results():
    handler = ActionHandlers()

    async def _result_like(*, page, limit):
        return [
            {"title": "Next", "href": "https://search.sample.test/?q=x&page=2", "description": ""},
            {"title": "WebSocket API", "href": "https://docs.sample.test/websocket", "description": "A protocol API."},
            {"title": "Search", "href": "#search", "description": ""},
        ]

    handler._collect_result_like_items_generic = _result_like  # type: ignore[method-assign]
    results = asyncio.run(
        handler._collect_search_results_generic(
            page=object(),
            args={"limit": 5},
            runtime_state={},
        )
    )

    assert [item["title"] for item in results] == ["WebSocket API"]
    assert results[0]["href"] == "https://docs.sample.test/websocket"


def test_search_result_score_uses_query_to_prefer_overview_over_guide_pages():
    overview = {
        "title": "WebSocket - Web APIs",
        "href": "https://docs.sample.test/web/api/websocket",
        "description": "Overview of the WebSocket API.",
    }
    guide = {
        "title": "Writing WebSocket servers",
        "href": "https://docs.sample.test/web/api/websockets_api/writing_websocket_servers",
        "description": "Guide for server implementations.",
    }

    assert ActionHandlers._search_result_score(overview, query="websocket") > ActionHandlers._search_result_score(
        guide,
        query="websocket",
    )


def test_first_result_click_uses_ranked_search_result_href():
    handler = ActionHandlers()

    class _Page:
        def __init__(self):
            self.opened = []

        async def goto(self, url, wait_until="domcontentloaded", timeout=20000):
            self.opened.append((url, wait_until, timeout))

    async def _wait(_page):
        return None

    async def _not_blocked(_page, *, runtime_state=None, stage=""):
        return None

    handler._wait_after_possible_navigation = _wait  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    page = _Page()
    runtime_state = {
        "last_search_results": [
            {"title": "WebSocket API", "href": "https://docs.sample.test/websocket", "score": 10}
        ]
    }

    label = asyncio.run(
        handler.click_by_semantic_target(
            page,
            {"target_text": "open first relevant result", "role": "link"},
            runtime_state,
        )
    )

    assert label == "WebSocket API"
    assert page.opened == [("https://docs.sample.test/websocket", "domcontentloaded", 20000)]
    assert runtime_state["last_opened_result"]["href"] == "https://docs.sample.test/websocket"


def test_search_page_target_click_prefers_overview_result_over_constructor_subpage():
    handler = ActionHandlers()

    class _Page:
        url = "https://docs.sample.test/search?q=WebSocket"

        def __init__(self):
            self.opened = []

        async def goto(self, url, wait_until="domcontentloaded", timeout=20000):
            self.opened.append((url, wait_until, timeout))

    async def _wait(_page):
        return None

    async def _not_blocked(_page, *, runtime_state=None, stage=""):
        return None

    async def _search_results(*, page, args, runtime_state=None):
        return [
            {
                "title": "WebSocket: WebSocket() constructor",
                "href": "https://docs.sample.test/web/api/websocket/websocket",
                "description": "Constructor reference.",
                "score": 5,
            },
            {
                "title": "WebSocket - Web APIs",
                "href": "https://docs.sample.test/web/api/websocket",
                "description": "Overview of the WebSocket API.",
                "score": 5,
            },
        ]

    handler._wait_after_possible_navigation = _wait  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    handler._collect_search_results_generic = _search_results  # type: ignore[method-assign]
    page = _Page()
    runtime_state = {}

    label = asyncio.run(
        handler.click_by_semantic_target(
            page,
            {"target_text": "WebSocket", "role": "link"},
            runtime_state,
        )
    )

    assert label == "WebSocket - Web APIs"
    assert page.opened == [("https://docs.sample.test/web/api/websocket", "domcontentloaded", 20000)]
    assert runtime_state["last_opened_result"]["title"] == "WebSocket - Web APIs"


def test_submit_search_query_from_current_url_uses_visible_searchbox():
    handler = ActionHandlers()

    class _Locator:
        first = None

        def __init__(self):
            self.first = self
            self.filled = []
            self.pressed = []

        async def count(self):
            return 1

        async def fill(self, value):
            self.filled.append(value)

        async def press(self, key):
            self.pressed.append(key)

    class _Page:
        url = "https://docs.sample.test/search?q=websocket"

        def __init__(self):
            self.searchbox = _Locator()

        def locator(self, selector):
            assert selector == "main input[type='search']"
            return self.searchbox

    async def _wait(_page):
        return None

    handler._wait_after_possible_navigation = _wait  # type: ignore[method-assign]
    page = _Page()
    runtime_state = {}

    submitted = asyncio.run(handler._submit_search_query_from_current_url(page=page, runtime_state=runtime_state))

    assert submitted is True
    assert page.searchbox.filled == ["websocket"]
    assert page.searchbox.pressed == ["Enter"]
    assert runtime_state["submitted_search_queries"]


def test_planner_inserts_result_click_for_search_url_metadata_extraction():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://docs.sample.test/search?q=websocket",
            "steps": [
                {"action": "open_url", "args": {"url": "https://docs.sample.test/search?q=websocket"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "field_schema",
                        "fields": {"metadata": {"type": "description"}},
                        "output_key": "metadata",
                    },
                    "save_as": "metadata",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["metadata"]},
        },
        "Open the search page, open the first relevant result, then extract metadata.",
    )

    assert [step["action"] for step in normalized["steps"]][:3] == [
        "open_url",
        "click_by_semantic_target",
        "observe_page",
    ]
    assert normalized["steps"][1]["args"] == {"target_text": "first relevant result", "role": "link"}


def test_replanner_inserts_result_click_for_search_url_metadata_extraction():
    snapshot = PageSnapshot(
        url="https://docs.sample.test/search?q=websocket",
        title="Search",
        screenshot_path="",
        page_text_excerpt="Search",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://docs.sample.test/search?q=websocket",
            "steps": [
                {"action": "open_url", "args": {"url": "https://docs.sample.test/search?q=websocket"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "field_schema",
                        "fields": {"metadata": {"type": "description"}},
                        "output_key": "metadata",
                    },
                    "save_as": "metadata",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["metadata"]},
        },
        user_goal="Open the search page, open the first relevant result, then extract metadata.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    assert [step["action"] for step in normalized["steps"]][:3] == [
        "open_url",
        "click_by_semantic_target",
        "observe_page",
    ]
    assert normalized["steps"][1]["args"] == {"target_text": "first relevant result", "role": "link"}


def test_planner_repairs_collection_goal_that_clicked_bare_link():
    goal = (
        "Open https://cards.sample.test and export story cards whose title contains Arm: "
        "story_title and short_description."
    )
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://cards.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://cards.sample.test"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "click_by_semantic_target", "args": {"target_text": "link", "role": "link"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "extract_by_intent", "args": {"intent": "text_block"}, "save_as": "story_title"},
                {"action": "extract_by_intent", "args": {"intent": "text_block"}, "save_as": "short_description"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["story_title", "short_description"]},
        },
        goal,
    )

    actions = [step["action"] for step in normalized["steps"]]
    assert "click_by_semantic_target" not in actions
    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "items"
    assert extract_step["args"]["condition"] == {"title": "Arm"}
    assert extract_step["args"]["fields"] == {
        "story_title": {"type": "title"},
        "short_description": {"type": "description"},
    }
    assert normalized["expected_result"]["required_fields"] == ["items"]


def test_replanner_repairs_collection_goal_that_clicked_bare_link():
    snapshot = PageSnapshot(
        url="https://cards.sample.test",
        title="Story cards",
        screenshot_path="",
        page_text_excerpt="Python on Arm: 2025 Update",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://cards.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://cards.sample.test"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "click_by_semantic_target", "args": {"target_text": "link", "role": "link"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "extract_by_intent", "args": {"intent": "text_block"}, "save_as": "story_title"},
                {"action": "extract_by_intent", "args": {"intent": "text_block"}, "save_as": "short_description"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["story_title", "short_description"]},
        },
        user_goal=(
            "Open https://cards.sample.test and export story cards whose title contains Arm: "
            "story_title and short_description."
        ),
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items"],
    )

    actions = [step["action"] for step in normalized["steps"]]
    assert "click_by_semantic_target" not in actions
    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["condition"] == {"title": "Arm"}
    assert normalized["expected_result"]["required_fields"] == ["items"]


def test_collection_repair_handles_mojibaked_cyrillic_title_condition():
    goal = (
        "Открой https://cards.sample.test и выгрузи карточки историй, "
        "в названии которых есть Arm: название и краткое описание."
    )
    mojibake_goal = goal.encode("utf-8").decode("cp1251")
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://cards.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://cards.sample.test"}},
                {
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "card_items",
                        "output_key": "extracted",
                        "condition": {"contains": "в названии которых есть Arm: название и краткое описание"},
                        "fields": {
                            "РЅР°Р·РІР°РЅРёРµ": {"type": "text"},
                            "РѕРїРёСЃР°РЅРёРµ": {"type": "text"},
                        },
                    },
                    "save_as": "extracted",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["РЅР°Р·РІР°РЅРёРµ", "РѕРїРёСЃР°РЅРёРµ"]},
        },
        mojibake_goal,
    )

    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert collection_condition_for_goal(mojibake_goal) == {"title": "Arm"}
    assert extract_step["args"]["condition"] == {"title": "Arm"}
    assert extract_step["args"]["fields"] == {
        "РЅР°Р·РІР°РЅРёРµ": {"type": "title"},
        "РѕРїРёСЃР°РЅРёРµ": {"type": "description"},
    }


def test_planner_repairs_anchor_value_goal_that_was_planned_as_cards():
    goal = (
        "Открой https://www.wikipedia.org и извлеки название языка Français и число статей рядом "
        "с этим языком. Верни language_name и article_count."
    )
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://www.wikipedia.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "action": "extract_by_intent",
                    "args": {"intent": "card_items", "output_key": "items", "limit": 20},
                    "save_as": "items",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {
                "required_fields": [
                    "www_wikipedia_org",
                    "извлеки_название_языка_français",
                    "article_count",
                ]
            },
        },
        goal,
    )

    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert extract_step["args"]["intent"] == "anchor_object"
    assert extract_step["args"]["anchor_text"] == "Français"
    assert extract_step["args"]["output_key"] == "metadata"
    assert extract_step["args"]["fields"] == {
        "language_name": {"type": "text"},
        "article_count": {"type": "number"},
    }
    assert "save_as" not in extract_step
    assert normalized["expected_result"]["required_fields"] == ["language_name", "article_count"]


def test_replanner_repairs_anchor_value_goal_that_was_planned_as_cards():
    snapshot = PageSnapshot(
        url="https://www.wikipedia.org/",
        title="Wikipedia",
        screenshot_path="",
        page_text_excerpt="Français 2 761 000+ articles",
        timestamp=datetime.now(timezone.utc),
    )
    goal = (
        "Открой https://www.wikipedia.org и извлеки название языка Français и число статей рядом "
        "с этим языком. Верни language_name и article_count."
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://www.wikipedia.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "action": "extract_by_intent",
                    "args": {"intent": "card_items", "output_key": "items", "limit": 20},
                    "save_as": "items",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {
                "required_fields": [
                    "www_wikipedia_org",
                    "извлеки_название_языка_français",
                    "article_count",
                ]
            },
        },
        user_goal=goal,
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["field_schema", "text_block", "value_near_anchor", "anchor_object", "card_items", "table_rows"],
    )

    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert extract_step["args"]["intent"] == "anchor_object"
    assert extract_step["args"]["anchor_text"] == "Français"
    assert extract_step["args"]["output_key"] == "metadata"
    assert "save_as" not in extract_step
    assert normalized["expected_result"]["required_fields"] == ["language_name", "article_count"]


def test_anchor_value_repair_handles_mojibaked_cyrillic_goal():
    goal = (
        "Открой https://www.wikipedia.org и извлеки название языка Français и число статей рядом "
        "с этим языком. Верни language_name и article_count."
    )
    mojibake_goal = goal.encode("utf-8").decode("cp1251")
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://www.wikipedia.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {
                    "action": "extract_by_intent",
                    "args": {"intent": "card_items", "output_key": "items"},
                    "save_as": "items",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["language_name", "article_count"]},
        },
        mojibake_goal,
    )

    extract_step = next(step for step in normalized["steps"] if step["action"] == "extract_by_intent")
    assert extract_step["args"]["intent"] == "anchor_object"
    assert extract_step["args"]["anchor_text"] == "Français"
    assert normalized["expected_result"]["required_fields"] == ["language_name", "article_count"]


def test_bare_link_semantic_click_is_controlled_failure():
    handler = ActionHandlers()

    class _Page:
        url = "https://cards.sample.test"

    with pytest.raises(StructuredExtractionError) as exc_info:
        asyncio.run(handler.click_by_semantic_target(_Page(), {"target_text": "link", "role": "link"}, {}))

    assert exc_info.value.code == "target_too_broad"
