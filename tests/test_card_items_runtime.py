from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.executor.action_handlers import ActionHandlers
from app.observer.page_observer import PageSnapshot
from app.planner.action_vocab import normalize_plan_action_aliases
from app.planner.planner import Planner
from app.planner.replanner import Replanner


def test_condition_term_groups_accept_generic_item_fields():
    assert ActionHandlers._condition_term_groups({"title": ["Python", "AI"]}) == [["Python"], ["AI"]]


def test_confident_item_filter_applies_only_explicit_conditions():
    rows = [
        {"title": "Python automation", "description": "Browser workflows", "text": "Python automation Browser workflows"},
        {"title": "Rust compiler", "description": "Release notes", "text": "Rust compiler Release notes"},
    ]
    runtime_state = {}

    unfiltered, skipped_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"target": "Python"},
        runtime_state=runtime_state,
        output_key="items",
    )
    filtered, applied_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"condition": {"title": "Python"}},
        runtime_state=runtime_state,
        output_key="items",
    )
    empty, empty_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"condition": {"title": "Go"}},
        runtime_state=runtime_state,
        output_key="items",
    )

    assert unfiltered == rows
    assert skipped_note == "condition_not_applied"
    assert filtered == [rows[0]]
    assert applied_note == "filter_applied"
    assert empty == []
    assert empty_note == "no_matching_items"
    assert {"output_key": "items", "reason": "no_matching_items", "condition": {"title": "Go"}} in runtime_state["condition_filter_diagnostics"]


def test_planner_converts_generic_card_structured_items_to_card_items_intent():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org/cards",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org/cards"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "item_type": "cards",
                        "fields": {"title": 1, "description": 2, "url": 3},
                    },
                    "save_as": "cards",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["cards"]},
        },
        "Open a project catalog and extract cards with title, description, and link.",
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "cards"
    assert extract_step["save_as"] == "cards"


def test_replanner_converts_generic_card_structured_items_to_card_items_intent():
    snapshot = PageSnapshot(
        url="https://example.org/cards",
        title="Cards",
        screenshot_path="",
        page_text_excerpt="Project cards",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org/cards"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": "article.card",
                        "item_type": "card",
                        "fields": {"title": "h2", "description": "p", "url": "a"},
                    },
                    "save_as": "cards",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["cards"]},
        },
        user_goal="Open a project catalog and extract cards with title, description, and link.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "cards"


def test_action_aliases_map_card_items_to_extract_by_intent():
    payload, oov = normalize_plan_action_aliases(
        {
            "steps": [
                {"action": "extract_card_items", "args": {"output_key": "cards"}},
            ]
        }
    )

    assert oov is False
    assert payload["steps"][0]["action"] == "extract_by_intent"
    assert payload["steps"][0]["args"]["intent"] == "card_items"


def test_extract_by_intent_routes_generic_search_product_and_card_intents():
    handler = ActionHandlers()

    async def _search(*, page, args, runtime_state=None):
        return [{"title": "Result", "href": "https://example.com"}]

    async def _products(*, page, args, runtime_state=None):
        return [{"title": "SSD", "price_value": 6900}]

    async def _cards(*, page, args, runtime_state=None):
        return [{"title": "Project Alpha", "description": "Automation toolkit", "href": "https://example.com/a"}]

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._collect_search_results_by_intent = _search  # type: ignore[method-assign]
    handler._collect_product_cards_generic = _products  # type: ignore[method-assign]
    handler._collect_card_items_generic = _cards  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    result = asyncio.run(handler.extract_by_intent(object(), {"intent": "search_results"}, {}))
    products = asyncio.run(handler.extract_by_intent(object(), {"intent": "product_cards"}, {}))
    cards = asyncio.run(handler.extract_by_intent(object(), {"intent": "card_items", "condition": {"title": "Alpha"}}, {}))

    assert result[0]["title"] == "Result"
    assert products[0]["price_value"] == 6900
    assert cards[0]["title"] == "Project Alpha"
