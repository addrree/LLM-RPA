from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.executor.action_handlers import ActionHandlers, StructuredExtractionError
from app.observer.page_observer import PageSnapshot
from app.planner.action_vocab import normalize_plan_action_aliases
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator


def test_condition_term_groups_accept_generic_item_fields():
    assert ActionHandlers._condition_term_groups({"title": ["Python", "AI"]}) == [["Python"], ["AI"]]
    assert ActionHandlers._condition_term_groups({"arbitrary_metric": "42"}) == [["42"]]


def test_row_action_contract_accepts_arbitrary_command():
    assert Replanner._row_action_name_for_goal('Archive the item called "Quarterly report"') == "archive"
    assert Replanner._row_condition_for_goal('Archive the item called "Quarterly report"') == {
        "field": None,
        "operator": "contains",
        "value": "Quarterly report",
    }
    PlanValidator._validate_click_row_action_args(
        {"action_name": "archive", "condition": {"contains": "Quarterly report"}}
    )


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


def test_title_condition_does_not_match_raw_text_when_title_is_null():
    rows = [
        {"title": None, "description": None, "raw_text": "Python on Arm: 2025 Update", "selector": "main"},
        {"title": "Python on Arm: 2025 Update", "description": "Architecture update.", "href": "/arm"},
    ]

    filtered, note = ActionHandlers._apply_confident_item_filter(
        items=rows,
        args={"condition": {"title": "Arm"}},
        runtime_state={},
        output_key="cards",
    )

    assert filtered == [rows[1]]
    assert note == "filter_applied"


def test_card_projection_keeps_requested_title_description_and_href():
    item = {
        "title": "Python on Arm: 2025 Update",
        "description": "A deployment story for Arm systems.",
        "href": "https://stories.sample.test/arm",
        "raw_text": "Python on Arm: 2025 Update A deployment story for Arm systems.",
    }

    projected = ActionHandlers._project_item_to_schema(
        item=item,
        fields={
            "title": {"type": "text"},
            "description": {"type": "description"},
            "href": {"type": "url"},
        },
    )

    assert projected["title"] == "Python on Arm: 2025 Update"
    assert projected["description"] == "A deployment story for Arm systems."
    assert projected["href"] == "https://stories.sample.test/arm"


def test_card_items_filters_before_projecting_requested_fields():
    handler = ActionHandlers()

    class _Page:
        url = "https://stories.sample.test"

        async def evaluate(self, *_args, **_kwargs):
            return [
                {
                    "title": "Python on Arm: 2025 Update",
                    "name": "Python on Arm: 2025 Update",
                    "description": "Arm platform update.",
                    "snippet": "Arm platform update.",
                    "href": "/arm",
                    "link": "/arm",
                    "selector": "article:nth-of-type(1)",
                    "raw_text": "Python on Arm: 2025 Update Arm platform update.",
                },
                {
                    "title": "Python on x86",
                    "name": "Python on x86",
                    "description": "Other platform update.",
                    "snippet": "Other platform update.",
                    "href": "/x86",
                    "link": "/x86",
                    "selector": "article:nth-of-type(2)",
                    "raw_text": "Python on x86 Other platform update.",
                },
            ]

    result = asyncio.run(
        handler.extract_by_intent(
            _Page(),
            {
                "intent": "card_items",
                "output_key": "arm_cards",
                "condition": {"title": "Arm"},
                "fields": {
                    "story_title": {"type": "title"},
                    "short_description": {"type": "description"},
                },
            },
            {},
        )
    )

    assert result == [
        {
            "story_title": "Python on Arm: 2025 Update",
            "short_description": "Arm platform update.",
            "selector": "article:nth-of-type(1)",
            "raw_text": "Python on Arm: 2025 Update Arm platform update.",
        }
    ]


def test_card_items_condition_can_match_content_link_candidate_before_broad_block():
    handler = ActionHandlers()

    class _Page:
        url = "https://stories.sample.test"

        async def evaluate(self, script, *_args, **_kwargs):
            if "resultContainers" in str(script):
                return [
                    {
                        "title": "Python on Arm: 2025 Update",
                        "text": "Python on Arm: 2025 Update",
                        "description": "Arm platform update.",
                        "href": "/arm",
                        "link": "/arm",
                        "selector": "h2 > a",
                    }
                ]
            return [
                {
                    "title": "Newest success stories",
                    "name": "Newest success stories",
                    "description": "Python on Arm: 2025 Update Arm platform update.",
                    "snippet": "Python on Arm: 2025 Update Arm platform update.",
                    "selector": "section",
                    "raw_text": "Newest success stories Python on Arm: 2025 Update Arm platform update.",
                }
            ]

    result = asyncio.run(
        handler.extract_by_intent(
            _Page(),
            {
                "intent": "card_items",
                "output_key": "cards",
                "condition": {"title": "Arm"},
                "fields": {
                    "story_title": {"type": "title"},
                    "short_description": {"type": "description"},
                },
            },
            {},
        )
    )

    assert result == [
        {
            "story_title": "Python on Arm: 2025 Update",
            "short_description": "Arm platform update.",
            "selector": "h2 > a",
        }
    ]


def test_module_like_card_items_are_ranked_ahead_of_toc_categories():
    cards = [
        {
            "title": "Introduction",
            "description": "Notes on availability",
            "raw_text": "Introduction Notes on availability",
        },
        {
            "title": "Built-in Types",
            "description": "Truth Value Testing",
            "raw_text": "Built-in Types " + ("many nested topics " * 80),
        },
        {
            "title": "asyncio — Asynchronous I/O.",
            "href": "https://docs.sample.test/asyncio.html",
            "raw_text": "asyncio Asynchronous I/O.",
        },
        {
            "title": "collections — Container datatypes.",
            "href": "https://docs.sample.test/collections.html",
            "raw_text": "collections Container datatypes.",
        },
        {
            "title": "concurrent.futures — Launching parallel tasks.",
            "href": "https://docs.sample.test/concurrent.futures.html",
            "raw_text": "concurrent.futures Launching parallel tasks.",
        },
    ]

    ranked = ActionHandlers._rank_module_like_items_if_requested(
        cards=cards,
        args={"output_key": "modules"},
        runtime_state={"user_goal": "Extract several standard library modules with module name and description."},
    )

    assert [item["title"] for item in ranked[:3]] == ["asyncio", "collections", "concurrent.futures"]
    assert ranked[0]["description"] == "Asynchronous I/O."


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


def test_planner_repairs_malformed_extract_items_to_card_items_intent():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org/cards",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org/cards"}},
                {
                    "action": "extract_items",
                    "args": {"fields": {"title": {"type": "text"}, "description": {"type": "text"}}},
                    "save_as": "cards",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["cards"]},
        },
        "Open a page and extract repeated cards with title and description.",
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "cards"
    assert extract_step["args"]["limit"] == 20
    assert extract_step["save_as"] == "cards"


def test_planner_strengthens_weak_wait_for_before_validation():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "wait_for", "args": {"text": "Pricing"}},
                {"action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["value"]},
        },
        "Open a page and extract the heading.",
    )

    wait_step = normalized["steps"][1]
    assert wait_step["action"] == "wait_for"
    assert wait_step["args"]["selector"] == "main h1, article h1, h1"
    assert "text" not in wait_step["args"]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_planner_adds_generic_visual_target_from_output_shape():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "visual_extract_object_count", "args": {}, "save_as": "visible_blocks"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["visible_blocks"]},
        },
        "Visually count visible objects.",
    )

    visual_step = normalized["steps"][1]
    assert visual_step["args"]["target"] == "visible_blocks"
    PlanValidator().validate(TaskSpec.model_validate(normalized))


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


def test_replanner_strengthens_remaining_weak_wait_for_before_validation():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="Example",
        screenshot_path="",
        page_text_excerpt="Pricing",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "wait_for", "args": {"text": "Pricing"}},
                {"action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["value"]},
        },
        user_goal="Open a page and extract the heading.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    wait_step = normalized["steps"][1]
    assert wait_step["args"]["selector"] == "main h1, article h1, h1"
    assert "text" not in wait_step["args"]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_planner_drops_empty_wait_for_before_validation():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "wait_for", "args": {}},
                {"action": "extract_by_intent", "args": {"intent": "page_title"}, "save_as": "page_title"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["page_title"]},
        },
        "Open the official site and return the page title.",
    )

    assert "wait_for" not in [step["action"] for step in normalized["steps"]]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_replanner_drops_empty_wait_for_before_validation():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="Example",
        screenshot_path="",
        page_text_excerpt="Example",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "wait_for", "args": {}},
                {"action": "extract_by_intent", "args": {"intent": "page_title"}, "save_as": "page_title"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["page_title"]},
        },
        user_goal="Open the official site and return the page title.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    assert "wait_for" not in [step["action"] for step in normalized["steps"]]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


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


def test_extract_by_intent_accepts_only_generic_card_intent():
    handler = ActionHandlers()

    async def _cards(*, page, args, runtime_state=None):
        return [
            {
                "title": "Project Alpha",
                "description": "Automation toolkit",
                "href": "https://sample.test/a",
                "numeric_value_required": bool(args.get("numeric_value_required")),
            }
        ]

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._collect_card_items_generic = _cards  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    cards = asyncio.run(
        handler.extract_by_intent(
            object(),
            {"intent": "card_items", "condition": {"title": "Alpha"}, "numeric_value_required": True},
            {},
        )
    )

    assert cards[0]["title"] == "Project Alpha"
    assert cards[0]["numeric_value_required"] is True

    with pytest.raises(StructuredExtractionError):
        asyncio.run(handler.extract_by_intent(object(), {"intent": "product_cards"}, {}))
