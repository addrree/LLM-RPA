import asyncio

from app.executor.action_handlers import ActionHandlers


def test_click_by_semantic_target_opens_first_result_like_link():
    handler = ActionHandlers()

    async def _card_items(*, page, args, runtime_state=None):
        return [{"title": "Project Alpha", "href": "https://projects.sample.test/alpha"}]

    async def _wait(_page):
        return None

    async def _raise(_page, *, runtime_state=None, stage=""):
        return None

    class _Page:
        def __init__(self):
            self.opened = []

        async def goto(self, url, wait_until="domcontentloaded", timeout=20000):
            self.opened.append((url, wait_until, timeout))

    handler._collect_card_items_generic = _card_items  # type: ignore[method-assign]
    handler._wait_after_possible_navigation = _wait  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _raise  # type: ignore[method-assign]

    page = _Page()
    runtime_state = {}
    result = asyncio.run(
        handler.click_by_semantic_target(
            page,
            {"target_text": "first relevant repository result", "role": "link"},
            runtime_state,
        )
    )

    assert result == "Project Alpha"
    assert page.opened[0][0] == "https://projects.sample.test/alpha"
    assert runtime_state["last_opened_result"]["title"] == "Project Alpha"


def test_visual_extract_object_count_uses_dom_geometry_for_countable_targets():
    handler = ActionHandlers()

    class _Page:
        async def evaluate(self, script, payload=None):
            if "shapeTags" in str(script):
                return {"shape_counts": {}, "shapes": []}
            if "targetKind" in str(script):
                return {"count": 10, "items": []}
            return {}

    runtime_state = {}
    result = asyncio.run(
        handler.visual_extract_object_count(
            _Page(),
            {"target": "link", "region": {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}},
            runtime_state,
        )
    )

    assert result == 10
    assert "visual_dom_geometry" in runtime_state["used_skills"]


def test_row_condition_prefers_specific_row_over_large_matching_container():
    rows = [
        {
            "tag": "div",
            "role": "",
            "text": "Tutorial page Hit the gym Remove Next",
            "selector": "main > div",
            "cells": [],
        },
        {
            "tag": "li",
            "role": "",
            "text": "Hit the gym",
            "selector": "#myUL > li:nth-of-type(1)",
            "cells": [],
        },
    ]

    selected = ActionHandlers._best_matching_row_by_terms(rows=rows, terms=["Hit the gym"])

    assert selected["selector"] == "#myUL > li:nth-of-type(1)"


def test_row_action_control_match_does_not_treat_next_as_delete_x():
    target_words = ["trash", "delete", "remove", "close", "×"]

    assert not ActionHandlers._row_action_control_matches(haystack="Next ws-btn", target_words=target_words)
    assert ActionHandlers._row_action_control_matches(haystack="× close", target_words=target_words)
