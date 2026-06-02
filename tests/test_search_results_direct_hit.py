from __future__ import annotations

import asyncio

from app.executor.action_handlers import ActionHandlers


def test_search_results_returns_current_page_for_direct_hit():
    class FakePage:
        url = "https://ru.wikipedia.org/wiki/Python"

        async def title(self):
            return "Python"

        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

        async def wait_for_timeout(self, *_args, **_kwargs):
            return None

    handler = ActionHandlers()

    async def _no_items(*, page, limit):
        return []

    handler._collect_result_like_items_generic = _no_items  # type: ignore[method-assign]

    results = asyncio.run(
        handler._collect_search_results_by_intent(
            page=FakePage(),
            args={"intent": "search_results"},
            runtime_state={"user_goal": "find Python and return search results"},
        )
    )

    assert results == [
        {
            "title": "Python",
            "text": "Python",
            "href": "https://ru.wikipedia.org/wiki/Python",
            "link": "https://ru.wikipedia.org/wiki/Python",
        }
    ]
