import asyncio

from app.executor.action_handlers import ActionHandlers


def test_table_rows_intent_filters_rows_by_requested_columns_from_goal():
    handler = ActionHandlers()
    rows = [
        {
            "headers": ["Company", "Contact", "Country"],
            "cells": ["Alfreds Futterkiste", "Maria Anders", "Germany"],
            "text": "Alfreds Futterkiste Maria Anders Germany",
        },
        {
            "headers": ["Tag", "Description"],
            "cells": ["<table>", "Defines a table"],
            "text": "<table> Defines a table",
        },
    ]

    async def _rows(*, page, limit):
        return list(rows)

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._extract_table_rows_as_dicts = _rows  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    result = asyncio.run(
        handler.extract_by_intent(
            object(),
            {"intent": "table_rows"},
            {"user_goal": "Extract visible table rows with columns Company, Contact, Country."},
        )
    )

    assert len(result) == 1
    assert result[0]["headers"] == ["Company", "Contact", "Country"]
    assert result[0]["cells"] == ["Alfreds Futterkiste", "Maria Anders", "Germany"]
    assert result[0]["fields_by_header"] == {
        "Company": "Alfreds Futterkiste",
        "Contact": "Maria Anders",
        "Country": "Germany",
    }


def test_table_rows_intent_filters_rows_by_explicit_columns_arg():
    handler = ActionHandlers()
    rows = [
        {"headers": ["Company", "Contact", "Country"], "cells": ["A", "B", "C"], "text": "A B C"},
        {"headers": ["Tag", "Description"], "cells": ["<td>", "Cell"], "text": "<td> Cell"},
    ]

    async def _rows(*, page, limit):
        return list(rows)

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._extract_table_rows_as_dicts = _rows  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    result = asyncio.run(
        handler.extract_by_intent(
            object(),
            {"intent": "table_rows", "columns": ["Company", "Country"]},
            {},
        )
    )

    assert len(result) == 1
    assert result[0]["headers"] == ["Company", "Country"]
    assert result[0]["cells"] == ["A", "C"]
    assert result[0]["fields_by_header"] == {"Company": "A", "Country": "C"}
    assert result[0]["company"] == "A"
    assert result[0]["country"] == "C"
    assert "contact" not in result[0]


def test_result_like_text_fallback_extracts_titles_and_links():
    source_text = """
Search results
Results 1 - 2 of 2
Event (computing)
less, "A Web Crawler With asyncio Coroutines" by A. Jesse Jiryu Davis and Guido van Rossum says implementation uses an asyncio.Event...
9 KB (917 words) - 22:26, 19 January 2026
Coroutine
gevent Stackless Python import asyncio import time from asyncio import Task async def main() -> None...
54 KB (5,484 words) - 02:06, 30 May 2026
Privacy policy
"""
    links = [
        {"text": "Event (computing)", "href": "https://example.org/Event_(computing)"},
        {"text": "Coroutine", "href": "https://example.org/Coroutine"},
    ]

    items = ActionHandlers._collect_result_like_items_from_text(
        source_text=source_text,
        links=links,
        limit=5,
    )

    assert [item["title"] for item in items] == ["Event (computing)", "Coroutine"]
    assert items[0]["href"] == "https://example.org/Event_(computing)"
    assert "asyncio.Event" in items[0]["description"]
