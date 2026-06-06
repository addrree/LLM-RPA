import asyncio
from datetime import datetime, timezone

from app.executor.action_handlers import ActionHandlers
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator


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


def test_table_row_condition_by_header_returns_correct_w3_like_row_split():
    handler = ActionHandlers()
    rows = [
        ActionHandlers._build_row_payload(
            {
                "headers": ["Company", "Contact", "Country"],
                "cells": ["Alfreds Futterkiste", "Maria Anders", "Germany"],
                "text": "Alfreds Futterkiste Maria Anders Germany",
                "selector": "table tr:nth-of-type(2)",
            }
        ),
        ActionHandlers._build_row_payload(
            {
                "headers": ["Company", "Contact", "Country"],
                "cells": ["Centro comercial Moctezuma", "Francisco Chang", "Mexico"],
                "text": "Centro comercial Moctezuma Francisco Chang Mexico",
                "selector": "table tr:nth-of-type(3)",
            }
        ),
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
            {
                "intent": "table_rows",
                "columns": ["Company", "Contact", "Country"],
                "condition": {"field": "Company", "operator": "contains", "value": "Alfreds"},
            },
            {},
        )
    )

    assert len(result) == 1
    assert result[0]["fields_by_header"] == {
        "Company": "Alfreds Futterkiste",
        "Contact": "Maria Anders",
        "Country": "Germany",
    }
    assert result[0]["company"] == "Alfreds Futterkiste"
    assert result[0]["contact"] == "Maria Anders"
    assert result[0]["country"] == "Germany"
    assert result[0]["company"] != result[0]["text"]
    assert result[0]["contact"] != result[0]["text"]


def test_table_row_condition_without_field_falls_back_to_full_row_text():
    rows = [
        {"company": "A", "contact": "B", "country": "C", "cells": ["A", "B", "C"], "text": "A B C"},
        {"company": "Alfreds Futterkiste", "contact": "Maria Anders", "country": "Germany", "cells": [], "text": "Alfreds Futterkiste Maria Anders Germany"},
    ]

    result = ActionHandlers._filter_structured_rows_by_condition(
        rows=rows,
        condition={"field": None, "operator": "contains", "value": "Alfreds"},
    )

    assert result == [rows[1]]


def test_field_schema_table_row_fallback_projects_requested_headers_from_source_text():
    result = ActionHandlers._field_schema_table_row_from_source_text(
        fields={
            "где_company_содержит_alfreds_верни_company": {"type": "text"},
            "contact": {"type": "text"},
            "country": {"type": "text"},
        },
        source_text="""
        Company\tContact\tCountry
        Alfreds Futterkiste\tMaria Anders\tGermany
        Centro comercial Moctezuma\tFrancisco Chang\tMexico
        """,
        runtime_state={"user_goal": "Найди строку таблицы, где Company содержит Alfreds. Верни Company, Contact и Country."},
    )

    assert result["где_company_содержит_alfreds_верни_company"] == "Alfreds Futterkiste"
    assert result["contact"] == "Maria Anders"
    assert result["country"] == "Germany"
    assert result["Company"] == "Alfreds Futterkiste"
    assert result["status"] == "success"


def test_planner_maps_field_schema_required_headers_to_parent_artifact():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://tables.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://tables.sample.test"}},
                {
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "field_schema",
                        "fields": {
                            "Company": {"type": "text"},
                            "Contact": {"type": "text"},
                            "Country": {"type": "text"},
                        },
                    },
                    "save_as": "extracted_fields",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["Company", "Contact", "Country"]},
        },
        "Find the table row where Company contains Alfreds. Return Company, Contact, and Country.",
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["expected_result"]["required_fields"] == ["extracted_fields"]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


def test_replanner_rechecks_required_fields_after_field_schema_coalesce():
    snapshot = PageSnapshot(
        url="https://tables.sample.test",
        title="Tables",
        screenshot_path="",
        page_text_excerpt="Company Contact Country Alfreds Futterkiste Maria Anders Germany",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://tables.sample.test",
            "steps": [
                {"action": "open_url", "args": {"url": "https://tables.sample.test"}},
                {"action": "extract_text", "args": {"selector": ".company"}, "save_as": "Company"},
                {"action": "extract_text", "args": {"selector": ".contact"}, "save_as": "Contact"},
                {"action": "extract_text", "args": {"selector": ".country"}, "save_as": "Country"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["Company", "Contact", "Country"]},
        },
        user_goal="Find the table row where Company contains Alfreds. Return Company, Contact, and Country.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    assert normalized["steps"][1]["action"] == "extract_by_intent"
    assert normalized["steps"][1]["save_as"] == "extracted_fields"
    assert normalized["expected_result"]["required_fields"] == ["extracted_fields"]
    PlanValidator().validate(TaskSpec.model_validate(normalized))


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
