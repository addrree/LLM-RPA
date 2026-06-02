from __future__ import annotations

import asyncio
from pathlib import Path

from app.executor.action_handlers import ActionHandlers
from app.planner.planner import Planner
from app.planner.task_router import TaskRouter


class _BodyLocator:
    def __init__(self, text: str):
        self._text = text

    async def inner_text(self):
        return self._text


class _EmptyLinksLocator:
    async def evaluate_all(self, _script):
        return []


class _FixturePage:
    url = "https://example.org/contacts"

    def __init__(self, text: str):
        self._text = text

    def locator(self, selector: str):
        if selector == "body":
            return _BodyLocator(self._text)
        return _EmptyLinksLocator()

    async def title(self):
        return "Contacts"


async def _not_blocked(*_args, **_kwargs):
    return None


def test_router_contact_query_is_not_repeated_items():
    route = TaskRouter().route("Открой https://example.org и найди контактные данные организации: адрес, телефон и email")

    assert route.task_type in {"single_entity_metadata", "semantic_navigation"}
    assert route.task_type != "repeated_items_extraction"
    assert {"address", "phone", "email"}.issubset(set(route.expected_fields))


def test_router_contact_page_link_is_not_list_output():
    route = TaskRouter().route("Открой https://example.org и верни ссылку на страницу контактов")

    assert route.task_type in {"single_entity_metadata", "semantic_navigation"}
    assert route.expected_output_type != "list"
    assert route.task_type != "repeated_items_extraction"
    assert "contact_page_url" in route.expected_fields


def test_schema_extraction_fixture_returns_structured_object():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    text = """
    Главная
    Контакты
    Адрес: 195251, Санкт-Петербург, Политехническая улица, дом 29
    Телефон: +7 (812) 555-12-34
    Email: office@example.org
    """

    result = asyncio.run(
        handler.extract_by_intent(
            _FixturePage(text),
            {
                "intent": "semantic_region_fields",
                "region_candidates": ["Контакты", "Contacts"],
                "fields": {
                    "address": {"type": "text", "anchors": ["Адрес", "Address"]},
                    "phone": {"type": "phone"},
                    "email": {"type": "email"},
                    "contact_page_url": {"type": "current_url"},
                },
                "output_key": "contact_info",
            },
            {},
        )
    )

    assert result["address"] == "195251, Санкт-Петербург, Политехническая улица, дом 29"
    assert result["phone"] == "+7 (812) 555-12-34"
    assert result["email"] == "office@example.org"
    assert result["contact_page_url"] == "https://example.org/contacts"
    assert result["status"] == "success"


def test_normalizer_rewrites_contact_structured_items_without_fields():
    normalized = Planner._normalize_plan_envelope(
        {
            "start_url": "https://example.org",
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "extract_structured_items", "args": {}, "save_as": "contact_info"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["address", "phone", "email", "contact_page_url"]},
        },
        "Открой https://example.org и найди контактные данные: адрес, телефон, email и ссылку на страницу контактов",
    )

    extract_step = normalized["steps"][2]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "semantic_region_fields"
    assert set(extract_step["args"]["fields"]) == {"address", "phone", "email", "contact_page_url"}


def test_runtime_code_has_no_site_specific_contact_strings():
    runtime_code = Path("app/executor/action_handlers.py").read_text().casefold()

    assert "spbstu" not in runtime_code
