from __future__ import annotations

import asyncio
from pathlib import Path

from app.executor.action_handlers import ActionHandlers
from app.planner.action_vocab import (
    build_field_schema_args,
    infer_schema_required_fields,
    normalize_plan_action_aliases,
    semantic_intent_for_structured_step,
)
from app.planner.prompts import build_profile_planner_prompt
from app.planner.task_router import PROFILES


class _MetaLocator:
    def __init__(self, value: str = ""):
        self.value = value
        self.first = self

    async def get_attribute(self, _name: str):
        return self.value

    async def inner_text(self):
        return self.value

    async def evaluate_all(self, _script: str):
        return []


class _ParagraphLocator(_MetaLocator):
    def __init__(self, paragraphs: list[str]):
        super().__init__("")
        self._paragraphs = paragraphs

    async def evaluate_all(self, _script: str):
        return list(self._paragraphs)


class _SchemaPage:
    url = "https://iana.org/domains/reserved"

    def __init__(self, title: str, description: str, paragraphs: list[str] | None = None):
        self._title = title
        self._description = description
        self._paragraphs = paragraphs or []

    async def title(self):
        return self._title

    def locator(self, selector: str):
        if selector == 'meta[name="description"]':
            return _MetaLocator(self._description)
        if selector.startswith("main p,"):
            return _ParagraphLocator(self._paragraphs)
        return _MetaLocator()


async def _not_blocked(*_args, **_kwargs):
    return None


def test_field_schema_builder_has_no_fixed_contact_profile():
    args = build_field_schema_args(
        "Return title, description, and URL",
        ["title", "description", "url"],
        output_key="page_details",
    )

    assert args["intent"] == "field_schema"
    assert set(args["fields"]) == {"title", "description", "url"}
    assert "region_candidates" not in args
    assert "contact_info" not in str(args)


def test_required_fields_are_inferred_from_arbitrary_explicit_lists():
    assert infer_schema_required_fields(
        "Extract serial number, owner label, and lifecycle state."
    ) == ["serial_number", "owner_label", "lifecycle_state"]
    assert infer_schema_required_fields(
        "Выгрузи поля: код, владелец и статус."
    ) == ["код", "владелец", "статус"]


def test_field_schema_extractor_supports_different_requested_objects():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    page = _SchemaPage("IANA-managed Reserved Domains", "Reserved names for documentation and testing.")

    page_details = asyncio.run(
        handler.extract_by_intent(
            page,
            {
                "intent": "field_schema",
                "fields": {
                    "heading": {"type": "page_title"},
                    "summary": {"type": "meta_description"},
                    "source": {"type": "current_url"},
                },
            },
            {},
        )
    )
    url_only = asyncio.run(
        handler.extract_by_intent(
            page,
            {"intent": "field_schema", "fields": {"canonical": {"type": "current_url"}}},
            {},
        )
    )

    assert page_details["heading"] == "IANA-managed Reserved Domains"
    assert page_details["summary"].startswith("Reserved names")
    assert page_details["source"] == page.url
    assert page_details["status"] == "success"
    assert url_only["canonical"] == page.url
    assert set(url_only["found_fields"]) == {"canonical"}


def test_field_schema_description_falls_back_to_first_meaningful_visible_paragraph():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]
    page = _SchemaPage(
        "Reference page",
        "",
        paragraphs=[
            "Short label",
            "This paragraph contains a meaningful general description of the current page and its contents.",
        ],
    )

    result = asyncio.run(
        handler.extract_by_intent(
            page,
            {"intent": "field_schema", "fields": {"overview": {"type": "description"}}},
            {},
        )
    )

    assert result["overview"].startswith("This paragraph contains")
    assert result["status"] == "success"


def test_task_profile_intents_are_not_silently_normalized():
    payload, oov = normalize_plan_action_aliases(
        {
            "steps": [
                {"action": "extract_by_intent", "args": {"intent": "paper_results"}},
                {"action": "extract_by_intent", "args": {"intent": "product_cards"}},
                {"action": "extract_by_intent", "args": {"intent": "repository_results"}},
            ]
        }
    )

    assert oov is False
    assert [step["args"]["intent"] for step in payload["steps"]] == [
        "paper_results",
        "product_cards",
        "repository_results",
    ]
    assert all("item_type_hint" not in step["args"] for step in payload["steps"])
    assert all("numeric_value_required" not in step["args"] for step in payload["steps"])


def test_structural_intent_inference_uses_shape_and_schema_not_domain_words():
    assert semantic_intent_for_structured_step(
        {"action": "extract_structured_items", "args": {"shape": "table", "fields": {"a": 1}}}
    ) == "table_rows"
    assert semantic_intent_for_structured_step(
        {"action": "extract_structured_items", "args": {"fields": {"heading": {"type": "page_title"}}}}
    ) == "field_schema"
    assert semantic_intent_for_structured_step(
        {"action": "extract_structured_items", "args": {"fields": {"title": 1, "description": 2}}}
    ) == "card_items"


def test_item_projection_changes_export_fields_without_site_logic():
    item = {
        "title": "Alpha",
        "description": "First description",
        "href": "https://sample.test/a",
        "selector": "main > article:nth-of-type(1)",
        "raw_text": "Alpha First description Score: 42",
    }

    compact = ActionHandlers._project_item_to_schema(
        item=item,
        fields={"name": {"type": "title"}, "link": {"type": "url"}},
    )
    detailed = ActionHandlers._project_item_to_schema(
        item=item,
        fields={
            "summary": {"type": "description"},
            "score": {"value_pattern": r"Score:\s*(\d+)", "group_index": 1},
        },
    )

    assert compact["name"] == "Alpha"
    assert compact["link"] == "https://sample.test/a"
    assert "summary" not in compact
    assert detailed["summary"] == "First description"
    assert detailed["score"] == "42"
    assert detailed["selector"]
    assert detailed["raw_text"]


def test_shared_description_is_removed_from_distinct_items():
    items = [
        {"title": "Alpha", "description": "Shared section heading", "snippet": "Shared section heading"},
        {"title": "Beta", "description": "Shared section heading", "snippet": "Shared section heading"},
        {"title": "Gamma", "description": "Unique item description", "snippet": "Unique item description"},
    ]

    cleaned = ActionHandlers._remove_shared_item_descriptions(items)

    assert "description" not in cleaned[0]
    assert "snippet" not in cleaned[1]
    assert cleaned[2]["description"] == "Unique item description"


def test_no_task_or_site_specific_collector_methods_remain():
    source = Path("app/executor/action_handlers.py").read_text(encoding="utf-8")

    for name in (
        "_collect_product_cards_generic",
        "_collect_article_like_results_generic",
        "_collect_paper_like_results_generic",
        "_collect_repository_like_results_generic",
        "_extract_package_metadata_generic",
        "_article_links_requested",
        "_paper_results_requested",
        "_repository_results_requested",
        "_extract_version_like_token",
        "_extract_date_like_token",
        "_extract_release_like_title",
        "_first_email_candidate",
        "_first_phone_candidate",
        "_extract_typed_href_values",
    ):
        assert f"def {name}" not in source


def test_profile_prompt_exposes_only_generic_structural_intents():
    prompt = build_profile_planner_prompt(PROFILES["generic_web_task"])
    runtime_line = next(line for line in prompt.splitlines() if line.startswith("- preferred_runtime_intents:"))

    assert "field_schema" in runtime_line
    assert "card_items" in runtime_line
    assert "table_rows" in runtime_line
    assert "product_cards" not in runtime_line
    assert "paper_results" not in runtime_line
    assert "semantic_region_fields" not in runtime_line
