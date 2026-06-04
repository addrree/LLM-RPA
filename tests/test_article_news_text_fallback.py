from __future__ import annotations

from pathlib import Path

from app.executor.action_handlers import ActionHandlers


def test_generic_text_result_fallback_extracts_requested_visible_blocks():
    source_text = """
    Results
    Alpha heading
    A meaningful description for the first visible result block.
    Beta heading
    A different meaningful description for the second visible result block.
    Privacy policy
    """
    links = [
        {"text": "Alpha heading", "href": "https://sample.test/alpha"},
        {"text": "Beta heading", "href": "https://sample.test/beta"},
    ]

    items = ActionHandlers._collect_result_like_items_from_text(source_text=source_text, links=links, limit=5)

    assert [item["title"] for item in items] == ["Alpha heading", "Beta heading"]
    assert [item["href"] for item in items] == ["https://sample.test/alpha", "https://sample.test/beta"]
    assert all(item["description"] for item in items)


def test_generic_item_projection_does_not_invent_known_profile_fields():
    projected = ActionHandlers._project_item_to_schema(
        item={
            "title": "Alpha heading",
            "description": "Visible summary",
            "href": "https://sample.test/alpha",
            "raw_text": "Alpha heading Visible summary 2026-05-10",
        },
        fields={"heading": {"type": "title"}, "summary": {"type": "description"}},
    )

    assert projected["heading"] == "Alpha heading"
    assert projected["summary"] == "Visible summary"
    assert "author" not in projected
    assert "publication_time" not in projected


def test_no_known_content_profile_helpers_remain():
    source = Path("app/executor/action_handlers.py").read_text(encoding="utf-8")

    for name in (
        "_collect_article_like_items_from_text",
        "_article_metadata_requested",
        "_filter_links_to_article_like_paths",
    ):
        assert f"def {name}" not in source
