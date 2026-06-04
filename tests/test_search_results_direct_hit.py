from __future__ import annotations

from pathlib import Path

from app.executor.action_handlers import ActionHandlers


def test_empty_generic_result_fallback_does_not_invent_current_page_item():
    items = ActionHandlers._collect_result_like_items_from_text(
        source_text="No repeated result blocks are visible.",
        links=[],
        limit=5,
    )

    assert items == []


def test_no_search_specific_runtime_collector_remains():
    source = Path("app/executor/action_handlers.py").read_text(encoding="utf-8")

    assert "def _collect_search_results_by_intent" not in source
