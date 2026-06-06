from datetime import datetime, timezone

from app.planner.replanner import Replanner
from app.schemas.page_snapshot import PageSnapshot


def test_replanner_compacts_large_page_snapshot_for_prompt():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="Example",
        screenshot_path="artifacts/screenshots/example.png",
        page_text_excerpt="x" * 6000,
        page_text="y" * 50000,
        visible_links=[{"text": str(index), "href": f"https://example.org/{index}"} for index in range(200)],
        text_lines=[str(index) for index in range(500)],
        timestamp=datetime.now(timezone.utc),
    )

    compact = Replanner._compact_page_snapshot_for_prompt(snapshot)

    assert len(compact["page_text_excerpt"]) < 2100
    assert len(compact["page_text"]) < 6100
    assert len(compact["visible_links"]) == 50
    assert len(compact["text_lines"]) == 100


def test_replanner_compacts_nested_rows_tables_and_drops_bbox_noise():
    huge_row = {
        "row_id": "row_1",
        "tag": "div",
        "role": "",
        "className": "x" * 2000,
        "text": "row text " * 300,
        "selector": "main > div",
        "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
        "cells": [{"text": "cell " * 100, "bbox": {"x": 1}} for _ in range(40)],
        "links": [{"text": "link " * 100, "href": "https://www.python.org/downloads/", "bbox": {"x": 1}} for _ in range(40)],
    }
    snapshot = PageSnapshot(
        url="https://www.python.org/downloads/",
        title="Python Downloads",
        screenshot_path="artifacts/screenshots/python.png",
        page_text_excerpt="Download Python " * 1000,
        page_text="Latest Python release " * 8000,
        rows=[huge_row for _ in range(200)],
        tables=[{"caption": "downloads", "headers": ["Version"], "rows": [huge_row for _ in range(80)]}],
        visible_links=[{"text": "Download Python", "href": "https://www.python.org/downloads/", "bbox": {"x": 1}}],
        timestamp=datetime.now(timezone.utc),
    )

    compact = Replanner._compact_page_snapshot_for_prompt(snapshot)

    assert "bbox" not in str(compact)
    assert len(compact["rows"]) == 10
    assert len(compact["rows"][0]["cells"]) == 8
    assert len(compact["rows"][0]["links"]) == 4
    assert len(compact["tables"][0]["rows"]) == 10
    assert len(str(compact)) < 60000


def test_replanner_compacts_execution_result_extracted_snapshot():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="Example",
        screenshot_path="artifacts/screenshots/example.png",
        page_text_excerpt="x" * 8000,
        page_text="y" * 50000,
        visible_links=[{"text": str(index), "href": f"https://example.org/{index}"} for index in range(200)],
        timestamp=datetime.now(timezone.utc),
    )
    execution_result = {
        "status": "success",
        "extracted_data": {
            "page_snapshot": snapshot.model_dump(mode="json"),
            "description": "A useful extracted description.",
            "huge_text": "z" * 100000,
        },
        "logs": [{"message": "noise"} for _ in range(500)],
    }

    compact = Replanner._compact_execution_result_for_prompt(execution_result)

    assert "logs" not in compact
    assert compact["extracted_data"]["description"] == "A useful extracted description."
    assert len(compact["extracted_data"]["page_snapshot"]["page_text"]) < 6100
    assert len(compact["extracted_data"]["huge_text"]) < 900
    assert len(str(compact)) < 20000
