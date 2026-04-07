import json
from pathlib import Path


def test_extraction_smoke_scenarios_cover_stage_targets():
    path = Path("tests/fixtures/extraction_smoke_scenarios.json")
    scenarios = json.loads(path.read_text(encoding="utf-8"))

    by_id = {scenario["id"]: scenario for scenario in scenarios}
    assert {"wiki_english_articles", "wiki_russian_articles", "wiki_top_10_languages"} <= set(by_id)

    top = by_id["wiki_top_10_languages"]
    assert top["recommended_action"] == "extract_items"
    assert set(top["expected_item_fields"]) == {"language_name", "article_count"}
