import asyncio

from app.executor.action_handlers import ActionHandlers


class _AnchorObjectPage:
    async def evaluate(self, _script, payload=None):
        anchors = [str(item).casefold() for item in (payload or {}).get("anchors", [])]
        rows = [
            {
                "label": "English",
                "name": "English",
                "title": "English",
                "value": "7 022 000+ articles",
                "count": "7 022 000+ articles",
                "number": "7 022 000+ articles",
                "href": "https://en.sample.test/",
                "link": "https://en.sample.test/",
                "selector": "main > a:nth-of-type(1)",
                "raw_text": "English 7 022 000+ articles",
            },
            {
                "label": "Francais",
                "name": "Francais",
                "title": "Francais",
                "value": "2 103 000+ articles",
                "count": "2 103 000+ articles",
                "number": "2 103 000+ articles",
                "href": "https://fr.sample.test/",
                "link": "https://fr.sample.test/",
                "selector": "main > a:nth-of-type(2)",
                "raw_text": "Francais 2 103 000+ articles",
            },
        ]
        if anchors:
            rows = [row for row in rows if any(anchor in row["raw_text"].casefold() for anchor in anchors)]
        return rows


async def _not_blocked(*_args, **_kwargs):
    return None


def test_anchor_object_maps_label_and_count_fields_without_known_answers():
    handler = ActionHandlers()
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    result = asyncio.run(
        handler.extract_by_intent(
            _AnchorObjectPage(),
            {
                "intent": "anchor_object",
                "anchor_candidates": ["Francais"],
                "fields": {
                    "language_name": {"type": "text"},
                    "article_count": {"type": "number"},
                },
            },
            {},
        )
    )

    assert result["language_name"] == "Francais"
    assert result["article_count"] == "2 103 000+ articles"
    assert result["raw_item"]["label"] == "Francais"
