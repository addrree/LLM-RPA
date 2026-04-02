import re
import asyncio

import pytest

from app.executor.action_handlers import ActionHandlers


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2 087 000+", 2087000),
        ("2,087,000+", 2087000),
        ("2.087.000+", 2087000),
        ("2\u00A0087\u00A0000+", 2087000),
        ("2\u202F087\u202F000+", 2087000),
        ("2087000", 2087000),
    ],
)
def test_normalize_number_token_grouped_integer_variants(raw_value, expected):
    assert (
        ActionHandlers._normalize_number_token(raw_value, number_type="int", strip_plus=True) == expected
    )


def test_normalize_number_token_raises_for_non_integer_like():
    with pytest.raises(ValueError):
        ActionHandlers._normalize_number_token("2,087,000.5+", number_type="int", strip_plus=True)


def test_extract_match_value_uses_group_index():
    match = next(iter(re.finditer(r"Русский\s*\n?\s*([0-9][0-9\s,\.]*)", "Русский\n2 087 000+")))
    assert ActionHandlers._extract_match_value(match, group_index=1).startswith("2 087 000")


def test_extract_pattern_match_not_found():
    text = "Русский\nno number"
    pattern = r"Русский\s*\n?\s*([0-9][0-9\s,\.]*)"
    assert list(re.finditer(pattern, text)) == []


class _FakeBodyLocator:
    def __init__(self, text: str):
        self._text = text

    async def inner_text(self):
        return self._text


class _FakePage:
    def __init__(self, text: str, evaluate_payload: list[dict] | None = None):
        self._text = text
        self._evaluate_payload = evaluate_payload or []

    def locator(self, selector: str):
        assert selector == "body"
        return _FakeBodyLocator(self._text)

    async def evaluate(self, _script, _payload):
        return self._evaluate_payload


def test_extract_pattern_returns_raw_value_when_normalization_disabled():
    page = _FakePage("Русский\n2 087 000+")
    handler = ActionHandlers()
    args = {"pattern": r"Русский\s*\n?\s*([0-9][0-9\s,\.]+\+?)", "group_index": 1, "normalize_number": False}

    value = asyncio.run(handler.extract_pattern_from_page_text(page, args, runtime_state={}))
    assert value == "2 087 000+"


def test_extract_pattern_raises_when_match_not_found():
    page = _FakePage("Русский\nнет числа")
    handler = ActionHandlers()
    with pytest.raises(ValueError):
        asyncio.run(
            handler.extract_pattern_from_page_text(
                page,
                {"pattern": r"Русский\s*\n?\s*([0-9][0-9\s,\.]+\+?)", "group_index": 1},
                runtime_state={},
            )
        )


def test_extract_text_near_text_returns_normalized_number():
    page = _FakePage("English\n6 987 000+ articles")
    handler = ActionHandlers()
    value = asyncio.run(
        handler.extract_text_near_text(
            page,
            {
                "anchor_text": "English",
                "pattern": r"English\s*\n?\s*([0-9][0-9\s,\.\u00A0\u202F\+]*)",
                "group_index": 1,
                "normalize_number": True,
                "number_type": "int",
                "strip_plus": True,
            },
            runtime_state={},
        )
    )
    assert value == 6987000


def test_extract_value_near_anchor_smoke_prefers_contextual_match():
    observed_text = "25 years of the free encyclopedia ... English ... 7,141,000+ articles"
    page = _FakePage(
        observed_text,
        evaluate_payload=[
            {
                "source": "dom_same_block",
                "window_text": observed_text,
                "anchor_idx_in_window": observed_text.index("English"),
            }
        ],
    )
    handler = ActionHandlers()
    value = asyncio.run(
        handler.extract_value_near_anchor(
            page,
            {
                "anchor_text": "English",
                "value_pattern": r"([0-9][0-9\s,\.\u00A0\u202F\+]*)",
                "search_direction": "after",
                "same_block_only": True,
                "required_right_context": "articles",
                "group_index": 1,
                "normalize_number": True,
                "number_type": "int",
                "strip_plus": True,
            },
            runtime_state={},
        )
    )
    assert value == 7141000
