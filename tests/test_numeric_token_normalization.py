from __future__ import annotations

from app.executor.action_handlers import ActionHandlers


def test_numeric_normalization_extracts_number_from_decorated_text():
    assert ActionHandlers._normalize_numeric_token("value: 51.77 units") == 51.77


def test_numeric_upper_bound_uses_generic_constraint():
    assert ActionHandlers._extract_numeric_upper_bound(args={"numeric_upper_bound": "1 250"}, runtime_state={}) == 1250
