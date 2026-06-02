from __future__ import annotations

from app.executor.action_handlers import ActionHandlers


def test_price_normalization_accepts_pound_prices():
    assert ActionHandlers._normalize_price_token("£51.77") == 51.77
