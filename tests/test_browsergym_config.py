from app.browsergym_integration.config import validate_webarena_env_vars


def test_missing_webarena_env_vars_returns_structured_message(monkeypatch):
    for key in [
        "WA_SHOPPING",
        "WA_SHOPPING_ADMIN",
        "WA_REDDIT",
        "WA_GITLAB",
        "WA_WIKIPEDIA",
        "WA_MAP",
        "WA_HOMEPAGE",
    ]:
        monkeypatch.delenv(key, raising=False)
    result = validate_webarena_env_vars("browsergym/webarena.10")
    assert result["ok"] is False
    assert "WA_SHOPPING" in result["missing"]
    assert "WebArena" in result["message"]
