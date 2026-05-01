from app.browsergym_integration.config import WEBARENA_REQUIRED_ENV_VARS


def test_missing_wa_vars(monkeypatch):
    for k in WEBARENA_REQUIRED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    missing = [k for k in WEBARENA_REQUIRED_ENV_VARS if not __import__("os").getenv(k)]
    assert sorted(missing) == sorted(WEBARENA_REQUIRED_ENV_VARS)
