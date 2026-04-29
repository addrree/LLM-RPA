from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context


def test_obs_with_text_title_url():
    ctx = browsergym_obs_to_page_context({"url": "https://example.com", "title": "Example", "text": "Hello"})
    assert ctx["url"] == "https://example.com"
    assert ctx["title"] == "Example"
    assert ctx["text"] == "Hello"


def test_obs_without_text_fallback():
    ctx = browsergym_obs_to_page_context({"url": "u", "payload": {"a": 1}})
    assert isinstance(ctx["text"], str)
    assert len(ctx["text"]) > 0


def test_obs_unknown_keys_not_crash():
    ctx = browsergym_obs_to_page_context({"foo": "bar", "x": 1}, {"meta": True})
    assert "foo" in ctx["obs_keys"]
