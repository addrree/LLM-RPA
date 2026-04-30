from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context


class _FakeArray:
    def __init__(self, shape=(10, 10, 3), dtype="uint8"):
        self.shape = shape
        self.dtype = dtype

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous.")


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


def test_obs_with_ndarray_screenshot_safe_summary():
    obs = {"url": "https://example.com", "screenshot": _FakeArray((10, 10, 3), "uint8")}
    ctx = browsergym_obs_to_page_context(obs, {})
    assert ctx["screenshot"] is None
    assert ctx["screenshot_summary"]["shape"] == (10, 10, 3)
    assert ctx["screenshot_summary"]["dtype"] == "uint8"


def test_obs_with_ndarray_not_using_bool_context():
    obs = {"text": "", "screenshot": _FakeArray((2, 2, 3), "uint8")}
    ctx = browsergym_obs_to_page_context(obs, {"text": "fallback"})
    assert ctx["text"] == ""
