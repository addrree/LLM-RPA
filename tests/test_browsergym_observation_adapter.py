import json

from app.browsergym_integration.local_extractor import extract_text_from_observation
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


def test_obs_unknown_keys_not_crash():
    ctx = browsergym_obs_to_page_context({"foo": "bar", "x": 1}, {"meta": True})
    assert "foo" in ctx["obs_keys"]


def test_obs_with_ndarray_screenshot_safe_summary():
    obs = {"url": "https://example.com", "screenshot": _FakeArray((10, 10, 3), "uint8")}
    ctx = browsergym_obs_to_page_context(obs, {})
    assert ctx["screenshot"] is None
    assert ctx["screenshot_summary"]["shape"] == (10, 10, 3)
    assert ctx["screenshot_summary"]["dtype"] == "uint8"


def test_observation_summary_sanitizes_numpy():
    obs = {"url": "https://example.com", "screenshot": _FakeArray((2, 2, 3), "uint8"), "open_pages_titles": ("Welcome to Python.org",)}
    ctx = browsergym_obs_to_page_context(obs, {})
    dumped = json.dumps(ctx, default=str)
    assert "array(" not in dumped


def test_smoke_extract_text_does_not_return_raw_observation():
    obs_summary = {
        "title": "Welcome to Python.org",
        "open_pages_titles": ["Welcome to Python.org"],
        "text_excerpt": "Welcome to Python.org\nStart here",
        "axtree_excerpt": "h1: Welcome to Python.org",
    }
    result = extract_text_from_observation(obs_summary, selector="h1", goal="Find the main heading")
    assert "array(" not in result
    assert "screenshot" not in result.lower()
    assert "Welcome" in result


def test_observation_adapter_extracts_miniwob_clickable_candidates_from_axtree():
    ctx = browsergym_obs_to_page_context(
        {"axtree_txt": '[7] button "submit"\n[8] link "cancel"', "goal": 'Click on the "submit" button.'},
        {},
    )
    assert ctx["clickable_candidates_count"] >= 2
    assert ctx["clickable_candidates"][0]["bid"] == "7"
    assert ctx["clickable_candidates"][0]["name"] == "submit"


def test_parse_axtree_object_extracts_candidate():
    ctx = browsergym_obs_to_page_context({"axtree_object": {"role": "button", "name": "Submit", "bid": "a12"}}, {})
    assert {"bid": "a12", "role": "button", "name": "Submit"}.items() <= ctx["clickable_candidates"][0].items()


def test_page_clickable_candidates_are_included():
    ctx = browsergym_obs_to_page_context({"page_clickable_candidates": [{"tag": "button", "text": "Submit", "center_x": 10, "center_y": 20}]}, {})
    assert ctx["clickable_candidates"][0]["text"] == "Submit"


def test_observation_adapter_preserves_browsergym_scaled_candidate_fields_and_safe_scalars():
    ctx = browsergym_obs_to_page_context(
        {
            "page_clickable_candidates": [
                {
                    "tag": "button",
                    "text": "Okay",
                    "center_x": 25.5,
                    "center_y": 147.5,
                    "page_center_x": 25.5,
                    "page_center_y": 147.5,
                    "browsergym_center_x": 38.25,
                    "browsergym_center_y": 221.25,
                    "browsergym_scale_factor": 1.5,
                    "coordinate_space": "page_css",
                    "action_coordinate_space": "browsergym_scaled",
                    "browsergym_bbox": {"x": 30, "y": 210, "width": 20, "height": 20},
                    "custom_score": 7,
                    "raw_dom": "<html>" * 200,
                }
            ]
        },
        {},
    )
    candidate = ctx["clickable_candidates"][0]
    assert candidate["page_center_x"] == 25.5
    assert candidate["browsergym_center_x"] == 38.25
    assert candidate["browsergym_center_y"] == 221.25
    assert candidate["browsergym_scale_factor"] == 1.5
    assert candidate["action_coordinate_space"] == "browsergym_scaled"
    assert candidate["browsergym_bbox"] == {"x": 30, "y": 210, "width": 20, "height": 20}
    assert candidate["custom_score"] == 7
    assert "raw_dom" not in candidate
