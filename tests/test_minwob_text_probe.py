from scripts.probe_minwob_text_api import _run_method
from app.browsergym_integration.miniwob_grounding import (
    extract_textbox_candidates_from_observation,
    ground_miniwob_action,
    map_login_textboxes,
    real_candidate_bid,
)
from app.browsergym_integration.runner import BrowserGymRunner


def test_probe_extracts_textbox_from_axtree_role_textbox_with_bid():
    obs = {
        "axtree_object": {
            "nodes": [
                {"role": {"value": "textbox"}, "bid": "14", "name": "Answer", "backendDOMNodeId": 101}
            ]
        }
    }

    candidates = extract_textbox_candidates_from_observation(obs, {})

    assert len(candidates) == 1
    assert real_candidate_bid(candidates[0]) == "14"
    assert candidates[0]["role"] == "textbox"
    assert candidates[0]["backendDOMNodeId"] == "101"


def test_probe_extracts_input_bid_from_dom_object():
    obs = {
        "dom_object": {
            "nodes": [
                {"nodeName": "INPUT", "attributes": ["type", "text", "bid", "16", "id", "plain-dom-id"], "inputValue": ""}
            ]
        }
    }

    candidates = extract_textbox_candidates_from_observation(obs, {})

    assert len(candidates) == 1
    assert real_candidate_bid(candidates[0]) == "16"
    assert candidates[0]["input_type"] == "text"
    assert candidates[0].get("id") != "plain-dom-id"


class _FakeEnv:
    def __init__(self, obs):
        self.obs = obs
        self.actions = []

    def reset(self, seed=None):
        return self.obs, {}

    def step(self, action):
        self.actions.append(action)
        return self.obs, 0, False, False, {}


def test_probe_does_not_click_submit_when_textbox_missing(monkeypatch):
    env = _FakeEnv({"goal": 'Enter "Alice" into the text field and press Submit.'})
    monkeypatch.setattr(
        BrowserGymRunner,
        "_extract_page_clickable_candidates",
        classmethod(lambda cls, env: ([{"bid": "15", "bid_source": "bid", "role": "button", "name": "Submit"}], False)),
    )

    result = _run_method(env, env_id="browsergym/miniwob.enter-text", seed=1, method="fill_then_submit")

    assert result["actions"] == []
    assert result["error"] == "missing textbox bid"
    assert result["terminated"] is False
    assert env.actions == []


def test_login_parser_maps_first_textbox_to_username_and_second_to_password():
    first = {"bid": "16", "bid_source": "bid", "role": "textbox"}
    second = {"bid": "17", "bid_source": "bid", "role": "textbox", "input_type": "password"}

    mapped = map_login_textboxes('Enter the username "u" and the password "p".', [first, second])

    assert mapped == {"username": first, "password": second}


def test_fill_action_is_preserved_and_not_converted_to_click():
    result = ground_miniwob_action(
        action='fill("14", "Alice")',
        parsed_response={},
        candidates=[{"bid": "14", "bid_source": "bid", "role": "textbox"}],
    )

    assert result.action == 'fill("14", "Alice")'
    assert result.mapping_strategy == "bid_fill"


def test_click_textbox_for_text_instruction_normalizes_to_fill():
    result = ground_miniwob_action(
        action='click("14", "left")',
        parsed_response={"miniwob_instruction": 'Enter "Alice" into the text field and press Submit.'},
        candidates=[{"bid": "14", "bid_source": "bid", "role": "textbox"}],
    )

    assert result.action == 'fill("14", "Alice")'
    assert result.mapping_strategy == "bid_fill"
