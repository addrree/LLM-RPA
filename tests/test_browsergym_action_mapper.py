import pytest

from app.browsergym_integration.action_mapper import browsergym_finish_action, task_step_to_browsergym_action
from app.browsergym_integration.errors import UnsupportedBrowserGymActionError


def test_click_text_mapping():
    assert "click" in task_step_to_browsergym_action({"action": "click", "args": {"text": "Docs"}})


def test_click_href_contains_mapping():
    action = task_step_to_browsergym_action({"action": "click", "args": {"href_contains": "pricing"}})
    assert "href_contains" in action


def test_type_mapping():
    action = task_step_to_browsergym_action({"action": "type", "args": {"selector": "#q", "text": "hi"}})
    assert "type(" in action


def test_unsupported_action():
    with pytest.raises(UnsupportedBrowserGymActionError):
        task_step_to_browsergym_action({"action": "open_url", "args": {"url": "https://x"}})


def test_finish_action_uses_final_answer_only():
    assert browsergym_finish_action("Welcome to Python.org") == "finish(answer='Welcome to Python.org')"
