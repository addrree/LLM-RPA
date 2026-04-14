import asyncio

from app.executor.action_handlers import ActionHandlers
from app.orchestrator.workflow_manager import WorkflowManager
from app.schemas.task_spec import TaskSpec


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


def test_stability_smoke_single_value_extraction():
    page = _FakePage("Users: 1250")
    handler = ActionHandlers()
    value = asyncio.run(
        handler.extract_pattern_from_page_text(
            page,
            {"pattern": r"Users:\s*(\d+)", "group_index": 1},
            runtime_state={},
        )
    )
    assert value == "1250"


def test_stability_smoke_anchored_value_extraction():
    text = "English 7,141,000+ articles"
    page = _FakePage(
        text,
        evaluate_payload=[
            {
                "source": "dom_same_block",
                "window_text": text,
                "anchor_idx_in_window": text.index("English"),
            }
        ],
    )
    handler = ActionHandlers()
    value = asyncio.run(
        handler.extract_value_near_anchor(
            page,
            {
                "anchor_text": "English",
                "value_pattern": r"([0-9][0-9,\+]*)",
                "group_index": 1,
            },
            runtime_state={},
        )
    )
    assert value == "7,141,000+"


def test_stability_smoke_repeated_structured_extraction():
    page = _FakePage("A 10\nB 20")
    handler = ActionHandlers()
    items = asyncio.run(
        handler.extract_pattern_from_page_text(
            page,
            {
                "pattern": r"([A-Z])\s+(\d+)",
                "limit": 2,
                "fields": {"name": 1, "count": {"group_index": 2, "normalize_number": True}},
            },
            runtime_state={},
        )
    )
    assert items == [{"name": "A", "count": 10}, {"name": "B", "count": 20}]


def test_stability_smoke_navigation_then_extraction_plan_normalization():
    raw = TaskSpec.model_validate(
        {
            "goal": "open and observe",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "observe", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {}},
                {"step_id": 2, "action": "observe_page", "args": {}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    normalized = WorkflowManager._normalize_plan_for_validation(raw)
    assert normalized.steps[0].args["url"] == "https://example.com/"
    assert normalized.steps[1].save_as == "page_snapshot"
