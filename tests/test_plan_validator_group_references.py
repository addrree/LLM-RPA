import pytest

from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator


def test_validator_rejects_extract_pattern_nonexistent_group_index():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_pattern_from_page_text",
                    "args": {"pattern": r"Users:\s*(\d+)", "group_index": 2},
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)


def test_validator_rejects_extract_structured_items_field_nonexistent_group_index():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["items"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": r"([A-Z])\s+(\d+)",
                        "limit": 2,
                        "fields": {"name": {"group_index": 3}},
                    },
                    "save_as": "items",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)
