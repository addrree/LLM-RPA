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


def test_validator_rejects_extract_structured_items_int_field_nonexistent_group_index():
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
                        "fields": {"name": 3},
                    },
                    "save_as": "items",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)


def test_validator_rejects_too_broad_click_selector():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"selector": "a"}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)


def test_validator_rejects_weak_text_only_click_target():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "click", "args": {"text": "More"}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)


def test_validator_rejects_invalid_anchor_matching_mode_for_extract_value_near_anchor():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_value_near_anchor",
                    "args": {
                        "anchor_text": "Email",
                        "value_type": "number",
                        "anchor_candidates": ["Email", "Contact"],
                        "anchor_matching_mode": "fuzzy",
                    },
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)


def test_validator_accepts_anchor_candidates_without_anchor_text_for_email_value_type():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_value_near_anchor",
                    "args": {
                        "anchor_candidates": ["Contact", "Support", "Email"],
                        "value_type": "email",
                        "anchor_matching_mode": "auto",
                        "page_language": "en",
                    },
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    PlanValidator().validate(plan)


def test_validator_rejects_typed_anchor_extraction_group_out_of_range():
    plan = TaskSpec.model_validate(
        {
            "goal": "g",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "max_verification_retries": 3, "timeout_sec": 20},
            "expected_result": {"description": "d", "required_fields": ["value"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_value_near_anchor",
                    "args": {
                        "anchor_candidates": ["Contact"],
                        "value_type": "email",
                        "group_index": 2,
                    },
                    "save_as": "value",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    with pytest.raises(PlanValidationError):
        PlanValidator().validate(plan)
