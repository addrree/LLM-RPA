from app.orchestrator.workflow_manager import WorkflowManager
from app.schemas.task_spec import TaskSpec


def _plan_without_open_url():
    return TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {
                    "step_id": 1,
                    "action": "extract_pattern_from_page_text",
                    "args": {"pattern": r"English\s+([0-9][0-9\s,\.\u00A0\u202F\+]*)"},
                    "save_as": "count",
                },
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )


def test_ensure_open_url_for_final_plan_injects_step():
    normalized = WorkflowManager._ensure_open_url_for_final_plan(_plan_without_open_url())

    assert normalized.steps[0].action == "open_url"
    assert normalized.steps[0].args["url"] == "https://www.wikipedia.org/"
    assert [step.step_id for step in normalized.steps] == [1, 2, 3]


def test_ensure_open_url_for_final_plan_keeps_existing_open_url():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org"}},
                {"step_id": 2, "action": "finish", "args": {}},
            ],
        }
    )

    normalized = WorkflowManager._ensure_open_url_for_final_plan(plan)
    assert len(normalized.steps) == 2
    assert normalized.steps[0].action == "open_url"
