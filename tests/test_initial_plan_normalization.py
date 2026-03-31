from app.planner.planner import Planner
from app.schemas.task_spec import TaskSpec


def test_normalize_initial_plan_to_full_taskspec_shape():
    raw_plan = {
        "steps": [
            {"action": "open_url", "url": "https://example.com/catalog"},
            {"action": "observe_page", "save_as": "page_snapshot"},
        ],
        "expected_result": {"required_fields": ["page_snapshot"]},
    }

    normalized = Planner._normalize_initial_plan(raw_plan, "Collect product cards")
    plan = TaskSpec.model_validate(normalized)

    assert plan.goal == "Collect product cards"
    assert str(plan.start_url) == "https://example.com/catalog"
    assert plan.allowed_domains == ["example.com"]
    assert plan.expected_result.description == "Collect page snapshot for replanning"
    assert [step.step_id for step in plan.steps] == [1, 2, 3]
    assert plan.steps[0].args["url"] == "https://example.com/catalog"
    assert plan.steps[-1].action == "finish"


def test_normalize_initial_plan_fills_observe_page_save_as_and_required_fields():
    raw_plan = {
        "steps": [
            {"action": "open_url", "url": "https://www.wikipedia.org"},
            {"action": "observe_page"},
            {"action": "finish"},
        ]
    }

    normalized = Planner._normalize_initial_plan(raw_plan, "Observe wikipedia landing page")
    plan = TaskSpec.model_validate(normalized)

    assert plan.steps[0].args["url"] == "https://www.wikipedia.org"
    assert plan.steps[1].action == "observe_page"
    assert plan.steps[1].save_as == "page_snapshot"
    assert [step.step_id for step in plan.steps] == [1, 2, 3]
    assert "page_snapshot" in plan.expected_result.required_fields
    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.constraints.max_steps == 4
    assert plan.constraints.max_replans == 1
    assert plan.constraints.timeout_sec == 30
    assert plan.goal == "Observe wikipedia landing page"
