import asyncio
from datetime import datetime, timezone

from app.orchestrator.workflow_manager import WorkflowManager, sanitize_benchmark_context_for_llm
from app.planner.replanner import Replanner
from app.schemas.execution import ExecutionResult
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError


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


def test_sanitize_benchmark_context_for_llm_strips_expected_leakage_fields():
    context = {
        "task_family": "repeated_structured_items",
        "expected_pattern": "(.+)",
        "anchor_candidates": ["leak"],
        "evaluator_metadata": {
            "scenario_id": "x",
            "expected_heading": "leak",
            "expected_item_fields": ["name"],
        },
    }
    sanitized = sanitize_benchmark_context_for_llm(context)
    assert sanitized is not None
    assert "expected_pattern" not in sanitized
    assert "anchor_candidates" not in sanitized
    assert "expected_heading" not in sanitized["evaluator_metadata"]


def test_normalize_final_plan_fills_required_shape_from_context():
    previous_plan = TaskSpec.model_validate(
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
    snapshot = PageSnapshot(
        url="https://www.wikipedia.org/",
        title="Wikipedia",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Wikipedia",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url"},
                {"action": "extract_value_near_anchor", "args": {"anchor_text": "English"}, "save_as": "count"},
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["count"]},
        },
        user_goal="Extract English count",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
    )
    plan = TaskSpec.model_validate(normalized)

    assert plan.goal == "Extract count"
    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.constraints.max_steps == 5
    assert plan.expected_result.description == "Count"
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org/"
    assert plan.steps[1].args == {"anchor_text": "English"}
    assert [step.step_id for step in plan.steps] == [1, 2, 3]


def test_normalize_final_plan_rejects_malformed_llm_url_and_preserves_previous_url():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Extract metadata",
            "start_url": "https://docs.sample.test/docs/Web/API/Fetch_API",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Metadata", "required_fields": ["title"]},
            "steps": [
                {
                    "step_id": 1,
                    "action": "open_url",
                    "args": {"url": "https://docs.sample.test/docs/Web/API/Fetch_API"},
                },
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://docs.sample.test/docs/Web/API/Fetch_API",
        title="API",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="API documentation",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://docs.sample.test/docs/Web/API Fetch_API",
            "allowed_domains": ["docs.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://docs.sample.test/docs/Web/API Fetch_API"}},
                {"action": "extract_by_intent", "args": {"intent": "page_title"}, "save_as": "title"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["title"]},
        },
        user_goal="Extract metadata",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
    )

    assert normalized["start_url"] == "https://docs.sample.test/docs/Web/API/Fetch_API"
    assert normalized["steps"][0]["args"]["url"] == "https://docs.sample.test/docs/Web/API/Fetch_API"


def test_normalize_final_plan_metadata_fallback_respects_preferred_runtime_intents():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Extract page title, short description, and current URL.",
            "start_url": "https://docs.sample.test/docs/Web/API/Fetch_API",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {
                    "step_id": 1,
                    "action": "open_url",
                    "args": {"url": "https://docs.sample.test/docs/Web/API/Fetch_API"},
                },
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://docs.sample.test/docs/Web/API/Fetch_API",
        title="API",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="API documentation with table rows and columns",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal="Extract the page title, a short description, and the current URL.",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=["current_url", "page_title", "field_schema", "value_near_anchor"],
    )

    assert normalized["steps"][1]["action"] == "extract_by_intent"
    assert normalized["steps"][1]["args"]["intent"] == "field_schema"
    assert set(normalized["steps"][1]["args"]["fields"]) == {"page_title", "description", "current_url"}


def test_normalize_final_plan_builds_generic_navigation_fallback():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Open site and follow the Downloads link.",
            "start_url": "https://docs.sample.test/",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://docs.sample.test/"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://docs.sample.test/",
        title="Docs",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Downloads",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal="Open the site, follow the Downloads link, then extract the page title, current URL, and visible links.",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=["current_url", "page_title"],
    )

    actions = [step["action"] for step in normalized["steps"]]
    assert actions == [
        "open_url",
        "click_by_semantic_target",
        "observe_page",
        "extract_by_intent",
        "extract_by_intent",
        "extract_visible_links",
        "finish",
    ]
    assert normalized["steps"][1]["args"] == {"target_text": "Downloads", "role": "link"}


def test_normalize_final_plan_collection_fallback_adds_confident_condition():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Extract story cards whose title contains Arm.",
            "start_url": "https://docs.sample.test/cards",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://docs.sample.test/cards"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://docs.sample.test/cards",
        title="Story cards",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Story cards",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal="Extract story cards whose title contains Arm, with title and link.",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items"],
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["condition"] == {"title": "Arm"}


def test_normalize_final_plan_row_action_fallback_clicks_matching_row():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Delete a matching table row.",
            "start_url": "https://tasks.sample.test/list",
            "allowed_domains": ["tasks.sample.test"],
            "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://tasks.sample.test/list"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://tasks.sample.test/list",
        title="Tasks",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Hit the gym",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal="Open the page and delete the table row named Hit the gym.",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=["table_rows", "card_items"],
    )

    assert [step["action"] for step in normalized["steps"]] == [
        "open_url",
        "find_row_by_condition",
        "click_row_action",
        "finish",
    ]
    assert normalized["steps"][1]["args"]["condition"] == {"contains": "Hit the gym"}
    assert normalized["steps"][2]["args"] == {"action_name": "delete", "condition": {"contains": "Hit the gym"}}
    assert normalized["expected_result"]["required_fields"] == ["row_action"]


def test_normalize_final_plan_visual_count_fallback_builds_visual_actions():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Visually count center links.",
            "start_url": "https://visual.sample.test/",
            "allowed_domains": ["visual.sample.test"],
            "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://visual.sample.test/"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://visual.sample.test/",
        title="Visual",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal="Open the page and visually count the large language links visible near the center.",
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=[],
    )

    assert [step["action"] for step in normalized["steps"]] == [
        "open_url",
        "visual_observe",
        "visual_extract_object_count",
        "finish",
    ]
    assert normalized["steps"][2]["args"]["target"] == "link"
    assert normalized["steps"][2]["save_as"] == "language_link_count"
    assert normalized["expected_result"]["required_fields"] == ["language_link_count"]


def test_normalize_final_plan_repairs_click_row_action_args_from_goal():
    snapshot = PageSnapshot(
        url="https://tasks.sample.test/list",
        title="Tasks",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Hit the gym",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://tasks.sample.test/list",
            "allowed_domains": ["tasks.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://tasks.sample.test/list"}},
                {"action": "click_row_action", "args": {"action_name": "remove"}},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["row_action"]},
        },
        user_goal="Open the page and delete the table row named Hit the gym.",
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["table_rows", "card_items"],
    )

    assert normalized["steps"][1]["args"] == {"action_name": "remove", "condition": {"contains": "Hit the gym"}}
    assert normalized["steps"][1]["save_as"] == "row_action"


def test_normalize_final_plan_rewrites_semantic_row_click_to_row_action():
    snapshot = PageSnapshot(
        url="https://tasks.sample.test/list",
        title="Tasks",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Hit the gym",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://tasks.sample.test/list",
            "allowed_domains": ["tasks.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://tasks.sample.test/list"}},
                {"action": "click_by_semantic_target", "args": {"target": "delete", "target_text": "Hit the gym"}},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["row_action"]},
        },
        user_goal="Open the page and delete the table row named Hit the gym.",
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["table_rows", "card_items"],
    )

    assert normalized["steps"][1]["action"] == "click_row_action"
    assert normalized["steps"][1]["args"] == {"action_name": "delete", "condition": {"contains": "Hit the gym"}}
    assert normalized["steps"][1]["save_as"] == "row_action"


def test_normalize_final_plan_drops_bare_text_wait_for_row_action():
    snapshot = PageSnapshot(
        url="https://tasks.sample.test/list",
        title="Tasks",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Hit the gym",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://tasks.sample.test/list",
            "allowed_domains": ["tasks.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://tasks.sample.test/list"}},
                {"action": "click_row_action", "args": {"action_name": "delete", "condition": {"contains": "Hit the gym"}}},
                {"action": "wait_for", "args": {"text": "Hit the gym"}},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["row_action"]},
        },
        user_goal="Open the page and delete the table row named Hit the gym.",
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["table_rows", "card_items"],
    )

    assert "wait_for" not in [step["action"] for step in normalized["steps"]]


def test_normalize_final_plan_search_result_navigation_keeps_description_requirement():
    previous_plan = TaskSpec.model_validate(
        {
            "goal": "Open first result and extract details.",
            "start_url": "https://search.sample.test/?q=browser+automation",
            "allowed_domains": ["search.sample.test"],
            "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Snapshot", "required_fields": ["page_snapshot"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://search.sample.test/?q=browser+automation"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=browser+automation",
        title="Search results",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={},
        user_goal=(
            "Open https://search.sample.test/?q=browser+automation, open the first relevant repository result, "
            "then extract the opened page title, a short description, and the current URL."
        ),
        previous_plan=previous_plan,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items", "current_url", "page_title", "field_schema"],
    )

    assert [step["action"] for step in normalized["steps"]] == [
        "open_url",
        "click_by_semantic_target",
        "observe_page",
        "extract_by_intent",
        "extract_by_intent",
        "extract_by_intent",
        "finish",
    ]
    assert normalized["steps"][1]["args"] == {"target_text": "first relevant repository result", "role": "link"}
    assert normalized["steps"][4]["args"] == {
        "intent": "field_schema",
        "fields": {"description": {"type": "meta_description"}},
        "output_key": "page_metadata",
    }
    assert normalized["steps"][5]["args"] == {"intent": "current_url"}
    assert normalized["steps"][4]["save_as"] == "page_metadata"
    assert normalized["steps"][5]["save_as"] == "final_url"
    assert normalized["expected_result"]["required_fields"] == ["page_title", "final_url", "page_metadata"]


def test_normalize_final_plan_rewrites_dynamic_result_open_url_to_click():
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=browser+automation",
        title="Search results",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://search.sample.test/?q=browser+automation",
            "allowed_domains": ["search.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=browser+automation"}},
                {"action": "extract_by_intent", "args": {"intent": "card_items"}, "save_as": "items"},
                {"action": "open_url", "args": {"url": "search_results_list[0].href"}},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["final_url"]},
        },
        user_goal=(
            "Open https://search.sample.test/?q=browser+automation, open the first relevant result, "
            "then extract the current URL."
        ),
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items", "current_url", "page_title", "field_schema"],
    )

    assert normalized["steps"][2]["action"] == "click_by_semantic_target"
    assert normalized["steps"][2]["args"] == {"target_text": "first relevant result", "role": "link"}
    assert normalized["steps"][2]["save_as"] == "clicked_text"


def test_normalize_final_plan_rewrites_collection_description_after_navigation_to_metadata():
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=browser+automation",
        title="Search results",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://search.sample.test/?q=browser+automation",
            "allowed_domains": ["search.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=browser+automation"}},
                {"action": "click_by_semantic_target", "args": {"target_text": "first search result"}},
                {"action": "extract_by_intent", "args": {"intent": "card_items"}, "save_as": "description"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["description"]},
        },
        user_goal=(
            "Open https://search.sample.test/?q=browser+automation, open the first relevant result, "
            "then extract a short description."
        ),
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items", "current_url", "page_title", "field_schema"],
    )

    assert normalized["steps"][2]["action"] == "extract_by_intent"
    assert normalized["steps"][2]["args"] == {
        "intent": "field_schema",
        "fields": {"description": {"type": "meta_description"}},
        "output_key": "description",
    }
    assert normalized["steps"][2]["save_as"] == "description"


def test_normalize_final_plan_moves_final_url_after_navigating_metadata():
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=browser+automation",
        title="Search results",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://search.sample.test/?q=browser+automation",
            "allowed_domains": ["search.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=browser+automation"}},
                {"action": "click_by_semantic_target", "args": {"target_text": "first search result"}},
                {"action": "extract_by_intent", "args": {"intent": "current_url"}, "save_as": "final_url"},
                {
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "field_schema",
                        "fields": {"description": {"type": "meta_description"}},
                        "output_key": "description",
                    },
                    "save_as": "description",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["description", "final_url"]},
        },
        user_goal=(
            "Open https://search.sample.test/?q=browser+automation, open the first relevant result, "
            "then extract a short description and current URL."
        ),
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items", "current_url", "page_title", "field_schema"],
    )

    actions_and_saves = [(step["action"], step.get("save_as")) for step in normalized["steps"]]
    assert actions_and_saves.index(("extract_by_intent", "description")) < actions_and_saves.index(
        ("extract_by_intent", "final_url")
    )


def test_normalize_final_plan_drops_brittle_wait_after_first_result_click():
    snapshot = PageSnapshot(
        url="https://search.sample.test/?q=browser+automation",
        title="Search results",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Search results",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "start_url": "https://search.sample.test/?q=browser+automation",
            "allowed_domains": ["search.sample.test"],
            "steps": [
                {"action": "open_url", "args": {"url": "https://search.sample.test/?q=browser+automation"}},
                {"action": "click_by_semantic_target", "args": {"target_text": "first search result"}},
                {"action": "wait_for", "args": {"url_contains": "guessed/path"}},
                {"action": "extract_by_intent", "args": {"intent": "current_url"}, "save_as": "final_url"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["final_url"]},
        },
        user_goal=(
            "Open https://search.sample.test/?q=browser+automation, open the first relevant result, "
            "then extract the current URL."
        ),
        previous_plan=None,
        page_snapshot=snapshot,
        preferred_runtime_intents=["card_items", "current_url", "page_title", "field_schema"],
    )

    assert "wait_for" not in [step["action"] for step in normalized["steps"]]


def test_normalize_final_plan_maps_structured_nested_required_fields_to_save_as():
    snapshot = PageSnapshot(
        url="https://www.wikipedia.org/",
        title="Wikipedia",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="Wikipedia",
        timestamp=datetime.now(timezone.utc),
    )

    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "goal": "Top languages",
            "start_url": "https://www.wikipedia.org/",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {"description": "Top language blocks", "required_fields": ["language_name", "article_count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {
                    "step_id": 2,
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": r"([A-Za-zА-Яа-яЁё]+)\\s+([0-9][0-9\\s,\\.\\u00A0\\u202F\\+]*)",
                        "limit": 10,
                        "fields": {
                            "language_name": {"group_index": 1},
                            "article_count": {"group_index": 2, "normalize_number": True, "number_type": "int"},
                        },
                    },
                    "save_as": "language_blocks",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        },
        user_goal="Extract top 10 languages",
        previous_plan=None,
        page_snapshot=snapshot,
    )
    plan = TaskSpec.model_validate(normalized)

    assert plan.expected_result.required_fields == ["language_blocks"]


def test_corrective_replanner_blacklists_empty_heading():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="RFC Index",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="RFC Index",
        timestamp=datetime.now(timezone.utc),
        headings=[
            {"text": "Introduction", "level": "h2", "index": 0, "visible": True, "line_count_after": 0},
            {"text": "The RFC Series", "level": "h2", "index": 1, "visible": True, "line_count_after": 5},
            {"text": "RFC Editor", "level": "h2", "index": 2, "visible": True, "line_count_after": 4},
        ],
    )
    plan = {
        "goal": "compare sections",
        "start_url": "https://example.org",
        "allowed_domains": ["example.org"],
        "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 30},
        "expected_result": {"description": "Compare", "required_fields": ["combined_result"]},
        "steps": [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "extract_section_lines", "args": {"heading_text": "Introduction", "limit": 7}, "save_as": "source_a"},
            {"step_id": 4, "action": "extract_section_lines", "args": {"heading_text": "Introduction", "limit": 7}, "save_as": "source_b"},
            {"step_id": 5, "action": "compare_structured_values", "args": {"left_key": "source_a", "right_key": "source_b"}, "save_as": "combined_result"},
            {"step_id": 6, "action": "finish", "args": {}},
        ],
    }
    rewritten = Replanner._repair_empty_section_corrective_plan(
        normalized_plan=plan,
        page_snapshot=snapshot,
        failed_args={"heading_text": "Introduction"},
        failure_details={"reason": "empty_section", "failed_heading": "Introduction"},
        error_message="section heading found but extracted zero lines",
    )
    headings = [
        step["args"]["heading_text"]
        for step in rewritten["steps"]
        if step.get("action") == "extract_section_lines"
    ]
    assert "Introduction" not in headings
    assert headings[0] in {"The RFC Series", "RFC Editor"}


def test_corrective_replanner_blacklists_heading_from_failure_diagnostics():
    snapshot = PageSnapshot(
        url="https://example.org",
        title="RFC Index",
        screenshot_path="artifacts/screenshots/a.png",
        page_text_excerpt="RFC Index",
        timestamp=datetime.now(timezone.utc),
        headings=[
            {"text": "About", "level": "h2", "index": 0, "visible": True, "line_count_after": 0, "region": "nav"},
            {"text": "The RFC Series", "level": "h2", "index": 1, "visible": True, "line_count_after": 5, "region": "main"},
            {"text": "RFC Editor", "level": "h2", "index": 2, "visible": True, "line_count_after": 4, "region": "main"},
        ],
    )
    plan = {
        "goal": "compare sections",
        "start_url": "https://example.org",
        "allowed_domains": ["example.org"],
        "constraints": {"max_steps": 8, "max_replans": 1, "timeout_sec": 30},
        "expected_result": {"description": "Compare", "required_fields": ["combined_result"]},
        "steps": [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "extract_section_lines", "args": {"heading_text": "About", "limit": 7}, "save_as": "source_a"},
            {"step_id": 4, "action": "extract_section_lines", "args": {"heading_text": "About", "limit": 7}, "save_as": "source_b"},
            {"step_id": 5, "action": "compare_structured_values", "args": {"left_key": "source_a", "right_key": "source_b"}, "save_as": "combined_result"},
            {"step_id": 6, "action": "finish", "args": {}},
        ],
    }
    rewritten = Replanner._repair_empty_section_corrective_plan(
        normalized_plan=plan,
        page_snapshot=snapshot,
        failed_args={"heading_text": "About"},
        failure_details={
            "reason": "empty_section",
            "failed_heading": "About",
            "available_non_empty_headings": [
                {"text": "The RFC Series", "line_count_after": 5},
                {"text": "RFC Editor", "line_count_after": 4},
            ],
        },
        error_message="section heading found but extracted zero lines",
    )
    headings = [
        step["args"]["heading_text"]
        for step in rewritten["steps"]
        if step.get("action") == "extract_section_lines"
    ]
    assert "About" not in headings
    assert set(headings).issubset({"The RFC Series", "RFC Editor"})


class _FakePlanner:
    def build_initial_plan(self, user_goal: str) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                "expected_result": {"description": "Observe page", "required_fields": ["page_snapshot"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )

    last_artifact = None
    last_initial_artifact = None


class _FakeValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, plan: TaskSpec) -> None:
        self.calls += 1
        if self.calls == 2:
            raise PlanValidationError("extract_value_near_anchor requires non-empty 'value_pattern'")


class _RecordingValidator:
    def __init__(self):
        self.allowed_actions_history: list[set[str] | None] = []

    def validate(self, plan: TaskSpec, allowed_actions: set[str] | None = None) -> None:
        self.allowed_actions_history.append(allowed_actions)


class _FakeExecutor:
    async def _start_session(self):
        return {"id": "session"}

    async def _close_session(self, session):
        return None

    async def execute(self, plan: TaskSpec, session=None, runtime_state=None) -> ExecutionResult:
        if any(step.action == "observe_page" for step in plan.steps):
            return ExecutionResult(
                status="success",
                extracted_data={
                    "page_snapshot": {
                        "url": "https://example.com",
                        "title": "Example",
                        "screenshot_path": "artifacts/screenshots/s.png",
                        "page_text_excerpt": "Example Domain",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                },
                logs=[],
            )
        return ExecutionResult(status="success", extracted_data={"answer": "42"}, logs=[])


class _FakeVerifier:
    last_artifact = None

    class _Verdict:
        verdict = "accept"
        confidence = 1.0
        task_completed = True

        def model_dump(self, mode="json"):
            return {"verdict": "accept", "confidence": 1.0, "task_completed": True}

    def verify(self, plan, execution):
        return self._Verdict()


class _SingleStagePlanner:
    last_artifact = None
    last_initial_artifact = None
    last_action_oov_detected = False

    def build_plan(self, user_goal: str, benchmark_context=None) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 20},
                "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                    {"step_id": 3, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "value"},
                    {"step_id": 4, "action": "finish", "args": {}},
                ],
            }
        )


class _FakeReplanner:
    def __init__(self):
        self.calls = 0
        self.last_artifact = None

    def revise_plan(self, user_goal, page_snapshot, previous_plan=None, validation_error=None, invalid_plan=None):
        self.calls += 1
        if self.calls == 1:
            return TaskSpec.model_validate(
                {
                    "goal": user_goal,
                    "start_url": "https://example.com",
                    "allowed_domains": ["example.com"],
                    "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                    "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                    "steps": [
                        {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                        {
                            "step_id": 2,
                            "action": "extract_value_near_anchor",
                            "args": {"anchor_text": "Users"},
                            "save_as": "value",
                        },
                        {"step_id": 3, "action": "finish", "args": {}},
                    ],
                }
            )
        assert validation_error is not None
        return TaskSpec.model_validate(
            {
                "goal": user_goal,
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
                "expected_result": {"description": "Extract value", "required_fields": ["value"]},
                "steps": [
                    {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                    {
                        "step_id": 2,
                        "action": "extract_value_near_anchor",
                        "args": {"anchor_text": "Users", "value_pattern": r"(\\d+)"},
                        "save_as": "value",
                    },
                    {"step_id": 3, "action": "finish", "args": {}},
                ],
            }
        )


def test_two_stage_workflow_retries_invalid_final_plan_once():
    manager = WorkflowManager(
        planner=_FakePlanner(),
        validator=_FakeValidator(),
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=_FakeReplanner(),
        two_stage_planning=True,
    )

    result = asyncio.run(manager.run("Extract users count"))

    assert result["execution_result"].status == "success"
    assert result["final_plan"] is not None
    assert result["final_plan"].steps[1].args["value_pattern"] == r"(\\d+)"


def test_two_stage_initial_plan_uses_observation_actions_in_benchmark_mode():
    validator = _RecordingValidator()
    manager = WorkflowManager(
        planner=_FakePlanner(),
        validator=validator,
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=_FakeReplanner(),
        two_stage_planning=True,
    )

    result = asyncio.run(
        manager.run(
            "Extract users count",
            benchmark_context={
                "task_family": "single_value_extraction",
                "allowed_actions": ["open_url", "extract_text", "extract_pattern_from_page_text", "finish"],
            },
        )
    )

    assert result["execution_result"].status == "success"
    assert validator.allowed_actions_history[0] == {"open_url", "observe_page", "finish"}
    assert validator.allowed_actions_history[1] == {"open_url", "extract_text", "extract_pattern_from_page_text", "finish"}


def test_single_stage_benchmark_run_does_not_raise_normalization_attribute_error():
    manager = WorkflowManager(
        planner=_SingleStagePlanner(),
        validator=_RecordingValidator(),
        executor=_FakeExecutor(),
        verifier=_FakeVerifier(),
        replanner=None,
        two_stage_planning=False,
    )

    result = asyncio.run(
        manager.run(
            "Extract users count",
            benchmark_context={
                "task_family": "single_value_extraction",
                "allowed_actions": ["open_url", "extract_text", "extract_pattern_from_page_text", "finish"],
            },
        )
    )

    assert result["execution_result"].status == "success"
    assert result["plan"] is not None
    assert [step.action for step in result["plan"].steps] == ["open_url", "extract_text", "finish"]
