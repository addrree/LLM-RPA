from app.planner.planner import Planner
from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec
from tests.fakes import DummyLLMClient
from app.validator.plan_validator import PlanValidationError, PlanValidator
from app.verifier.llm_verifier import LLMVerifier


def test_dummy_planner_uses_url_from_goal():
    llm = DummyLLMClient()
    planner = Planner(llm)

    plan = planner.build_plan("Open https://www.wikipedia.org and extract h1")

    assert str(plan.start_url) == "https://www.wikipedia.org/"
    assert plan.allowed_domains == ["www.wikipedia.org"]
    assert plan.steps[0].args["url"] == "https://www.wikipedia.org"


def test_planner_validator_verifier_smoke_success():
    llm = DummyLLMClient()

    planner = Planner(llm)
    plan = planner.build_plan("Open https://example.com and extract h1")

    validator = PlanValidator()
    validator.validate(plan)

    result = ExecutionResult(
        status="success",
        extracted_data={"heading": "Example Domain"},
        final_url="https://example.com",
        page_title="Example Domain",
        page_text_excerpt="Example Domain This domain is for use in illustrative examples",
        screenshot_path="artifacts/screenshots/step_3.png",
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, result)

    assert verdict.verdict == "accept"
    assert verdict.task_completed is True


def test_dummy_verifier_rejects_failed_execution():
    llm = DummyLLMClient()

    planner = Planner(llm)
    plan = planner.build_plan("Open https://example.com and extract h1")

    failed_result = ExecutionResult(
        status="failed",
        extracted_data={},
        logs=[StepLog(step_id=1, action="open_url", status="failed", message="DNS error")],
        error_message="DNS error",
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, failed_result)

    assert verdict.verdict == "reject"
    assert verdict.task_completed is False


def test_validator_requires_save_as_for_extract_items():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract products",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Extract products", "required_fields": ["products"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_items",
                    "args": {
                        "container_selector": ".card",
                        "limit": 10,
                        "fields": {"title": ".title", "link": {"selector": "a", "attr": "href"}},
                    },
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    try:
        validator.validate(plan)
        raise AssertionError("Expected validator to reject extract_items without save_as")
    except PlanValidationError as exc:
        assert "save_as" in str(exc)


def test_validator_accepts_structured_extract_items_rules():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract top languages",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {"description": "Top language blocks", "required_fields": ["top_languages"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {
                    "step_id": 2,
                    "action": "extract_items",
                    "args": {
                        "container_selector": ".central-featured-lang",
                        "limit": 10,
                        "fields": {
                            "language_name": ".link-box strong",
                            "article_count": {
                                "selector": ".link-box small",
                                "pattern": r"([0-9][0-9\\s,\\.\\u00A0\\u202F\\+]*)",
                                "group_index": 1,
                                "normalize_number": True,
                                "number_type": "int",
                                "strip_plus": True,
                            },
                        },
                    },
                    "save_as": "top_languages",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_accepts_nested_required_fields_for_structured_output():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract top languages",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {"description": "Top language blocks", "required_fields": ["language_name", "article_count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {
                    "step_id": 2,
                    "action": "extract_items",
                    "args": {
                        "container_selector": ".central-featured-lang",
                        "limit": 10,
                        "fields": {
                            "language_name": ".link-box strong",
                            "article_count": {
                                "selector": ".link-box small",
                                "pattern": r"([0-9][0-9\\s,\\.\\u00A0\\u202F\\+]*)",
                                "group_index": 1,
                                "normalize_number": True,
                                "number_type": "int",
                                "strip_plus": True,
                            },
                        },
                    },
                    "save_as": "language_blocks",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_accepts_extract_structured_items():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract top languages from text snapshot",
            "start_url": "https://www.wikipedia.org",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 30},
            "expected_result": {"description": "Top language blocks", "required_fields": ["language_blocks"]},
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
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_accepts_parent_domain_for_subdomain_start_url():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract heading",
            "start_url": "https://www.wikipedia.org/",
            "allowed_domains": ["wikipedia.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Extract heading", "required_fields": ["heading"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "heading"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_rejects_unrelated_domain():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract heading",
            "start_url": "https://www.wikipedia.org/",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Extract heading", "required_fields": ["heading"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wikipedia.org/"}},
                {"step_id": 2, "action": "extract_text", "args": {"selector": "h1"}, "save_as": "heading"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    try:
        validator.validate(plan)
        raise AssertionError("Expected validator to reject unrelated allowed_domains")
    except PlanValidationError as exc:
        assert "start_url domain is not allowed" in str(exc)


def test_validator_requires_save_as_for_observe_page():
    plan = TaskSpec.model_validate(
        {
            "goal": "Observe page",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Observe page", "required_fields": []},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "observe_page", "args": {}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    try:
        validator.validate(plan)
        raise AssertionError("Expected validator to reject observe_page without save_as")
    except PlanValidationError as exc:
        assert "observe_page requires 'save_as'" in str(exc)


def test_validator_accepts_extract_pattern_from_page_text():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract count",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {
                    "step_id": 3,
                    "action": "extract_pattern_from_page_text",
                    "args": {
                        "pattern": r"Русский\s*\n?\s*([0-9][0-9\s,\.\u00A0\u202F\+]*)",
                        "occurrence": 1,
                        "group_index": 1,
                        "normalize_number": True,
                        "number_type": "int",
                        "strip_plus": True,
                    },
                    "save_as": "count",
                },
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_accepts_extract_text_near_text():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract count near anchor",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_text_near_text",
                    "args": {
                        "anchor_text": "English",
                        "pattern": r"English\s*([0-9][0-9\s,\.\u00A0\u202F\+]*)",
                        "group_index": 1,
                        "normalize_number": True,
                        "number_type": "int",
                    },
                    "save_as": "count",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_validator_accepts_extract_value_near_anchor():
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract count near anchor in same block",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 6, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Count", "required_fields": ["count"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "step_id": 2,
                    "action": "extract_value_near_anchor",
                    "args": {
                        "anchor_text": "English",
                        "value_pattern": r"([0-9][0-9\s,\.\u00A0\u202F\+]*)",
                        "search_direction": "after",
                        "same_block_only": True,
                        "required_right_context": "articles",
                        "group_index": 1,
                        "normalize_number": True,
                        "number_type": "int",
                        "strip_plus": True,
                    },
                    "save_as": "count",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    validator = PlanValidator()
    validator.validate(plan)


def test_verifier_ignores_technical_screenshot_required_field():
    llm = DummyLLMClient()

    plan = TaskSpec.model_validate(
        {
            "goal": "Open page and take screenshot",
            "start_url": "https://example.com",
            "allowed_domains": ["example.com"],
            "constraints": {"max_steps": 4, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {
                "description": "Screenshot artifact",
                "required_fields": ["screenshot_path"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
                {"step_id": 2, "action": "screenshot", "args": {"path": "artifacts/screenshots/a.png"}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    result = ExecutionResult(
        status="success",
        extracted_data={},
        final_url="https://example.com",
        page_title="Example Domain",
        page_text_excerpt="Example Domain",
        screenshot_path="artifacts/screenshots/a.png",
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    verifier = LLMVerifier(llm)
    verdict = verifier.verify(plan, result)

    assert verdict.verdict == "accept"
