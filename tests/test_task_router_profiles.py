import json

import pytest

from app.planner.prompts import build_profile_planner_prompt
from app.planner.task_router import SUPPORTED_RUNTIME_INTENTS, TaskRouter
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidationError, PlanValidator


def test_task_router_classifies_single_entity_metadata_with_runtime_safe_intents():
    route = TaskRouter().route(
        "Open https://example.org, search for package requests, and extract package name, latest version, and description."
    )

    assert route.task_type == "single_entity_metadata"
    assert route.confidence >= 0.65
    assert "entity_metadata" in route.profile.conceptual_intents
    assert "entity_metadata" not in route.profile.preferred_intents
    assert set(route.profile.preferred_intents).issubset(SUPPORTED_RUNTIME_INTENTS)


def test_task_router_classifies_search_results_table_cards_and_repeated_items():
    router = TaskRouter()

    search = router.route("Search for browser automation articles and return the results list with titles and links.")
    table = router.route("Extract all rows and columns from the visible pricing table.")
    cards = router.route("Collect product cards with product name, price, and rating.")
    repeated = router.route("Extract the visible list of article links from the page.")

    assert search.task_type == "search_results_extraction"
    assert table.task_type == "structured_table_extraction"
    assert cards.task_type == "catalog_or_card_extraction"
    assert repeated.task_type == "repeated_items_extraction"
    assert "card_or_catalog_items" in cards.profile.conceptual_intents
    assert "card_items" in cards.profile.preferred_runtime_intents
    assert "repeated_items" in repeated.profile.conceptual_intents
    assert "repeated_items" not in repeated.profile.preferred_intents


def test_task_router_classifies_russian_product_cards_as_catalog_profile():
    route = TaskRouter().route(
        "Открой маркетплейс, найди ssd и выгрузи карточки товаров: название, цену и ссылку."
    )

    assert route.task_type == "catalog_or_card_extraction"
    assert "product_cards" in route.profile.preferred_runtime_intents
    assert "product" == route.item_type


def test_task_router_uses_domain_words_only_as_weak_hints():
    product_value = TaskRouter().route("Open the page and extract the product price.")
    project_cards = TaskRouter().route("Open a catalog page and extract project cards with title, description, and link.")

    assert product_value.task_type != "catalog_or_card_extraction"
    assert "product_detail_hint" not in product_value.signals
    assert project_cards.task_type == "catalog_or_card_extraction"
    assert project_cards.item_type == "card"
    assert "card_items" in project_cards.profile.preferred_runtime_intents


def test_task_router_prioritizes_search_navigation_structure_over_metadata_words():
    route = TaskRouter().route(
        "Open https://search.sample.test/?q=browser+automation, open the first relevant repository result, "
        "then extract the opened page title, a short description, and the current URL."
    )

    assert route.task_type == "search_results_extraction"
    assert "search_navigation_then_extraction" in route.signals
    assert route.task_type != "single_entity_metadata"
    assert "current_url" in route.profile.preferred_runtime_intents
    assert "page_title" in route.profile.preferred_runtime_intents


def test_task_router_does_not_treat_table_export_as_row_action():
    route = TaskRouter().route(
        "Open https://example.org and extract table rows with columns Company, Contact, Country."
    )

    assert route.task_type == "structured_table_extraction"


def test_task_router_classifies_click_then_url_title_as_semantic_navigation():
    route = TaskRouter().route(
        "Open Wikipedia, click English on the landing page, and return the opened page title and current URL."
    )

    assert route.task_type == "semantic_navigation"
    assert "click_by_semantic_target" in route.profile.allowed_actions
    assert "current_url" in route.profile.preferred_runtime_intents
    assert "page_title" in route.profile.preferred_runtime_intents


def test_task_router_low_confidence_uses_restricted_generic_profile():
    route = TaskRouter().route("Open the site and continue.")

    assert route.task_type == "generic_web_task"
    assert "router_confidence_below_threshold_using_restricted_generic_profile" in route.warnings
    assert route.profile.full_vocabulary_was_used is False
    assert "click_by_semantic_target" in route.profile.allowed_actions
    assert len(route.profile.allowed_actions) < 30


def test_profile_prompt_does_not_include_full_vocabulary_or_conceptual_runtime_intent():
    route = TaskRouter().route(
        "Open https://example.org, search for package requests, and extract package name, latest version, and description."
    )
    prompt = build_profile_planner_prompt(route.profile)
    runtime_line = next(line for line in prompt.splitlines() if line.startswith("- preferred_runtime_intents:"))

    assert "visual_click_by_geometry" not in prompt
    assert "compare_structured_values" not in prompt
    assert "entity_metadata" not in runtime_line
    assert "package_metadata" in runtime_line


def test_profile_prompt_includes_card_items_as_runtime_intent_only():
    route = TaskRouter().route("Open a project catalog and extract cards with titles, descriptions, and links.")
    prompt = build_profile_planner_prompt(route.profile)
    runtime_line = next(line for line in prompt.splitlines() if line.startswith("- preferred_runtime_intents:"))
    conceptual_line = next(line for line in prompt.splitlines() if line.startswith("- conceptual_profile_intents_diagnostics_only:"))

    assert route.task_type == "catalog_or_card_extraction"
    assert "card_items" in runtime_line
    assert "card_items" not in conceptual_line


def test_profile_validation_rejects_action_outside_profile_with_controlled_diagnostics():
    route = TaskRouter().route("Extract the visible support email value from https://example.org.")
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract the visible support email value from https://example.org.",
            "start_url": "https://example.org",
            "allowed_domains": ["example.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Email", "required_fields": ["email"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
                {"step_id": 2, "action": "screenshot", "args": {"path": "unused.png"}},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    with pytest.raises(PlanValidationError) as exc_info:
        PlanValidator().validate(
            plan,
            allowed_actions=set(route.profile.allowed_actions),
            allowed_intents=set(route.profile.preferred_runtime_intents),
            forbidden_actions=set(route.profile.forbidden_actions),
            profile_diagnostics=route.diagnostics(),
        )

    message = str(exc_info.value)
    assert message.startswith("planner_validation_failed:")
    diagnostics = json.loads(message.split("planner_validation_failed: ", 1)[1])
    assert diagnostics["task_type"] == "direct_value_extraction"
    assert diagnostics["invalid_action"] == "screenshot"
    assert "screenshot" not in diagnostics["allowed_actions"]
    assert diagnostics["full_vocabulary_was_used"] is False


def test_profile_validation_rejects_conceptual_intent_in_task_spec():
    route = TaskRouter().route(
        "Open https://example.org, search for package requests, and extract package name, latest version, and description."
    )
    plan = TaskSpec.model_validate(
        {
            "goal": "Extract package metadata",
            "start_url": "https://example.org",
            "allowed_domains": ["example.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Metadata", "required_fields": ["metadata"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
                {
                    "step_id": 2,
                    "action": "extract_by_intent",
                    "args": {"intent": "entity_metadata", "output_key": "metadata"},
                    "save_as": "metadata",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )

    with pytest.raises(PlanValidationError) as exc_info:
        PlanValidator().validate(
            plan,
            allowed_actions=set(route.profile.allowed_actions),
            allowed_intents=set(route.profile.preferred_runtime_intents),
            forbidden_actions=set(route.profile.forbidden_actions),
            profile_diagnostics=route.diagnostics(),
        )

    diagnostics = json.loads(str(exc_info.value).split("planner_validation_failed: ", 1)[1])
    assert diagnostics["invalid_intent"] == "entity_metadata"
    assert "entity_metadata" in diagnostics["conceptual_profile_intents"]
    assert "entity_metadata" not in diagnostics["preferred_runtime_intents"]
