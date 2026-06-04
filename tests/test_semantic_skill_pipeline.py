from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord, enrich_miniwob_report
from app.executor.action_handlers import ActionHandlers
from app.interaction.action_grounder import ActionGrounder
from app.observer.page_observer import PageObserver
from app.orchestrator.workflow_manager import WorkflowManager, normalize_benchmark_plan
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.schemas.page_snapshot import PageSnapshot
from app.schemas.task_spec import Constraints, ExpectedResult, TaskSpec
from app.validator.plan_validator import PlanValidator
from scripts.run_real_web_skill_suite import (
    _controlled_preflight_failure_stage,
    _execution_has_antibot,
    _augment_extracted_data_for_scenario,
    _extract_query_from_goal_or_url,
    _is_llm_quota_error,
    _missing_top_level_fields,
    _preflight_status,
    _scenario_real_workflow_context,
    _scenario_goal,
    _skill_coverage,
    build_quota_skipped_result,
    build_summary,
    parse_args,
    select_scenarios,
)


def _plan_with_steps(steps, required_fields):
    return TaskSpec(
        goal="Extract public data",
        start_url="https://example.com",
        allowed_domains=["example.com"],
        constraints=Constraints(max_steps=10, max_replans=1, max_verification_retries=1, timeout_sec=30),
        expected_result=ExpectedResult(description="result", required_fields=required_fields),
        steps=steps,
    )


def test_validator_accepts_semantic_and_extraction_actions():
    plan = _plan_with_steps(
        [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://example.com"}},
            {"step_id": 2, "action": "observe_page", "args": {}, "save_as": "page_snapshot"},
            {"step_id": 3, "action": "click_by_semantic_target", "args": {"target_text": "Help", "role": "link", "exact": True}},
            {"step_id": 4, "action": "fill_by_semantic_target", "args": {"field_hint": "Search", "value": "maps"}},
            {"step_id": 5, "action": "select_by_semantic_target", "args": {"control_hint": "Language", "option_text": "English"}},
            {"step_id": 6, "action": "extract_visible_links", "args": {"output_key": "links"}},
            {"step_id": 7, "action": "find_row_by_condition", "args": {"condition": {"contains": "USD"}}, "save_as": "currency"},
            {"step_id": 8, "action": "finish", "args": {}},
        ],
        ["links", "currency"],
    )

    PlanValidator().validate(plan)


def test_miniwob_report_flags_mark_component_benchmark():
    report = BrowserGymRunReport(
        env_id="browsergym/miniwob.find-word",
        goal="g",
        status="success",
        reward=1.0,
        steps=[
            BrowserGymStepRecord(
                step_idx=0,
                action_string='fill("textbox", "word")',
                mapping_strategy="extraction_ordinal_word_fill",
                mapping_diagnostics={"extraction_intent": "ordinal_word_extraction"},
            )
        ],
    )

    enriched = enrich_miniwob_report(report)

    assert enriched.planner_used is False
    assert enriched.verifier_used is False
    assert enriched.extraction_controller_used is True
    assert enriched.llm_used is False
    assert enriched.skill_name == "semantic_fill"
    assert enriched.steps[0].controller_name == "extraction_controller"


def test_row_payload_maps_generic_table_headers_to_common_fields():
    payload = ActionHandlers._build_row_payload(
        {
            "row_id": "row_1",
            "selector": "table tr:nth-of-type(2)",
            "text": "840 USD 1 Доллар США 79,10",
            "headers": ["Цифр. код", "Букв. код", "Единиц", "Валюта", "Курс"],
            "cells": ["840", "USD", "1", "Доллар США", "79,10"],
        }
    )

    assert payload["currency"] == "USD"
    assert payload["nominal"] == "1"
    assert payload["name"] == "Доллар США"
    assert payload["rate"] == "79,10"


def test_condition_terms_accept_structured_value_condition():
    assert ActionHandlers._condition_terms({"field": "currency_name", "operator": "contains", "value": "USD"}) == ["USD"]


def test_condition_terms_flatten_list_values_without_stringifying_python_lists():
    assert ActionHandlers._condition_terms({"code": ["USD", "EUR"]}) == ["USD", "EUR"]
    assert ActionHandlers._condition_term_groups({"code": ["USD", "EUR"]}) == [["USD"], ["EUR"]]


def test_condition_term_groups_treat_or_expression_as_alternative_rows():
    assert ActionHandlers._condition_term_groups("name == 'Dollar US' or name == 'Euro'") == [["Dollar US"], ["Euro"]]


def test_condition_term_groups_accept_generic_item_fields():
    assert ActionHandlers._condition_term_groups({"title": ["Python", "AI"]}) == [["Python"], ["AI"]]


def test_confident_item_filter_applies_only_explicit_conditions():
    rows = [
        {"title": "Python automation", "description": "Browser workflows", "text": "Python automation Browser workflows"},
        {"title": "Rust compiler", "description": "Release notes", "text": "Rust compiler Release notes"},
    ]
    runtime_state = {}

    unfiltered, skipped_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"target": "Python"},
        runtime_state=runtime_state,
        output_key="items",
    )
    filtered, applied_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"condition": {"title": "Python"}},
        runtime_state=runtime_state,
        output_key="items",
    )
    empty, empty_note = ActionHandlers._apply_confident_item_filter(
        items=list(rows),
        args={"condition": {"title": "Go"}},
        runtime_state=runtime_state,
        output_key="items",
    )

    assert unfiltered == rows
    assert skipped_note == "condition_not_applied"
    assert filtered == [rows[0]]
    assert applied_note == "filter_applied"
    assert empty == []
    assert empty_note == "no_matching_items"
    assert {"output_key": "items", "reason": "no_matching_items", "condition": {"title": "Go"}} in runtime_state["condition_filter_diagnostics"]


def test_find_row_by_condition_matches_structured_value_condition():
    handler = ActionHandlers()

    async def _rows(*_args, **_kwargs):
        return [
            {
                "row_id": "row_1",
                "selector": "table tr:nth-of-type(2)",
                "text": "840 USD 1 Dollar US 79,10",
                "headers": ["numeric code", "currency code", "nominal", "name", "rate"],
                "cells": ["840", "USD", "1", "Dollar US", "79,10"],
            }
        ]

    handler._collect_row_candidates_generic = _rows  # type: ignore[method-assign]
    runtime_state = {}
    result = asyncio.run(
        handler.find_row_by_condition(
            page=object(),
            args={"condition": {"field": "currency_name", "operator": "contains", "value": "Dollar US"}},
            runtime_state=runtime_state,
        )
    )

    assert result["currency"] == "USD"
    assert result["name"] == "Dollar US"
    assert runtime_state["last_row_by_condition"]["rate"] == "79,10"


def test_find_row_by_condition_returns_multiple_rows_for_list_condition():
    handler = ActionHandlers()

    async def _rows(*_args, **_kwargs):
        return [
            {
                "row_id": "row_1",
                "selector": "table tr:nth-of-type(2)",
                "text": "840 USD 1 Dollar US 70,90",
                "headers": ["numeric code", "currency code", "nominal", "name", "rate"],
                "cells": ["840", "USD", "1", "Dollar US", "70,90"],
            },
            {
                "row_id": "row_2",
                "selector": "table tr:nth-of-type(3)",
                "text": "978 EUR 1 Euro 82,72",
                "headers": ["numeric code", "currency code", "nominal", "name", "rate"],
                "cells": ["978", "EUR", "1", "Euro", "82,72"],
            },
        ]

    handler._collect_row_candidates_generic = _rows  # type: ignore[method-assign]
    runtime_state = {}
    result = asyncio.run(
        handler.find_row_by_condition(
            page=object(),
            args={"condition": {"code": ["USD", "EUR"]}},
            runtime_state=runtime_state,
        )
    )

    assert [row["currency"] for row in result] == ["USD", "EUR"]
    assert result[0]["rate"] == "70,90"
    assert result[1]["rate"] == "82,72"
    assert runtime_state["last_rows_by_condition"] == result


def test_extract_by_intent_returns_last_row_details_for_currency_like_rows():
    class _Page:
        url = "https://example.com"

        async def title(self):
            return "Example"

    runtime_state = {
        "last_row_by_condition": {
            "currency": "USD",
            "nominal": "1",
            "name": "Dollar US",
            "rate": "79,10",
        }
    }

    result = asyncio.run(
        ActionHandlers().extract_by_intent(
            _Page(),
            {"intent": "currency_details"},
            runtime_state,
        )
    )

    assert result == {"currency": "USD", "name": "Dollar US", "nominal": "1", "rate": "79,10"}


def test_pattern_extraction_marks_used_skill():
    class _BodyLocator:
        async def inner_text(self):
            return "English 7,180,000 articles"

    class _Page:
        def locator(self, selector):
            assert selector == "body"
            return _BodyLocator()

    runtime_state = {}
    result = asyncio.run(
        ActionHandlers().extract_pattern_from_page_text(
            _Page(),
            {
                "pattern": r"English\s+([0-9,]+)\s+articles",
                "group_index": 1,
                "normalize_number": True,
                "number_type": "int",
            },
            runtime_state,
        )
    )

    assert result == 7180000
    assert "numeric_extraction" in runtime_state["used_skills"]


def test_selector_wait_is_advisory_unless_strict():
    class _Page:
        async def wait_for_selector(self, *_args, **_kwargs):
            raise TimeoutError("missing selector")

    args = {"selector": "a[href='/en']", "timeout_ms": 1}
    asyncio.run(ActionHandlers().wait_for(_Page(), args, {}))

    assert "continued" in args["_executor_note"]

    try:
        asyncio.run(ActionHandlers().wait_for(_Page(), {"selector": "a[href='/en']", "timeout_ms": 1, "strict": True}, {}))
    except TimeoutError:
        pass
    else:
        raise AssertionError("strict selector wait should raise")


def test_language_link_filter_keeps_same_root_domain_links():
    links = [
        {"text": "English", "href": "https://en.wiktionary.org/"},
        {"text": "Русский", "href": "https://ru.wiktionary.org/"},
        {"text": "Wikipedia", "href": "https://www.wikipedia.org/"},
        {"text": "License", "href": "https://creativecommons.org/licenses/by-sa/4.0/"},
    ]

    filtered = ActionHandlers._filter_links_to_same_root_domain(
        links,
        current_url="https://www.wiktionary.org",
    )

    assert [item["text"] for item in filtered] == ["English", "Русский"]


def test_known_content_profile_request_helpers_are_removed():
    for name in (
        "_language_links_requested",
        "_filter_links_to_article_like_paths",
        "_article_metadata_requested",
        "_paper_results_requested",
        "_repository_results_requested",
    ):
        assert not hasattr(ActionHandlers, name)


def test_page_snapshot_supports_rich_observation_fields():
    snapshot = PageSnapshot(
        url="https://example.com",
        title="Example",
        screenshot_path="artifacts/screenshots/x.png",
        page_text_excerpt="Help\nSearch",
        visible_headings=[],
        visible_labels=[],
        visible_buttons=[],
        visible_inputs=[],
        timestamp=datetime.now(timezone.utc),
        page_text="Help\nSearch",
        text_lines=["Help", "Search"],
        links=[{"text": "Help", "href": "https://example.com/help"}],
        candidates=[{"candidate_id": "obs_1", "text": "Help", "role": "link"}],
        rows=[{"text": "USD Доллар США"}],
    )

    assert snapshot.text_lines == ["Help", "Search"]
    assert snapshot.links[0]["text"] == "Help"
    assert snapshot.rows[0]["text"] == "USD Доллар США"


def test_observe_page_collects_rich_context_from_generic_dom_payload():
    class _BodyLocator:
        async def inner_text(self):
            return "Home\nEnglish\n7,180,000+ articles"

    class _Page:
        url = "https://example.org"

        def __init__(self):
            self.calls = 0

        def locator(self, selector):
            if selector == "body":
                return _BodyLocator()
            raise AssertionError(selector)

        async def screenshot(self, **_kwargs):
            return None

        async def title(self):
            return "Example"

        async def evaluate(self, _script, _args=None):
            self.calls += 1
            if self.calls == 1:
                return [{"text": "Home", "level": "h1", "index": 0, "visible": True}]
            return {
                "candidates": [{"candidate_id": "obs_1", "tag": "a", "role": "link", "text": "English", "href": "https://example.org/en"}],
                "links": [{"text": "English", "href": "https://example.org/en"}],
                "rows": [{"row_id": "row_1", "text": "English 7,180,000+ articles"}],
                "tables": [],
            }

    observer = PageObserver()

    async def _empty_list(*_args, **_kwargs):
        return []

    observer._collect_texts = _empty_list  # type: ignore[method-assign]
    observer._collect_inputs = _empty_list  # type: ignore[method-assign]
    snapshot = asyncio.run(observer.observe_page(page=_Page(), screenshot_path="artifacts/screenshots/t.png"))

    assert snapshot.text_lines[:2] == ["Home", "English"]
    assert snapshot.links[0]["text"] == "English"
    assert snapshot.candidates[0]["innerText"] == "English"


def test_real_web_summary_excludes_skipped_from_attempted_rate():
    summary = build_summary(
        [
            {"id": "a", "status": "skipped", "success": False, "failure_stage": "preflight_unavailable"},
            {"id": "b", "status": "success", "success": True},
            {"id": "c", "status": "failed", "success": False, "failure_stage": "runtime"},
        ],
        suite={"suite_id": "suite", "suite_type": "real_web"},
    )

    assert summary["total_scenarios"] == 3
    assert summary["attempted_scenarios"] == 2
    assert summary["skipped_scenarios"] == 1
    assert summary["success_rate_attempted"] == 0.5
    assert summary["failure_buckets"] == {"runtime": 1}


def test_real_web_expected_field_check_accepts_semantic_output_aliases():
    assert _missing_top_level_fields(["links[]"], {"language_links": [{"text": "English"}]}) == []
    assert _missing_top_level_fields(["articles[]"], {"article_results": [{"title": "Post"}]}) == []
    assert _missing_top_level_fields(["currencies[]"], {"currency_data": [{"currency": "USD"}]}) == []
    assert _missing_top_level_fields(["papers[]"], {"articles": [{"title": "Paper"}]}) == []
    assert _missing_top_level_fields(["products[]"], {"links": [{"text": "Not a product"}]}) == ["products[]"]


def test_real_web_query_is_recovered_from_goal_or_final_url():
    assert _extract_query_from_goal_or_url('Открой arXiv и найди "web agents".') == "web agents"
    assert _extract_query_from_goal_or_url("Search site", "https://example.com/search?q=asyncio") == "asyncio"


def test_real_web_extracted_data_augmentation_lifts_nested_scalar_fields():
    class _Execution:
        final_url = "https://pypi.org/project/playwright/"
        extracted_data = {"package": {"package_name": "playwright", "version": "1.2.3", "description": "Browser automation"}}

    execution = _Execution()
    _augment_extracted_data_for_scenario(
        scenario={"expected_fields": ["package_name", "version", "description"]},
        user_goal='Открой PyPI и найди пакет "playwright".',
        execution=execution,
    )

    assert execution.extracted_data["package_name"] == "playwright"
    assert execution.extracted_data["version"] == "1.2.3"
    assert execution.extracted_data["description"] == "Browser automation"


def test_real_web_quota_detection_and_skipped_artifact(tmp_path):
    assert _is_llm_quota_error("Ollama request failed (status_code=429): reached your session usage limit")

    result = build_quota_skipped_result(
        {"id": "later", "site": "example", "url": "https://example.com", "goal": "Extract data"},
        backend="ollama_cloud",
        output_dir=tmp_path,
        error_message="status_code=429 usage limit",
    )

    assert result["status"] == "skipped"
    assert result["failure_stage"] == "llm_quota_exceeded"
    assert result["planner_used"] is False
    assert Path(result["artifact_path"]).is_file()


def test_real_web_preflight_treats_http_auth_status_as_unavailable():
    status = _preflight_status(
        {"http_ok": False, "http_status": 401},
        {"browser_ok": True, "browser_status": 401, "title": ""},
    )

    assert status == "unavailable"


def test_real_web_preflight_classifies_antibot_and_auth_stages():
    assert _preflight_status(
        {"http_ok": True, "http_status": 200},
        {"browser_ok": True, "browser_status": 200, "title": "Robot check", "body_text_excerpt": "captcha"},
    ) == "captcha_or_antibot"
    assert _controlled_preflight_failure_stage({"http_status": 401, "browser_status": 401}) == "skipped_http_401"
    assert _controlled_preflight_failure_stage({"http_status": 403, "browser_status": 403}) == "skipped_http_403"
    assert _controlled_preflight_failure_stage(
        {"status": "captcha_or_antibot", "title": "Access denied", "body_text_excerpt": "cloudflare"}
    ) == "skipped_captcha_or_antibot"


def test_real_web_execution_antibot_detection_from_snapshot_text():
    class _Log:
        message = ""

    class _Execution:
        extracted_data = {"page_snapshot": {"title": "Client Challenge", "page_text_excerpt": "Download audio CAPTCHA"}}
        logs = [_Log()]
        error_message = ""

    assert _execution_has_antibot(_Execution())


def test_real_web_default_selection_excludes_optional_marketplaces():
    suite = {
        "scenarios": [
            {"id": "wikipedia", "category": "anchor_value"},
            {"id": "ozon", "category": "marketplace", "optional": True, "default_enabled": False},
        ]
    }

    assert [item["id"] for item in select_scenarios(suite, scenario_ids=None, limit=None)] == ["wikipedia"]
    assert [item["id"] for item in select_scenarios(suite, scenario_ids=None, limit=None, include_optional=True)] == [
        "wikipedia",
        "ozon",
    ]
    assert [item["id"] for item in select_scenarios(suite, scenario_ids="ozon", limit=None)] == ["ozon"]


def test_real_web_skill_coverage_maps_action_names_to_used_skill_names():
    coverage = _skill_coverage(
        ["observe_page", "fill_by_semantic_target", "extract_structured_items"],
        ["observe_page", "semantic_fill", "row_list_extraction"],
    )

    assert coverage["coverage"] == 1.0
    assert coverage["missing_expected_skills"] == []


def test_action_grounder_matches_generic_search_field_and_submit_button():
    grounder = ActionGrounder()
    candidates = [
        {
            "candidate_id": "q",
            "kind": "textbox",
            "selector": "#id-search-field",
            "id": "id-search-field",
            "name": "q",
            "input_type": "search",
            "placeholder": "Search",
            "enabled": True,
        },
        {
            "candidate_id": "go",
            "kind": "button",
            "selector": "#submit",
            "id": "submit",
            "text": "GO",
            "value": "GO",
            "input_type": "submit",
            "enabled": True,
        },
    ]

    fill = grounder.ground({"action": "fill", "target": "search field", "value": "asyncio"}, candidates)
    click = grounder.ground({"action": "click", "target_text": "search button"}, candidates)

    assert fill.actions[0].args["selector"] == "#id-search-field"
    assert click.actions[0].args["selector"] == "#submit"


def test_action_grounder_rejects_skip_links_for_search_button():
    grounder = ActionGrounder()
    candidates = [
        {
            "candidate_id": "skip",
            "kind": "link",
            "selector": "ul.a11y-menu a",
            "text": "Skip to search",
            "href": "#search",
            "enabled": True,
        },
        {
            "candidate_id": "button",
            "kind": "button",
            "selector": "button.search",
            "text": "Search",
            "enabled": True,
        },
    ]

    click = grounder.ground({"action": "click", "target_text": "search button"}, candidates)

    assert click.actions[0].args["selector"] == "button.search"


def test_real_web_suite_default_prompt_is_plain_user_request():
    scenario = {
        "goal": "Открой Wikipedia и извлеки число статей рядом с English.",
        "url": "https://www.wikipedia.org",
        "expected_fields": ["english_article_count"],
        "expected_skills": ["extract_value_near_anchor"],
    }

    prompt = _scenario_goal(scenario)
    benchmark_prompt = _scenario_goal(scenario, request_mode="benchmark")

    assert parse_args([]).request_mode == "real"
    assert parse_args([]).task_timeout_sec == 1000.0
    assert "URL: https://www.wikipedia.org" in prompt
    assert "Expected output fields" not in prompt
    assert "Expected reusable skills" not in prompt
    assert "semantic/extraction action vocabulary" not in prompt
    assert "Expected reusable skills" in benchmark_prompt


def test_real_web_real_workflow_context_has_no_expected_skill_hints():
    context = _scenario_real_workflow_context(
        {
            "url": "https://www.wikipedia.org",
            "expected_fields": ["english_article_count"],
            "expected_skills": ["extract_value_near_anchor"],
        }
    )

    assert context["task_family"] == "real_web_user_request"
    assert "allowed_actions" in context
    assert "required_fields" not in context
    assert "expected_skills" not in context
    assert "evaluator_metadata" not in context


def test_real_web_smoke_disables_corrective_retry_loop():
    assert WorkflowManager._effective_max_retries_for_context(
        max_retries=1,
        benchmark_context={"task_family": "real_web_skill_smoke"},
    ) == 0
    assert WorkflowManager._effective_max_retries_for_context(
        max_retries=1,
        benchmark_context={"task_family": "real_web_user_request"},
    ) == 0


def test_planner_normalizes_partial_semantic_skill_plan_envelope():
    raw_plan = {
        "steps": [
            {"step_id": 1, "action": "open_url", "args": {"url": "https://www.wiktionary.org"}},
            {"step_id": 2, "action": "observe_page", "args": {}},
            {"step_id": 3, "action": "extract_visible_links", "args": {}},
            {"step_id": 4, "action": "finish", "args": {}},
        ],
        "expected_result": {"required_fields": ["links"]},
    }

    normalized = Planner._normalize_plan_envelope(
        raw_plan,
        "Open https://www.wiktionary.org and extract visible language links.",
        benchmark_context={"start_url": "https://www.wiktionary.org", "max_steps": 8},
    )

    assert normalized["goal"].startswith("Open https://www.wiktionary.org")
    assert normalized["allowed_domains"] == ["www.wiktionary.org"]
    assert normalized["expected_result"]["description"]
    assert normalized["steps"][1]["save_as"] == "page_snapshot"
    assert normalized["steps"][2]["save_as"] == "links"


def test_planner_normalizes_structured_fields_list_and_top_level_required_fields():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.com"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": r"(\d{3})\s+(Dollar US)\s+([0-9.,]+)",
                        "limit": 1,
                        "fields": [{"group_index": 1}, {"group_index": 2}, {"group_index": 3}],
                    },
                    "save_as": "usd_data",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["code", "name", "rate"]},
        },
        "Open rates and extract Dollar US row.",
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    fields = normalized["steps"][1]["args"]["fields"]
    assert list(fields) == ["code", "name", "rate"]
    assert normalized["expected_result"]["required_fields"] == ["usd_data"]


def test_planner_normalizes_top_level_step_array_with_output_key():
    raw_steps = [
        {"action": "open_url", "url": "https://www.wiktionary.org"},
        {"action": "extract_visible_links", "output_key": "language_links"},
        {"action": "finish", "expected_result": {"required_fields": ["text", "href"]}},
    ]
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(raw_steps)
    assert oov is False
    normalized_plan = Planner._normalize_plan_envelope(
        payload,
        "Открой Wiktionary и извлеки видимые языковые ссылки с текстом и href.\nURL: https://www.wiktionary.org",
    )
    normalized_plan = Planner._normalize_required_fields_against_steps(normalized_plan)

    assert normalized_plan["steps"][0]["args"]["url"] == "https://www.wiktionary.org"
    assert normalized_plan["steps"][1]["args"]["output_key"] == "language_links"
    assert normalized_plan["steps"][1]["save_as"] == "language_links"
    assert normalized_plan["expected_result"]["required_fields"] == ["language_links"]


def test_planner_normalizes_params_and_url_title_required_aliases():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "params": {"url": "https://wiki.openstreetmap.org"}},
                {"action": "click_by_semantic_target", "params": {"target": "Help"}},
                {"action": "observe_page", "params": {}},
            ],
            "expected_result": {"required_fields": ["url", "title"]},
        },
        "Открой сайт и нажми Help, затем верни текущий URL и заголовок страницы.",
    )

    actions = [step["action"] for step in normalized["steps"]]
    assert normalized["steps"][0]["args"]["url"] == "https://wiki.openstreetmap.org"
    assert normalized["steps"][1]["args"]["target_text"] == "Help"
    assert "extract_by_intent" in actions
    assert normalized["expected_result"]["required_fields"] == ["final_url", "page_title"]
    assert any(step.get("save_as") == "final_url" for step in normalized["steps"])
    assert any(step.get("save_as") == "page_title" for step in normalized["steps"])


def test_planner_recovers_non_string_observe_save_as_for_metadata_request():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://wiki.openstreetmap.org"}},
                {"action": "click_by_semantic_target", "args": {"target": "Help"}},
                {"action": "observe_page", "save_as": {"url": "current_url", "title": "page_title"}},
            ],
            "expected_result": {"required_fields": ["url", "title"]},
        },
        "Open OpenStreetMap Wiki, click Help, then return current URL and page title.",
    )

    assert normalized["steps"][2]["save_as"] == "page_snapshot"
    assert any(step.get("save_as") == "final_url" for step in normalized["steps"])
    assert any(step.get("save_as") == "page_title" for step in normalized["steps"])


def test_planner_normalizes_metadata_extraction_aliases_from_llm_plan():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://wiki.openstreetmap.org"}},
                {"action": "click_by_semantic_target", "args": {"text": "Wiki"}},
                {"action": "observe_page"},
                {"action": "extract_text", "args": {"selector": "title"}, "save_as": "title"},
                {"action": "extract_value_near_anchor", "args": {"anchor_text": "Current URL"}, "save_as": "url"},
            ],
            "expected_result": {"required_fields": ["title", "url"]},
        },
        "Open OpenStreetMap Wiki, click Wiki or Help, then return current URL and page title.",
    )

    assert normalized["steps"][1]["args"]["target_text"] == "Wiki"
    metadata_steps = [step for step in normalized["steps"] if step["action"] == "extract_by_intent"]
    assert {"intent": "page_title"} in [step["args"] for step in metadata_steps]
    assert {"intent": "current_url"} in [step["args"] for step in metadata_steps]
    assert any(step.get("save_as") == "page_title" for step in metadata_steps)
    assert any(step.get("save_as") == "final_url" for step in metadata_steps)


def test_replanner_normalizes_metadata_aliases_and_inserts_extractors():
    snapshot = PageSnapshot(
        url="https://wiki.openstreetmap.org/",
        title="OpenStreetMap Wiki",
        screenshot_path="",
        page_text_excerpt="Help Wiki",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url", "args": {"url": "https://wiki.openstreetmap.org/"}},
                {"action": "click_by_semantic_target", "args": {"target": "Help"}},
                {"action": "observe_page", "args": {}, "save_as": "page_snapshot"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["url", "title"]},
        },
        user_goal="Open OSM Wiki, click Help, then return URL and title.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    assert normalized["expected_result"]["required_fields"] == ["final_url", "page_title"]
    assert normalized["steps"][1]["args"]["target_text"] == "Help"
    metadata_steps = [step for step in normalized["steps"] if step["action"] == "extract_by_intent"]
    assert {"intent": "current_url"} in [step["args"] for step in metadata_steps]
    assert {"intent": "page_title"} in [step["args"] for step in metadata_steps]
    assert any(step.get("save_as") == "final_url" for step in metadata_steps)
    assert any(step.get("save_as") == "page_title" for step in metadata_steps)


def test_replanner_rewrites_current_url_anchor_extraction_to_metadata_intent():
    snapshot = PageSnapshot(
        url="https://wiki.openstreetmap.org/",
        title="OpenStreetMap Wiki",
        screenshot_path="",
        page_text_excerpt="Help Wiki",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url", "args": {"url": "https://wiki.openstreetmap.org/"}},
                {"action": "click_by_semantic_target", "args": {"text": "Wiki or Help"}},
                {"action": "observe_page"},
                {"action": "extract_text", "args": {"selector": "title"}, "save_as": "title"},
                {"action": "extract_value_near_anchor", "args": {"anchor_text": "Current URL"}, "save_as": "url"},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["url", "title"]},
        },
        user_goal="Открой сайт OpenStreetMap Wiki и нажми Wiki или Help, затем верни текущий URL и заголовок страницы.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    click_step = next(step for step in normalized["steps"] if step["action"] == "click_by_semantic_target")
    metadata_steps = [step for step in normalized["steps"] if step["action"] == "extract_by_intent"]

    assert click_step["save_as"] == "clicked_text"
    assert click_step["args"]["target_candidates"] == ["Wiki", "Help"]
    assert {"intent": "current_url"} in [step["args"] for step in metadata_steps]
    assert {"intent": "page_title"} in [step["args"] for step in metadata_steps]
    assert normalized["expected_result"]["required_fields"] == ["final_url", "page_title"]


def test_replanner_converts_selector_structured_items_to_generic_intent():
    snapshot = PageSnapshot(
        url="https://habr.com/ru/articles/",
        title="All articles",
        screenshot_path="",
        page_text_excerpt="Article title Author Time",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url", "args": {"url": "https://habr.com/ru/articles/"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": "article.tm-articles-list__item",
                        "item_type": "article",
                        "fields": {
                            "title": "h2.tm-title_h2 a.tm-title__link",
                            "url": "h2.tm-title_h2 a.tm-title__link",
                            "author": "span.tm-user-info__username",
                        },
                    },
                    "save_as": "articles",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["articles"]},
        },
        user_goal="Open Habr and extract visible articles.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["item_type_hint"] == "article"
    assert extract_step["args"]["output_key"] == "articles"
    assert extract_step["args"]["limit"] == 20
    assert extract_step["save_as"] == "articles"


def test_planner_converts_generic_card_structured_items_to_card_items_intent():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org/cards"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "item_type": "cards",
                        "fields": {"title": 1, "description": 2, "url": 3},
                    },
                    "save_as": "cards",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["cards"]},
        },
        "Open a project catalog and extract cards with title, description, and link.",
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "cards"
    assert extract_step["save_as"] == "cards"


def test_replanner_converts_generic_card_structured_items_to_card_items_intent():
    snapshot = PageSnapshot(
        url="https://example.org/cards",
        title="Cards",
        screenshot_path="",
        page_text_excerpt="Project cards",
        timestamp=datetime.now(timezone.utc),
    )
    normalized = Replanner.normalize_final_plan(
        raw_plan={
            "steps": [
                {"action": "open_url", "args": {"url": "https://example.org/cards"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "pattern": "article.card",
                        "item_type": "card",
                        "fields": {"title": "h2", "description": "p", "url": "a"},
                    },
                    "save_as": "cards",
                },
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["cards"]},
        },
        user_goal="Open a project catalog and extract cards with title, description, and link.",
        previous_plan=None,
        page_snapshot=snapshot,
    )

    extract_step = normalized["steps"][1]
    assert extract_step["action"] == "extract_by_intent"
    assert extract_step["args"]["intent"] == "card_items"
    assert extract_step["args"]["output_key"] == "cards"


def test_planner_simplifies_generic_result_link_click_targets():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://pypi.org"}},
                {"action": "click_by_semantic_target", "args": {"target": "package link for playwright"}},
            ],
            "expected_result": {"required_fields": ["package_name"]},
        },
        "Open PyPI and click the package link for playwright.",
    )

    assert normalized["steps"][1]["args"]["target_text"] == "playwright"


def test_action_aliases_map_to_semantic_and_intent_actions():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "steps": [
                {"action": "fill_input", "args": {"target": "search", "value": "requests"}},
                {"action": "click_link", "args": {"text": "Help"}},
                {"action": "extract_package_info", "args": {"output_key": "package"}},
                {"action": "extract_product_cards", "args": {"output_key": "products"}},
                {"action": "extract_card_items", "args": {"output_key": "cards"}},
            ]
        }
    )

    assert oov is False
    assert [step["action"] for step in payload["steps"]] == [
        "fill_by_semantic_target",
        "click_by_semantic_target",
        "extract_by_intent",
        "extract_by_intent",
        "extract_by_intent",
    ]
    assert payload["steps"][2]["args"]["intent"] == "field_schema"
    assert payload["steps"][3]["args"]["intent"] == "card_items"
    assert payload["steps"][4]["args"]["intent"] == "card_items"
    assert payload["_normalized_action_aliases"] == [
        {"from": "fill_input", "to": "fill_by_semantic_target"},
        {"from": "click_link", "to": "click_by_semantic_target"},
        {"from": "extract_package_info", "to": "extract_by_intent"},
        {"from": "extract_product_cards", "to": "extract_by_intent"},
        {"from": "extract_card_items", "to": "extract_by_intent"},
    ]


def test_extract_by_intent_routes_generic_search_product_and_card_intents():
    handler = ActionHandlers()

    async def _cards(*, page, args, runtime_state=None):
        return [
            {
                "title": str(args.get("item_type_hint") or "Project Alpha"),
                "description": "Automation toolkit",
                "href": "https://example.com/a",
                "price_aware": bool(args.get("price_aware")),
            }
        ]

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._collect_card_items_generic = _cards  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    result = asyncio.run(handler.extract_by_intent(object(), {"intent": "search_results"}, {}))
    products = asyncio.run(handler.extract_by_intent(object(), {"intent": "product_cards"}, {}))
    cards = asyncio.run(handler.extract_by_intent(object(), {"intent": "card_items", "condition": {"title": "Alpha"}}, {}))

    assert result[0]["title"] == "result"
    assert products[0]["title"] == "product"
    assert products[0]["price_aware"] is True
    assert cards[0]["title"] == "Project Alpha"


def test_product_price_normalization_and_budget_detection():
    assert ActionHandlers._normalize_price_token("7 000 ₽") == 7000
    assert ActionHandlers._normalize_price_token("7000 руб.") == 7000
    assert ActionHandlers._extract_budget_from_goal_or_args(
        args={},
        runtime_state={"user_goal": "найди SSD до 7000 рублей"},
    ) == 7000


def test_table_rows_filter_by_condition_uses_generic_terms():
    rows = [
        {"currency": "USD", "name": "Доллар США", "cells": ["USD", "Доллар США"], "text": "USD Доллар США"},
        {"currency": "CNY", "name": "Юань", "cells": ["CNY", "Юань"], "text": "CNY Юань"},
    ]

    matched = ActionHandlers._filter_structured_rows_by_condition(rows=rows, condition={"code": ["USD", "EUR"]})

    assert matched == [rows[0]]


def test_currency_table_rows_intent_does_not_default_to_usd_eur():
    handler = ActionHandlers()
    rows = [
        {"currency": "USD", "name": "Dollar", "cells": ["USD", "Dollar"], "text": "USD Dollar"},
        {"currency": "CNY", "name": "Yuan", "cells": ["CNY", "Yuan"], "text": "CNY Yuan"},
    ]

    async def _rows(*, page, limit):
        return list(rows)

    async def _not_blocked(*_args, **_kwargs):
        return None

    handler._extract_table_rows_as_dicts = _rows  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _not_blocked  # type: ignore[method-assign]

    all_rows = asyncio.run(handler.extract_by_intent(object(), {"intent": "currency_table_rows"}, {}))
    usd_only = asyncio.run(
        handler.extract_by_intent(object(), {"intent": "currency_table_rows", "condition": {"code": "USD"}}, {})
    )

    assert all_rows == rows
    assert usd_only == [rows[0]]


def test_preflight_classifies_rate_limit_as_controlled_skip():
    preflight = {"http_status": 429, "title": "Rate exceeded", "body_text_excerpt": "Rate exceeded."}

    assert _controlled_preflight_failure_stage(preflight) == "skipped_rate_limited"


def test_real_web_suite_defaults_to_two_stage_planning():
    assert parse_args([]).two_stage_planning is True
    assert parse_args(["--two-stage-planning", "false"]).two_stage_planning is False


def test_planner_moves_root_step_parameters_into_args():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "url": "https://wiki.openstreetmap.org"},
                {"action": "click_by_semantic_target", "target_text": "Wiki|Help"},
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["url", "title"]},
        },
        "Open site and click Wiki or Help.",
    )

    click_step = normalized["steps"][1]
    assert normalized["steps"][0]["args"]["url"] == "https://wiki.openstreetmap.org"
    assert click_step["args"]["target_text"] == "Wiki|Help"
    assert click_step["args"]["target_candidates"] == ["Wiki", "Help"]


def test_planner_normalizes_actions_key_and_structured_without_pattern_to_links():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "actions": [
                {"action": "open_url", "params": {"url": "https://habr.com/ru/articles/"}},
                {
                    "action": "extract_structured_items",
                    "params": {"intent": "article titles and links", "output_key": "articles"},
                },
            ],
            "expected_result": {"required_fields": ["titles", "links"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Открой Habr articles и извлеки первые видимые заголовки статей и ссылки на них.",
        benchmark_context={"allowed_actions": ["open_url", "extract_visible_links", "finish"]},
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["steps"][1]["action"] == "extract_visible_links"
    assert normalized["steps"][1]["args"]["output_key"] == "articles"
    assert normalized["steps"][1]["save_as"] == "articles"
    assert normalized["expected_result"]["required_fields"] == ["articles"]


def test_planner_canonicalizes_structured_item_type_to_supported_intent():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://habr.com/ru/articles/"}},
                {
                    "action": "extract_structured_items",
                    "args": {"item_type": "article", "output_key": "articles"},
                    "save_as": "articles",
                },
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["articles"]},
        },
        "Open Habr and extract article cards.",
        benchmark_context={"allowed_actions": ["open_url", "extract_by_intent", "finish"]},
    )

    assert normalized["steps"][1]["action"] == "extract_by_intent"
    assert normalized["steps"][1]["args"]["intent"] == "card_items"
    assert normalized["steps"][1]["args"]["item_type_hint"] == "article"


def test_planner_normalizes_tasks_and_action_params():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "tasks": [
                {"action": "open_url", "action_params": {"url": "https://habr.com/ru/articles/"}},
                {
                    "action": "extract_structured_items",
                    "action_params": {
                        "item_selector": "article",
                        "fields": {
                            "title": {"selector": "h2 a", "attribute": "text"},
                            "url": {"selector": "h2 a", "attribute": "href"},
                        },
                        "limit": 5,
                    },
                    "save_as": "articles",
                },
            ],
            "expected_result": {"required_fields": ["title", "url"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open Habr articles and extract first visible article titles and links.",
        benchmark_context={"allowed_actions": ["open_url", "extract_items", "extract_visible_links", "finish"]},
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["steps"][0]["args"]["url"] == "https://habr.com/ru/articles/"
    assert normalized["steps"][1]["action"] == "extract_items"
    assert normalized["steps"][1]["args"]["container_selector"] == "article"
    assert normalized["steps"][1]["args"]["fields"]["url"]["attr"] == "href"
    assert "attr" not in normalized["steps"][1]["args"]["fields"]["title"]
    assert normalized["steps"][1]["save_as"] == "articles"
    assert normalized["expected_result"]["required_fields"] == ["articles"]


def test_planner_keeps_russian_article_metadata_collection_as_articles():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://habr.com/ru/articles/"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "fields": {
                            "title": "title",
                            "link": "link",
                            "author": "author",
                            "publication_time": "time",
                        }
                    },
                },
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["articles"]},
        },
        "Открой Habr и выгрузи первые видимые статьи: заголовок, ссылку, автора и время публикации.",
        benchmark_context={"allowed_actions": ["open_url", "extract_visible_links", "finish"]},
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["steps"][1]["action"] == "extract_visible_links"
    assert normalized["steps"][1]["args"]["output_key"] == "articles"
    assert normalized["steps"][1]["save_as"] == "articles"


def test_planner_accepts_order_key_as_step_list():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "order": [
                {"action": "open_url", "args": {"url": "https://arxiv.org"}},
                {"action": "fill_by_semantic_target", "args": {"target": "search input", "query": "web agents"}},
                {"action": "click_by_semantic_target", "args": {"target": "search button"}},
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["results"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open arXiv and search for web agents.",
        benchmark_context={
            "allowed_actions": [
                "open_url",
                "fill_by_semantic_target",
                "click_by_semantic_target",
                "finish",
            ]
        },
    )

    assert [step["action"] for step in normalized["steps"][:3]] == [
        "open_url",
        "fill_by_semantic_target",
        "click_by_semantic_target",
    ]
    assert normalized["steps"][1]["args"]["value"] == "web agents"


def test_planner_normalizes_natural_language_item_selector_to_visible_links():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "actions": [
                {"action": "open_url", "url": "https://arxiv.org"},
                {
                    "action": "extract_structured_items",
                    "item_selector": "search result entry",
                    "fields": {
                        "title": {"action": "extract_text", "target": "title"},
                        "link": {"action": "extract_text", "target": "paper link"},
                    },
                    "output_key": "results",
                },
            ],
            "expected_result": {"required_fields": ["title", "link"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open arXiv, search web agents, and extract result titles and links.",
        benchmark_context={"allowed_actions": ["open_url", "extract_items", "extract_visible_links", "finish"]},
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["steps"][1]["action"] == "extract_visible_links"
    assert normalized["steps"][1]["args"]["output_key"] == "results"
    assert normalized["steps"][1]["save_as"] == "results"
    assert normalized["expected_result"]["required_fields"] == ["results"]


def test_planner_normalizes_filtered_structured_items_to_row_condition():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "tasks": [
                {"action": "open_url", "args": {"url": "https://www.cbr.ru/currency_base/daily/"}},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "fields": ["code", "nominal", "name", "rate"],
                        "filter": {"code": ["USD", "EUR"]},
                        "output_key": "currency_data",
                    },
                },
            ],
            "expected_result": {"required_fields": ["currency_data"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open the central bank currency table and extract rows for USD and EUR.",
        benchmark_context={
            "allowed_actions": [
                "open_url",
                "extract_structured_items",
                "find_row_by_condition",
                "extract_visible_links",
                "finish",
            ]
        },
    )

    assert normalized["steps"][1]["action"] == "find_row_by_condition"
    assert normalized["steps"][1]["args"]["condition"] == {"code": ["USD", "EUR"]}
    assert normalized["steps"][1]["save_as"] == "currency_data"


def test_planner_normalizes_conditions_list_structured_items_to_row_condition():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "actions": [
                {"action": "open_url", "args": {"url": "https://www.cbr.ru/currency_base/daily/"}},
                {"action": "observe_page"},
                {
                    "action": "extract_structured_items",
                    "args": {
                        "fields": [
                            {"name": "code", "selector": "currency_code"},
                            {"name": "rate", "selector": "currency_rate"},
                        ],
                        "conditions": [
                            {"field": "code", "operator": "in", "value": ["USD", "EUR"]}
                        ],
                    },
                    "save_as": "currency_rates",
                },
            ],
            "expected_result": {"required_fields": ["currency_rates"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open the central bank currency table and extract rows for USD and EUR.",
        benchmark_context={
            "allowed_actions": [
                "open_url",
                "observe_page",
                "extract_structured_items",
                "find_row_by_condition",
                "extract_visible_links",
                "finish",
            ]
        },
    )

    assert normalized["steps"][2]["action"] == "find_row_by_condition"
    assert normalized["steps"][2]["args"]["condition"] == {"field": "code", "operator": "in", "value": ["USD", "EUR"]}
    assert normalized["steps"][2]["save_as"] == "currency_rates"


def test_planner_normalizes_semantic_target_and_incomplete_structured_extraction():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://arxiv.org"}},
                {"action": "fill_by_semantic_target", "args": {"semantic_target": "search input", "text": "web agents"}},
                {"action": "click_by_semantic_target", "args": {"semantic_target": "search button"}},
                {"action": "extract_structured_items", "args": {"output_key": "results"}},
                {"action": "finish"},
            ],
            "expected_result": {"required_fields": ["title", "authors", "link"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open arXiv, search web agents, and extract first results.",
        benchmark_context={"allowed_actions": ["open_url", "fill_by_semantic_target", "click_by_semantic_target", "extract_visible_links", "finish"]},
    )
    normalized = Planner._normalize_required_fields_against_steps(normalized)

    assert normalized["steps"][1]["args"]["target"] == "search input"
    assert normalized["steps"][2]["args"]["target_text"] == "search button"
    assert normalized["steps"][3]["action"] == "extract_visible_links"
    assert normalized["steps"][3]["save_as"] == "results"
    assert normalized["expected_result"]["required_fields"] == ["results"]


def test_planner_normalizes_query_alias_for_semantic_fill():
    from app.planner.action_vocab import normalize_plan_action_aliases

    payload, oov = normalize_plan_action_aliases(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://developer.mozilla.org"}},
                {"action": "fill_by_semantic_target", "args": {"target": "search input", "query": "fetch API"}},
            ],
            "expected_result": {"required_fields": ["results"]},
        }
    )
    assert oov is False

    normalized = Planner._normalize_plan_envelope(
        payload,
        "Open MDN and search fetch API.",
        benchmark_context={"allowed_actions": ["open_url", "fill_by_semantic_target", "finish"]},
    )

    assert normalized["steps"][1]["args"]["value"] == "fetch API"
    assert "query" not in normalized["steps"][1]["args"]


def test_structured_items_fallback_matches_table_row_by_pattern_literals():
    handler = ActionHandlers()

    async def _raise_pattern(*_args, **_kwargs):
        raise ValueError("pattern not found")

    async def _rows(*_args, **_kwargs):
        return [
            {
                "row_id": "row_1",
                "selector": "table tr:nth-of-type(2)",
                "text": "840 USD 1 Dollar US 79,10",
                "headers": ["numeric code", "currency code", "nominal", "name", "rate"],
                "cells": ["840", "USD", "1", "Dollar US", "79,10"],
            }
        ]

    handler.extract_pattern_from_page_text = _raise_pattern  # type: ignore[method-assign]
    handler._collect_row_candidates_generic = _rows  # type: ignore[method-assign]
    result = asyncio.run(
        handler.extract_structured_items(
            object(),
            {
                "pattern": r"(\d{3})\s+(Dollar US)\s+([0-9.,]+)",
                "limit": 1,
                "fields": {"code": {"group_index": 1}, "name": {"group_index": 2}, "rate": {"group_index": 3}},
            },
            {},
        )
    )

    assert result[0]["currency"] == "USD"
    assert result[0]["name"] == "Dollar US"
    assert result[0]["rate"] == "79,10"


def test_benchmark_normalizer_preserves_plan_required_fields_when_context_is_sanitized():
    plan = TaskSpec.model_validate(
        Planner._normalize_plan_envelope(
            {
                "steps": [
                    {"action": "open_url", "args": {"url": "https://www.wiktionary.org"}},
                    {"action": "extract_visible_links", "args": {}, "save_as": "links"},
                    {"action": "finish", "args": {}},
                ],
                "expected_result": {"required_fields": ["links"]},
            },
            "Open https://www.wiktionary.org and extract links.",
            benchmark_context={"start_url": "https://www.wiktionary.org", "max_steps": 8},
        )
    )

    normalized = normalize_benchmark_plan(
        plan,
        {"task_family": "real_web_skill_smoke", "allowed_actions": ["open_url", "extract_visible_links", "finish"]},
    )

    assert normalized.expected_result.required_fields == ["links"]


def test_planner_normalizes_common_semantic_action_aliases():
    normalized = Planner._normalize_plan_envelope(
        {
            "steps": [
                {"action": "open_url", "args": {"url": "https://wiki.openstreetmap.org"}},
                {"action": "click_by_semantic_target", "args": {"target": "visible link with text Wiki or Help"}},
                {"action": "extract_value_near_anchor", "args": {"anchor": "English", "value_type": "number"}},
                {"action": "finish", "args": {}},
            ],
            "expected_result": {"required_fields": ["clicked_text", "english_article_count", "final_url", "page_title"]},
        },
        "Open https://wiki.openstreetmap.org and click Wiki or Help.",
        benchmark_context={
            "start_url": "https://wiki.openstreetmap.org",
            "max_steps": 8,
            "allowed_actions": ["open_url", "click_by_semantic_target", "extract_value_near_anchor", "extract_by_intent", "finish"],
        },
    )

    click_step = normalized["steps"][1]
    anchor_step = normalized["steps"][2]
    actions = [step["action"] for step in normalized["steps"]]

    assert click_step["args"]["target_text"] == "Wiki or Help"
    assert click_step["args"]["target_candidates"] == ["Wiki", "Help"]
    assert click_step["args"]["target_candidates"] == ["Wiki", "Help"]
    assert click_step["save_as"] == "clicked_text"
    assert anchor_step["args"]["anchor_text"] == "English"
    assert "extract_by_intent" in actions
    assert any(step.get("save_as") == "final_url" for step in normalized["steps"])
    assert any(step.get("save_as") == "page_title" for step in normalized["steps"])


def test_benchmark_skill_mapping_document_mentions_component_scope():
    text = Path("docs/benchmark_skill_mapping.md").read_text(encoding="utf-8")

    assert "MiniWoB validates reusable lower-level skills" in text
    assert "semantic_click" in text
    assert "canvas_geometry" in text
