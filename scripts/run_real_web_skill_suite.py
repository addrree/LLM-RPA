#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import build_llm_client, build_workflow
from app.orchestrator.persistence import export_results, save_artifacts
from app.orchestrator.workflow_manager import WorkflowStageError
from app.utils.block_detection import detect_blocked_or_limited_payload, detect_blocked_or_limited_text

UTC = timezone.utc
DEFAULT_SUITE_PATH = Path("configs/eval/real_web_skill_suite.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/real_web")
REAL_WEB_ALLOWED_ACTIONS = {
    "open_url",
    "observe_page",
    "finish",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_items",
    "extract_structured_items",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
    "extract_by_intent",
    "extract_visible_links",
    "find_row_by_condition",
    "click_by_semantic_target",
    "fill_by_semantic_target",
    "select_by_semantic_target",
    "choose_autocomplete_suggestion",
    "click_row_action",
    "visual_observe",
    "visual_extract_object_count",
    "visual_click_by_geometry",
}
ANTIBOT_MARKERS = (
    "rate exceeded",
    "too many requests",
    "http 429",
    "rate limit",
    "captcha",
    "robot check",
    "cloudflare",
    "access denied",
    "verify you are human",
    "are you a human",
    "just a moment",
    "checking your browser",
    "ddos-guard",
    "perimeterx",
    "datadome",
    "not authorized",
    "unauthorized",
    "permission denied",
    "подтвердите, что вы не робот",
    "проверка безопасности",
    "доступ ограничен",
    "доступ запрещен",
    "доступ запрещён",
)
SKILL_COVERAGE_ALIASES = {
    "observe_page": {"observe_page"},
    "click_by_semantic_target": {"click_by_semantic_target", "semantic_click"},
    "fill_by_semantic_target": {"fill_by_semantic_target", "semantic_fill"},
    "select_by_semantic_target": {"select_by_semantic_target", "select_list_autocomplete"},
    "choose_autocomplete_suggestion": {"choose_autocomplete_suggestion", "select_list_autocomplete"},
    "extract_structured_items": {"extract_structured_items", "extract_items", "row_list_extraction"},
    "extract_items": {"extract_items", "row_list_extraction"},
    "extract_visible_links": {"extract_visible_links"},
    "extract_value_near_anchor": {"extract_value_near_anchor", "numeric_extraction"},
    "extract_by_intent": {"extract_by_intent", "numeric_extraction", "row_list_extraction"},
    "find_row_by_condition": {"find_row_by_condition", "row_list_extraction"},
    "click_row_action": {"click_row_action", "row_list_email_action"},
    "visual_observe": {"visual_observe"},
    "visual_extract_object_count": {"visual_extract_object_count", "visual_svg_recognition"},
    "visual_click_by_geometry": {"visual_click_by_geometry", "canvas_geometry"},
}


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run real-web semantic skill smoke suite through the main LLM+RPA pipeline")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--slow-mo", type=int, default=0)
    parser.add_argument("--export-format", action="append", choices=["json", "csv"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario-ids", default=None, help="Comma-separated scenario IDs")
    parser.add_argument("--skip-preflight", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--task-timeout-sec", type=float, default=1000.0)
    parser.add_argument("--llm-timeout-sec", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--include-optional", action="store_true", help="Include optional scenarios such as marketplaces")
    parser.add_argument("--include-marketplaces", action="store_true", help="Alias for --include-optional for marketplace scenarios")
    parser.add_argument("--category", default=None, help="Comma-separated category filter")
    parser.add_argument(
        "--request-mode",
        choices=["real", "benchmark"],
        default="real",
        help="real sends only the human goal+URL to the planner; benchmark also includes expected fields/skills hints",
    )
    parser.add_argument(
        "--two-stage-planning",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Use open_url -> observe_page -> context-aware replanning for real web scenarios (default: true)",
    )
    return parser.parse_args(argv)


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_optional_scenario(scenario: dict[str, Any]) -> bool:
    return (
        bool(scenario.get("optional"))
        or scenario.get("default_enabled") is False
        or str(scenario.get("category", "")).strip().lower() == "marketplace"
    )


def select_scenarios(
    suite: dict[str, Any],
    *,
    scenario_ids: str | None,
    limit: int | None,
    include_optional: bool = False,
    category: str | None = None,
) -> list[dict[str, Any]]:
    scenarios = list(suite.get("scenarios") or [])
    if scenario_ids:
        wanted = {item.strip() for item in scenario_ids.split(",") if item.strip()}
        scenarios = [scenario for scenario in scenarios if str(scenario.get("id")) in wanted]
    else:
        if not include_optional:
            scenarios = [scenario for scenario in scenarios if not _is_optional_scenario(scenario)]
        if category:
            wanted_categories = {item.strip().lower() for item in category.split(",") if item.strip()}
            scenarios = [
                scenario
                for scenario in scenarios
                if str(scenario.get("category", "")).strip().lower() in wanted_categories
            ]
    if limit is not None:
        scenarios = scenarios[: max(limit, 0)]
    return scenarios


def _http_probe(url: str, timeout_sec: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 LLM-RPA real-web preflight"})
    context = ssl.create_default_context()
    try:
        with urlopen(request, timeout=timeout_sec, context=context) as response:
            return {
                "http_ok": 200 <= int(response.status) < 400,
                "http_status": int(response.status),
                "http_final_url": response.geturl(),
            }
    except HTTPError as exc:
        return {
            "http_ok": False,
            "http_status": int(exc.code),
            "http_final_url": exc.url,
            "http_warning": str(exc),
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {"http_ok": False, "http_error": str(exc)}


def _preflight_status(http_result: dict[str, Any], browser_result: dict[str, Any]) -> str:
    if _detect_antibot(browser_result):
        return "captcha_or_antibot"
    browser_status = browser_result.get("browser_status")
    browser_status_ok = browser_status is None or 200 <= int(browser_status) < 400
    if bool(http_result.get("http_ok")) and bool(browser_result.get("browser_ok")) and browser_status_ok:
        return "available"
    return "unavailable"


async def check_url_accessible(url: str, timeout_sec: int = 15, *, show_browser: bool = False, slow_mo: int = 0) -> dict[str, Any]:
    http_result = _http_probe(url, timeout_sec=timeout_sec)
    browser_result: dict[str, Any] = {}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=not show_browser, slow_mo=slow_mo)
            page = await browser.new_page()
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_sec * 1000)
            browser_result = {
                "browser_ok": True,
                "browser_status": response.status if response else None,
                "final_url": page.url,
                "title": await page.title(),
            }
            try:
                body_text = await page.locator("body").inner_text(timeout=2500)
                browser_result["body_text_excerpt"] = body_text[:4000]
                browser_result["antibot_detected"] = _detect_antibot(browser_result)
                browser_result["antibot_markers"] = _matched_antibot_markers(browser_result)
            except Exception as exc:  # noqa: BLE001
                browser_result["body_text_error"] = str(exc)
            await browser.close()
    except Exception as exc:  # noqa: BLE001
        browser_result = {"browser_ok": False, "browser_error": str(exc)}

    return {"status": _preflight_status(http_result, browser_result), **http_result, **browser_result}


def _matched_antibot_markers(payload: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(payload.get("title", "")),
            str(payload.get("body_text_excerpt", "")),
            str(payload.get("http_warning", "")),
            str(payload.get("browser_error", "")),
        ]
    ).casefold()
    markers = []
    for marker in ANTIBOT_MARKERS:
        if marker.casefold() in text:
            markers.append(marker)
    return markers


def _detect_antibot(payload: dict[str, Any]) -> bool:
    return bool(_matched_antibot_markers(payload))


def _text_has_antibot_markers(text: str) -> bool:
    return detect_blocked_or_limited_text(text).blocked


def _execution_has_antibot(execution) -> bool:
    payload_parts: list[str] = []
    try:
        payload_parts.append(json.dumps(execution.extracted_data, ensure_ascii=False))
    except Exception:
        pass
    try:
        payload_parts.extend(str(log.message or "") for log in execution.logs)
    except Exception:
        pass
    payload_parts.append(str(getattr(execution, "error_message", "") or ""))
    return _text_has_antibot_markers("\n".join(payload_parts))


def _controlled_preflight_failure_stage(preflight: dict[str, Any]) -> str:
    status_codes = [
        preflight.get("http_status"),
        preflight.get("browser_status"),
    ]
    blocked = detect_blocked_or_limited_payload(preflight)
    if blocked.blocked and blocked.failure_stage:
        return blocked.failure_stage
    if preflight.get("status") == "captcha_or_antibot" or _detect_antibot(preflight):
        return "skipped_captcha_or_antibot"
    if 401 in status_codes:
        return "skipped_http_401"
    if 403 in status_codes:
        return "skipped_http_403"
    return "preflight_unavailable"


def _skill_coverage(expected_skills: list[str], used_skills: list[str]) -> dict[str, Any]:
    expected = [str(skill).strip() for skill in expected_skills if str(skill).strip()]
    used = {str(skill).strip() for skill in used_skills if str(skill).strip()}
    matched: list[str] = []
    missing: list[str] = []
    for skill in expected:
        aliases = SKILL_COVERAGE_ALIASES.get(skill, {skill})
        if used.intersection(aliases):
            matched.append(skill)
        else:
            missing.append(skill)
    return {
        "expected_skills": expected,
        "used_skills": sorted(used),
        "matched_expected_skills": matched,
        "missing_expected_skills": missing,
        "coverage": (len(matched) / len(expected)) if expected else None,
    }


def _artifact_to_json(artifact) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "raw_response": artifact.raw_response,
        "parsed_response": artifact.parsed_response,
        "generation": artifact.generation.model_dump(mode="json"),
    }


def _is_llm_quota_error(error: Exception | str) -> bool:
    text = str(error).lower()
    return "status_code=429" in text or "usage limit" in text or "reached your session usage limit" in text


def _normalize_expected_field(field: str) -> str:
    value = str(field or "").strip()
    return value[:-2] if value.endswith("[]") else value


def _candidate_output_keys_for_expected_field(field: str) -> list[str]:
    key = _normalize_expected_field(field)
    aliases = {
        "links": ["links", "visible_links", "language_links", "languages"],
        "articles": ["articles", "article_links", "article_results"],
        "news": ["news", "news_items", "headlines"],
        "results": ["results", "search_results"],
        "papers": ["papers", "paper_results", "articles"],
        "repositories": ["repositories", "repository_results", "repo_results", "repos"],
        "products": ["products", "product_cards"],
        "currencies": ["currencies", "currency_rows", "currency_data", "rates", "all_currency_data"],
    }
    candidates = [key]
    for alias in aliases.get(key, []):
        if alias not in candidates:
            candidates.append(alias)
    return [candidate for candidate in candidates if candidate]


def _missing_top_level_fields(expected_fields: list[str], extracted_data: dict[str, Any]) -> list[str]:
    missing = []
    for field in expected_fields:
        keys = _candidate_output_keys_for_expected_field(field)
        if not keys:
            continue
        if all(extracted_data.get(key) in (None, "", [], {}) for key in keys):
            missing.append(field)
    return missing


def _has_useful_extracted_data(extracted_data: dict[str, Any]) -> bool:
    for key, value in (extracted_data or {}).items():
        if key == "page_snapshot":
            continue
        if value not in (None, "", [], {}):
            return True
    return False


def _extract_query_from_goal_or_url(user_goal: str, final_url: str | None = None) -> str:
    goal = str(user_goal or "")
    quoted = re.search(r'"([^"]{1,200})"|«([^»]{1,200})»', goal)
    if quoted:
        return str(quoted.group(1) or quoted.group(2) or "").strip()
    try:
        params = parse_qs(urlparse(str(final_url or "")).query)
    except Exception:
        params = {}
    for key in ("q", "query", "search", "search_query", "term", "text"):
        values = params.get(key)
        if values and str(values[0]).strip():
            return str(values[0]).strip()
    return ""


def _augment_extracted_data_for_scenario(*, scenario: dict[str, Any], user_goal: str, execution) -> None:
    extracted_data = execution.extracted_data if isinstance(execution.extracted_data, dict) else {}
    expected_fields = [str(field).strip() for field in scenario.get("expected_fields") or [] if str(field).strip()]
    if "query" in expected_fields and not str(extracted_data.get("query", "")).strip():
        query = _extract_query_from_goal_or_url(user_goal, getattr(execution, "final_url", None))
        if query:
            extracted_data["query"] = query
    if "final_url" in expected_fields and not str(extracted_data.get("final_url", "")).strip():
        final_url = str(getattr(execution, "final_url", "") or "").strip()
        if final_url:
            extracted_data["final_url"] = final_url
    if "page_title" in expected_fields and not str(extracted_data.get("page_title", "")).strip():
        page_title = str(getattr(execution, "page_title", "") or "").strip()
        if page_title:
            extracted_data["page_title"] = page_title
    missing_scalar_fields = [
        field
        for field in expected_fields
        if not field.endswith("[]") and extracted_data.get(field) in (None, "", [], {})
    ]
    if missing_scalar_fields:
        for value in list(extracted_data.values()):
            if not isinstance(value, dict):
                continue
            for field in list(missing_scalar_fields):
                candidate = value.get(field)
                if candidate not in (None, "", [], {}):
                    extracted_data[field] = candidate
                    missing_scalar_fields.remove(field)
            if not missing_scalar_fields:
                break
    expected_collections = {_normalize_expected_field(field) for field in expected_fields if field.endswith("[]")}
    if "currencies" in expected_collections and extracted_data.get("currencies") in (None, "", [], {}):
        combined = []
        for key in ("currency_rows", "currency_data", "rates", "all_currency_data"):
            value = extracted_data.get(key)
            if isinstance(value, list) and value:
                combined.extend(item for item in value if isinstance(item, dict))
        for key in ("usd_data", "eur_data"):
            value = extracted_data.get(key)
            if isinstance(value, dict) and value:
                combined.append(value)
        if combined:
            seen = set()
            unique = []
            for item in combined:
                fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                unique.append(item)
            extracted_data["currencies"] = unique
    execution.extracted_data = extracted_data


def _scenario_goal(scenario: dict[str, Any], *, request_mode: str = "real") -> str:
    goal = str(scenario["goal"]).strip()
    url = str(scenario["url"]).strip()
    if request_mode == "real":
        if url and url not in goal:
            return f"{goal}\nURL: {url}"
        return goal

    fields = ", ".join(scenario.get("expected_fields") or [])
    skills = ", ".join(scenario.get("expected_skills") or [])
    return (
        f"{goal}\n"
        f"URL: {url}\n"
        f"Expected output fields: {fields}.\n"
        f"Expected reusable skills: {skills}.\n"
        "Use the main semantic/extraction action vocabulary. Prefer observe_page before extraction. "
        "Do not use site-specific CSS/XPath when a semantic action can express the step."
    )


def _scenario_benchmark_context(scenario: dict[str, Any]) -> dict[str, Any]:
    expected_skills = [str(item).strip() for item in scenario.get("expected_skills") or [] if str(item).strip()]
    expected_fields = [str(item).strip() for item in scenario.get("expected_fields") or [] if str(item).strip()]
    allowed_actions = {
        "open_url",
        "observe_page",
        "finish",
        "wait_for",
        "extract_text",
    }
    for skill in expected_skills:
        if skill == "visual_observe":
            allowed_actions.update({"visual_observe", "visual_extract_object_count", "visual_click_by_geometry"})
        else:
            allowed_actions.add(skill)
    if "extract_visible_links" in expected_skills:
        allowed_actions.add("extract_visible_links")
    if "extract_value_near_anchor" in expected_skills:
        allowed_actions.update({"extract_value_near_anchor", "extract_pattern_from_page_text"})
    if "extract_structured_items" in expected_skills:
        allowed_actions.update({"extract_structured_items", "extract_items", "extract_by_intent"})
    if "find_row_by_condition" in expected_skills or "extract_by_intent" in expected_skills:
        allowed_actions.update({"find_row_by_condition", "extract_by_intent", "extract_visible_links"})
    if "click_by_semantic_target" in expected_skills:
        allowed_actions.update({"click_by_semantic_target", "extract_by_intent", "extract_text"})

    return {
        "task_family": "real_web_skill_smoke",
        "allowed_actions": sorted(allowed_actions),
        "required_top_level_fields": expected_fields,
        "required_fields": expected_fields,
        "start_url": scenario.get("url"),
        "max_steps": 8,
        "evaluator_metadata": {
            "scenario_id": scenario.get("id"),
            "suite_type": "real_web_llm_rpa_skill_smoke",
            "expected_skills": expected_skills,
        },
    }


def _scenario_real_workflow_context(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_family": "real_web_user_request",
        "allowed_actions": sorted(REAL_WEB_ALLOWED_ACTIONS),
        "start_url": scenario.get("url"),
        "max_steps": 8,
    }


async def run_scenario(
    scenario: dict[str, Any],
    *,
    backend: str | None,
    show_browser: bool,
    slow_mo: int,
    export_formats: list[str],
    output_dir: Path,
    skip_preflight: bool,
    task_timeout_sec: float,
    llm_timeout_sec: int | None,
    request_mode: str,
    two_stage_planning: bool,
) -> dict[str, Any]:
    started = time.time()
    scenario_id = str(scenario["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {"status": "skipped_by_cli"} if skip_preflight else await check_url_accessible(
        str(scenario["url"]),
        timeout_sec=15,
        show_browser=False,
        slow_mo=0,
    )
    if not skip_preflight and preflight.get("status") != "available":
        failure_stage = _controlled_preflight_failure_stage(preflight)
        result = {
            "id": scenario_id,
            "site": scenario.get("site"),
            "category": scenario.get("category"),
            "url": scenario.get("url"),
            "user_goal": scenario.get("goal"),
            "status": "skipped",
            "success": False,
            "partial_success": False,
            "missing_fields": scenario.get("expected_fields") or [],
            "failure_stage": failure_stage,
            "error_message": preflight.get("browser_error") or preflight.get("http_error") or preflight.get("http_warning"),
            "preflight_status": preflight,
            "planner_used": False,
            "verifier_used": False,
            "used_skills": [],
            "blocked_or_limited_status": (
                detected_block.__dict__ if (detected_block := detect_blocked_or_limited_payload(preflight)).blocked else None
            ),
            "rate_limit_detected": failure_stage == "skipped_rate_limited",
            "normalized_action_aliases": [],
            "action_oov_detected": False,
            "expected_fields": scenario.get("expected_fields") or [],
            "expected_skills": scenario.get("expected_skills") or [],
            "skill_coverage": _skill_coverage(scenario.get("expected_skills") or [], []),
            "allowed_controlled_outcomes": scenario.get("allowed_controlled_outcomes") or [],
            "notes": scenario.get("notes"),
            "runtime_sec": round(time.time() - started, 3),
        }
        artifact_path = output_dir / f"{scenario_id}_{_timestamp()}.json"
        result["artifact_path"] = str(artifact_path)
        artifact_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    llm_client = build_llm_client(backend=backend)
    if llm_timeout_sec is not None:
        llm_client.timeout_sec = int(llm_timeout_sec)
    workflow = build_workflow(
        llm_client=llm_client,
        show_browser=show_browser,
        slow_mo=slow_mo,
        record_video=False,
        two_stage_planning=two_stage_planning,
        interaction_mode="plan",
    )
    user_goal = _scenario_goal(scenario, request_mode=request_mode)
    evaluator_context = _scenario_benchmark_context(scenario)
    workflow_context = evaluator_context if request_mode == "benchmark" else _scenario_real_workflow_context(scenario)
    if request_mode == "real" and not two_stage_planning:
        # Real-user smoke keeps the main planner/verifier path but avoids a second LLM
        # corrective replanning call after verifier rejection.
        workflow.replanner = None
    run_id = f"real_web_{scenario_id}_{_timestamp()}"
    try:
        workflow_result = await asyncio.wait_for(
            workflow.run(user_goal, benchmark_context=workflow_context),
            timeout=task_timeout_sec,
        )
        execution = workflow_result["execution_result"]
        _augment_extracted_data_for_scenario(scenario=scenario, user_goal=user_goal, execution=execution)
        artifact_paths = save_artifacts(workflow_result, run_id=run_id)
        export_paths = export_results(workflow_result, run_id=run_id, export_formats=export_formats)
        verdict = workflow_result["verdict"]
        planner_artifact = workflow_result.get("planner_artifact") or workflow_result.get("initial_planner_artifact")
        verifier_artifact = workflow_result.get("verifier_artifact")
        blocked_status = execution.blocked_or_limited_status or {}
        blocked_failure_stage = str(blocked_status.get("failure_stage") or "")
        anti_bot_after_load = _execution_has_antibot(execution) or bool(blocked_failure_stage)
        extracted_data = execution.extracted_data or {}
        expected_fields = scenario.get("expected_fields") or []
        missing_fields = _missing_top_level_fields(expected_fields, extracted_data)
        useful_extracted_data = _has_useful_extracted_data(extracted_data)
        success = (
            execution.status == "success"
            and verdict.verdict == "accept"
            and not anti_bot_after_load
            and not missing_fields
        )
        partial_success = (
            not success
            and not anti_bot_after_load
            and useful_extracted_data
            and execution.status in {"success", "failed"}
        )
        status = "success" if success else "failed"
        failure_stage = None if success else (execution.failure_type or "verifier_reject")
        allowed_controlled = set(str(item) for item in (scenario.get("allowed_controlled_outcomes") or []))
        if anti_bot_after_load:
            status = "skipped"
            failure_stage = blocked_failure_stage or "skipped_captcha_or_antibot"
        elif failure_stage and failure_stage in allowed_controlled:
            status = "skipped"
        elif partial_success:
            status = "partial_success"
            failure_stage = failure_stage or ("missing_required_fields" if missing_fields else "verifier_reject")
        result = {
            "id": scenario_id,
            "site": scenario.get("site"),
            "category": scenario.get("category"),
            "url": scenario.get("url"),
            "user_goal": user_goal,
            "status": status,
            "success": bool(success),
            "partial_success": bool(partial_success),
            "missing_fields": missing_fields,
            "failure_stage": failure_stage,
            "error_message": execution.error_message,
            "preflight_status": preflight,
            "backend": llm_client.backend,
            "llm_used": True,
            "planner_used": True,
            "verifier_used": verifier_artifact is not None,
            "planner_model": planner_artifact.generation.model if planner_artifact else llm_client.planner_model,
            "verifier_model": verifier_artifact.generation.model if verifier_artifact else llm_client.verifier_model,
            "request_mode": request_mode,
            "llm_user_prompt": user_goal,
            "planner_raw_output": planner_artifact.raw_response if planner_artifact else None,
            "initial_planner_raw_output": workflow_result.get("initial_planner_artifact").raw_response if workflow_result.get("initial_planner_artifact") else None,
            "replanner_raw_output": workflow_result.get("replanner_artifact").raw_response if workflow_result.get("replanner_artifact") else None,
            "task_spec": workflow_result["plan"].model_dump(mode="json"),
            "validation_result": {"valid": bool(workflow_result.get("final_plan_valid", True))},
            "benchmark_context": evaluator_context,
            "workflow_benchmark_context": workflow_context,
            "planning_mode": workflow_result.get("planning_mode"),
            "execution_logs": [log.model_dump(mode="json") for log in execution.logs],
            "used_skills": execution.used_skills,
            "expected_fields": expected_fields,
            "expected_skills": scenario.get("expected_skills") or [],
            "skill_coverage": _skill_coverage(scenario.get("expected_skills") or [], execution.used_skills),
            "extracted_data": extracted_data,
            "verifier_raw_output": verifier_artifact.raw_response if verifier_artifact else None,
            "verifier_verdict": verdict.model_dump(mode="json"),
            "export_path": [str(path) for path in export_paths],
            "artifact_paths": {key: str(value) if value else None for key, value in artifact_paths.items()},
            "screenshots": [execution.screenshot_path] if execution.screenshot_path else [],
            "runtime_sec": round(time.time() - started, 3),
            "planner_artifact": _artifact_to_json(planner_artifact),
            "verifier_artifact": _artifact_to_json(verifier_artifact),
            "blocked_or_limited_status": blocked_status or None,
            "rate_limit_detected": bool(execution.rate_limit_detected or failure_stage == "skipped_rate_limited"),
            "normalized_action_aliases": workflow_result.get("normalized_action_aliases") or [],
            "action_oov_detected": bool(workflow_result.get("action_oov_detected", False)),
            "allowed_controlled_outcomes": scenario.get("allowed_controlled_outcomes") or [],
            "notes": scenario.get("notes"),
        }
    except (WorkflowStageError, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        failure_stage = "timeout" if isinstance(exc, asyncio.TimeoutError) else getattr(exc, "stage", "runtime")
        if _is_llm_quota_error(exc):
            failure_stage = "llm_quota_exceeded"
        quota_exceeded = failure_stage == "llm_quota_exceeded"
        result = {
            "id": scenario_id,
            "site": scenario.get("site"),
            "category": scenario.get("category"),
            "url": scenario.get("url"),
            "user_goal": user_goal,
            "status": "skipped" if quota_exceeded else "failed",
            "success": False,
            "partial_success": False,
            "missing_fields": scenario.get("expected_fields") or [],
            "failure_stage": failure_stage,
            "error_message": str(exc),
            "preflight_status": preflight,
            "backend": backend,
            "llm_used": True,
            "planner_used": True,
            "verifier_used": False,
            "planner_model": getattr(llm_client, "planner_model", None),
            "verifier_model": getattr(llm_client, "verifier_model", None),
            "request_mode": request_mode,
            "llm_user_prompt": user_goal,
            "planner_raw_output": getattr(getattr(workflow, "planner", None), "last_artifact", None).raw_response
            if getattr(getattr(workflow, "planner", None), "last_artifact", None)
            else None,
            "task_spec": None,
            "validation_result": {"valid": False},
            "benchmark_context": evaluator_context,
            "workflow_benchmark_context": workflow_context,
            "used_skills": [],
            "expected_fields": scenario.get("expected_fields") or [],
            "expected_skills": scenario.get("expected_skills") or [],
            "skill_coverage": _skill_coverage(scenario.get("expected_skills") or [], []),
            "extracted_data": {},
            "blocked_or_limited_status": None,
            "rate_limit_detected": failure_stage == "skipped_rate_limited",
            "normalized_action_aliases": [],
            "action_oov_detected": False,
            "allowed_controlled_outcomes": scenario.get("allowed_controlled_outcomes") or [],
            "notes": scenario.get("notes"),
            "runtime_sec": round(time.time() - started, 3),
        }

    artifact_path = output_dir / f"{scenario_id}_{_timestamp()}.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_quota_skipped_result(
    scenario: dict[str, Any],
    *,
    backend: str | None,
    output_dir: Path,
    error_message: str,
) -> dict[str, Any]:
    started = time.time()
    scenario_id = str(scenario["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "id": scenario_id,
        "site": scenario.get("site"),
        "category": scenario.get("category"),
        "url": scenario.get("url"),
        "user_goal": scenario.get("goal"),
        "status": "skipped",
        "success": False,
        "partial_success": False,
        "missing_fields": scenario.get("expected_fields") or [],
        "failure_stage": "llm_quota_exceeded",
        "error_message": error_message,
        "preflight_status": {"status": "not_run_after_llm_quota_exceeded"},
        "backend": backend,
        "llm_used": False,
        "planner_used": False,
        "verifier_used": False,
        "used_skills": [],
        "expected_fields": scenario.get("expected_fields") or [],
        "expected_skills": scenario.get("expected_skills") or [],
        "skill_coverage": _skill_coverage(scenario.get("expected_skills") or [], []),
        "extracted_data": {},
        "blocked_or_limited_status": None,
        "rate_limit_detected": False,
        "normalized_action_aliases": [],
        "action_oov_detected": False,
        "runtime_sec": round(time.time() - started, 3),
    }
    artifact_path = output_dir / f"{scenario_id}_{_timestamp()}.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_summary(results: list[dict[str, Any]], *, suite: dict[str, Any]) -> dict[str, Any]:
    skipped = [result for result in results if result.get("status") == "skipped"]
    attempted = [result for result in results if result.get("status") != "skipped"]
    successful = [result for result in attempted if result.get("success")]
    partial = [result for result in attempted if result.get("partial_success") or result.get("status") == "partial_success"]
    failure_buckets: dict[str, int] = {}
    by_category: dict[str, dict[str, Any]] = {}
    expected_skill_counts: dict[str, int] = {}
    matched_skill_counts: dict[str, int] = {}
    for result in attempted:
        if result.get("success"):
            pass
        elif result.get("partial_success") or result.get("status") == "partial_success":
            key = str(result.get("failure_stage") or "partial_success")
            failure_buckets[key] = failure_buckets.get(key, 0) + 1
        else:
            key = str(result.get("failure_stage") or "failed")
            failure_buckets[key] = failure_buckets.get(key, 0) + 1
    for result in results:
        category = str(result.get("category") or "uncategorized")
        bucket = by_category.setdefault(
            category,
            {
                "total": 0,
                "attempted": 0,
                "skipped": 0,
                "successful": 0,
                "partial_success": 0,
                "failure_buckets": {},
            },
        )
        bucket["total"] += 1
        if result.get("status") == "skipped":
            bucket["skipped"] += 1
        else:
            bucket["attempted"] += 1
            if result.get("success"):
                bucket["successful"] += 1
            elif result.get("partial_success") or result.get("status") == "partial_success":
                bucket["partial_success"] += 1
            else:
                key = str(result.get("failure_stage") or "failed")
                bucket["failure_buckets"][key] = bucket["failure_buckets"].get(key, 0) + 1
        coverage = result.get("skill_coverage") if isinstance(result.get("skill_coverage"), dict) else {}
        for skill in coverage.get("expected_skills") or []:
            expected_skill_counts[skill] = expected_skill_counts.get(skill, 0) + 1
        for skill in coverage.get("matched_expected_skills") or []:
            matched_skill_counts[skill] = matched_skill_counts.get(skill, 0) + 1
    for bucket in by_category.values():
        attempted_count = int(bucket["attempted"])
        bucket["success_rate_attempted"] = (bucket["successful"] / attempted_count) if attempted_count else 0.0
        bucket["failure_buckets"] = dict(sorted(bucket["failure_buckets"].items()))
    skill_coverage = {
        skill: {
            "matched": matched_skill_counts.get(skill, 0),
            "expected": expected,
            "coverage": (matched_skill_counts.get(skill, 0) / expected) if expected else None,
        }
        for skill, expected in sorted(expected_skill_counts.items())
    }
    return {
        "suite_id": suite.get("suite_id", "real_web_skill_suite_v1"),
        "suite_type": suite.get("suite_type", "real_web_llm_rpa_skill_smoke"),
        "generated_at": datetime.now(UTC).isoformat(),
        "total_scenarios": len(results),
        "attempted_scenarios": len(attempted),
        "skipped_scenarios": len(skipped),
        "successful_scenarios": len(successful),
        "partial_success_scenarios": len(partial),
        "success_rate_attempted": (len(successful) / len(attempted)) if attempted else 0.0,
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "by_category": dict(sorted(by_category.items())),
        "skill_coverage": skill_coverage,
        "results": results,
    }


async def main_async(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    export_formats = args.export_format or ["json"]
    suite = load_suite(args.suite_path)
    include_optional = bool(args.include_optional or args.include_marketplaces)
    scenarios = select_scenarios(
        suite,
        scenario_ids=args.scenario_ids,
        limit=args.limit,
        include_optional=include_optional,
        category=args.category,
    )
    results = []
    quota_error_message = ""
    for index, scenario in enumerate(scenarios):
        result = await run_scenario(
            scenario,
            backend=args.backend,
            show_browser=args.show_browser,
            slow_mo=args.slow_mo,
            export_formats=export_formats,
            output_dir=args.output_dir,
            skip_preflight=bool(args.skip_preflight),
            task_timeout_sec=float(args.task_timeout_sec),
            llm_timeout_sec=args.llm_timeout_sec if args.llm_timeout_sec is not None else int(os.getenv("OLLAMA_TIMEOUT_SEC", "1000")),
            request_mode=str(args.request_mode),
            two_stage_planning=bool(args.two_stage_planning),
        )
        results.append(result)
        print(json.dumps({"id": result["id"], "status": result["status"], "success": result["success"], "partial_success": result.get("partial_success", False), "failure_stage": result.get("failure_stage")}, ensure_ascii=False))
        if result.get("failure_stage") == "llm_quota_exceeded":
            quota_error_message = str(result.get("error_message") or "LLM is unavailable; remaining scenarios were not attempted.")
            for remaining in scenarios[index + 1 :]:
                skipped = build_quota_skipped_result(
                    remaining,
                    backend=args.backend,
                    output_dir=args.output_dir,
                    error_message=quota_error_message,
                )
                results.append(skipped)
                print(json.dumps({"id": skipped["id"], "status": skipped["status"], "success": skipped["success"], "partial_success": skipped.get("partial_success", False), "failure_stage": skipped.get("failure_stage")}, ensure_ascii=False))
            break
    summary = build_summary(results, suite=suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"real_web_skill_suite_{_timestamp()}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "summary": {k: summary[k] for k in ["total_scenarios", "attempted_scenarios", "skipped_scenarios", "successful_scenarios", "partial_success_scenarios", "success_rate_attempted", "failure_buckets"]}}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
