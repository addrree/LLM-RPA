from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_FIELD_ALIASES: dict[str, str] = {
    "url": "final_url",
    "current_url": "final_url",
    "title": "page_title",
    "current_title": "page_title",
}

PACKAGE_METADATA_FIELDS = {
    "package",
    "package_metadata",
    "package_name",
    "name",
    "latest_version",
    "version",
    "description",
    "summary",
}

ACTION_ALIASES: dict[str, str] = {
    "enter_text": "fill_by_semantic_target",
    "input_text": "fill_by_semantic_target",
    "fill_input": "fill_by_semantic_target",
    "fill_by_selector": "fill",
    "type_text": "fill_by_semantic_target",
    "select": "select_option",
    "choose_list": "select_option",
    "autocomplete": "select_autocomplete",
    "choose_autocomplete": "choose_autocomplete_suggestion",
    "click_button": "click_by_semantic_target",
    "click_link": "click_by_semantic_target",
    "click_text": "click_by_semantic_target",
    "click_by_selector": "click",
    "submit_search": "click_by_semantic_target",
    "click_semantic": "click_by_semantic_target",
    "semantic_click": "click_by_semantic_target",
    "fill_semantic": "fill_by_semantic_target",
    "semantic_fill": "fill_by_semantic_target",
    "select_semantic": "select_by_semantic_target",
    "visible_links": "extract_visible_links",
    "extract_results": "extract_by_intent",
    "extract_search_results": "extract_by_intent",
    "extract_package_info": "extract_by_intent",
    "extract_product_cards": "extract_by_intent",
    "extract_card_items": "extract_by_intent",
    "extract_cards": "extract_by_intent",
    "extract_article_links": "extract_by_intent",
    "extract_fields_from_region": "extract_by_intent",
}


ACTION_ALIAS_INTENTS: dict[str, str] = {
    "extract_results": "search_results",
    "extract_search_results": "search_results",
    "extract_package_info": "package_metadata",
    "extract_product_cards": "product_cards",
    "extract_card_items": "card_items",
    "extract_cards": "card_items",
    "extract_article_links": "article_results",
}

INTENT_ALIASES: dict[str, str] = {
    "package": "package_metadata",
    "packages": "package_metadata",
    "package_info": "package_metadata",
    "library": "package_metadata",
    "library_info": "package_metadata",
    "library_metadata": "package_metadata",
    "card": "card_items",
    "cards": "card_items",
    "catalog": "card_items",
    "catalog_items": "card_items",
    "listing": "card_items",
    "listings": "card_items",
    "currency": "table_rows",
    "currency_row": "table_rows",
    "currency_rows": "table_rows",
    "currency_table_rows": "table_rows",
    "field_schema": "semantic_region_fields",
    "fields_schema": "semantic_region_fields",
    "region_fields": "semantic_region_fields",
    "extract_fields_from_region": "semantic_region_fields",
}


CANONICAL_ACTIONS = {
    "open_url",
    "click",
    "navigate_to_relevant_section",
    "type",
    "fill",
    "focus",
    "clear",
    "press",
    "hover",
    "select_option",
    "check",
    "uncheck",
    "select_autocomplete",
    "choose_autocomplete_suggestion",
    "choose_date",
    "click_by_semantic_target",
    "fill_by_semantic_target",
    "select_by_semantic_target",
    "wait_for",
    "extract_text",
    "extract_html",
    "extract_items",
    "extract_structured_items",
    "extract_section_lines",
    "extract_value_from_section",
    "extract_structured_items_from_region",
    "compare_structured_values",
    "assert_page_contains",
    "screenshot",
    "observe_page",
    "extract_by_intent",
    "extract_visible_links",
    "extract_pattern_from_page_text",
    "extract_text_near_text",
    "extract_value_near_anchor",
    "find_row_by_condition",
    "click_row_action",
    "visual_observe",
    "visual_extract_object_count",
    "visual_click_by_geometry",
    "finish",
}


CONTACT_REGION_CANDIDATES = [
    "Контакты",
    "Contacts",
    "Contact",
    "Связаться",
    "Support",
    "Поддержка",
    "About",
    "О нас",
]


def goal_requests_semantic_region_fields(goal: object, required_fields: list[str] | None = None) -> bool:
    text = str(goal or "").casefold()
    fields = {str(field or "").strip().casefold() for field in (required_fields or [])}
    contact_terms = (
        "contact",
        "contacts",
        "support",
        "about",
        "контакт",
        "связ",
        "поддерж",
        "реквизит",
        "сведения об организа",
    )
    field_terms = (
        "email",
        "e-mail",
        "mail",
        "phone",
        "telephone",
        "address",
        "почт",
        "телефон",
        "адрес",
    )
    schema_fields = {"email", "phone", "address", "contact_page_url", "final_url", "url"}
    return any(term in text for term in contact_terms) and (
        any(term in text for term in field_terms)
        or bool(fields & schema_fields)
        or any(term in text for term in ("данн", "информац", "сведения", "details", "info", "link", "page", "ссылк", "страниц"))
    )


def infer_semantic_region_required_fields(goal: object, required_fields: list[str] | None = None) -> list[str]:
    text = str(goal or "").casefold()
    inferred = [str(field or "").strip() for field in (required_fields or []) if str(field or "").strip()]

    def add(field: str) -> None:
        if field not in inferred:
            inferred.append(field)

    broad_contact = any(term in text for term in ("contact data", "contact details", "контактные данн", "контактную информац", "реквизит", "сведения об организа"))
    if broad_contact:
        for field in ("address", "phone", "email", "contact_page_url"):
            add(field)
    if any(term in text for term in ("address", "адрес")):
        add("address")
    if any(term in text for term in ("phone", "telephone", "tel", "телефон")):
        add("phone")
    if any(term in text for term in ("email", "e-mail", "mail", "почт")):
        add("email")
    if any(term in text for term in ("contact page", "contacts page", "contact link", "ссылк", "страниц")) and any(term in text for term in ("contact", "контакт")):
        add("contact_page_url")
    return inferred


def build_semantic_region_fields_args(goal: object, required_fields: list[str] | None = None, output_key: str = "contact_info") -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for field in infer_semantic_region_required_fields(goal, required_fields):
        normalized = normalize_required_field_alias(field)
        if normalized in {"email"}:
            fields["email"] = {"type": "email"}
        elif normalized in {"phone"}:
            fields["phone"] = {"type": "phone"}
        elif normalized in {"address"}:
            fields["address"] = {"type": "text", "anchors": ["Адрес", "Address"]}
        elif normalized in {"contact_page_url", "final_url", "url"}:
            fields["contact_page_url"] = {"type": "current_url"}
    if not fields:
        fields = {
            "address": {"type": "text", "anchors": ["Адрес", "Address"]},
            "phone": {"type": "phone"},
            "email": {"type": "email"},
            "contact_page_url": {"type": "current_url"},
        }
    return {
        "intent": "semantic_region_fields",
        "region_hint": "contacts/support/about/details",
        "region_candidates": CONTACT_REGION_CANDIDATES,
        "fields": fields,
        "output_key": output_key,
    }


def normalize_required_field_alias(field: str) -> str:
    normalized_field = str(field or "").strip()
    if normalized_field.endswith("[]"):
        normalized_field = normalized_field[:-2]
    return REQUIRED_FIELD_ALIASES.get(normalized_field.lower(), normalized_field)


CONTACT_REGION_CANDIDATES = [
    "Контакты",
    "Contacts",
    "Contact",
    "Связаться",
    "Support",
    "Поддержка",
    "About",
    "О нас",
]


def goal_requests_semantic_region_fields(goal: object, required_fields: list[str] | None = None) -> bool:
    text = str(goal or "").casefold()
    fields = {str(field or "").strip().casefold() for field in (required_fields or [])}
    contact_terms = (
        "contact",
        "contacts",
        "support",
        "about",
        "контакт",
        "связ",
        "поддерж",
        "реквизит",
        "сведения об организа",
    )
    field_terms = (
        "email",
        "e-mail",
        "mail",
        "phone",
        "telephone",
        "address",
        "почт",
        "телефон",
        "адрес",
    )
    schema_fields = {"email", "phone", "address", "contact_page_url", "final_url", "url"}
    return any(term in text for term in contact_terms) and (
        any(term in text for term in field_terms)
        or bool(fields & schema_fields)
        or any(term in text for term in ("данн", "информац", "сведения", "details", "info", "link", "page", "ссылк", "страниц"))
    )


def infer_semantic_region_required_fields(goal: object, required_fields: list[str] | None = None) -> list[str]:
    text = str(goal or "").casefold()
    inferred = [str(field or "").strip() for field in (required_fields or []) if str(field or "").strip()]

    def add(field: str) -> None:
        if field not in inferred:
            inferred.append(field)

    broad_contact = any(term in text for term in ("contact data", "contact details", "контактные данн", "контактную информац", "реквизит", "сведения об организа"))
    if broad_contact:
        for field in ("address", "phone", "email", "contact_page_url"):
            add(field)
    if any(term in text for term in ("address", "адрес")):
        add("address")
    if any(term in text for term in ("phone", "telephone", "tel", "телефон")):
        add("phone")
    if any(term in text for term in ("email", "e-mail", "mail", "почт")):
        add("email")
    if any(term in text for term in ("contact page", "contacts page", "contact link", "ссылк", "страниц")) and any(term in text for term in ("contact", "контакт")):
        add("contact_page_url")
    return inferred


def build_semantic_region_fields_args(goal: object, required_fields: list[str] | None = None, output_key: str = "contact_info") -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for field in infer_semantic_region_required_fields(goal, required_fields):
        normalized = normalize_required_field_alias(field)
        if normalized in {"email"}:
            fields["email"] = {"type": "email"}
        elif normalized in {"phone"}:
            fields["phone"] = {"type": "phone"}
        elif normalized in {"address"}:
            fields["address"] = {"type": "text", "anchors": ["Адрес", "Address"]}
        elif normalized in {"contact_page_url", "final_url", "url"}:
            fields["contact_page_url"] = {"type": "current_url"}
    if not fields:
        fields = {
            "address": {"type": "text", "anchors": ["Адрес", "Address"]},
            "phone": {"type": "phone"},
            "email": {"type": "email"},
            "contact_page_url": {"type": "current_url"},
        }
    return {
        "intent": "semantic_region_fields",
        "region_hint": "contacts/support/about/details",
        "region_candidates": CONTACT_REGION_CANDIDATES,
        "fields": fields,
        "output_key": output_key,
    }


def normalize_required_field_aliases(fields: list[str]) -> list[str]:
    normalized: list[str] = []
    for field in fields:
        key = normalize_required_field_alias(field)
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def normalize_intent_alias(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    return INTENT_ALIASES.get(normalized, normalized)


def looks_like_css_selector(value: object) -> bool:
    selector = str(value or "").strip()
    if not selector:
        return False
    if re.search(r"[#.\[>:,*+~]", selector):
        return True
    if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", selector):
        return True
    return False


def semantic_intent_for_structured_step(step: dict[str, Any]) -> str | None:
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    fields = args.get("fields")
    hint = " ".join(
        [
            str(args.get("intent", "")),
            str(args.get("item_type", "")),
            str(args.get("type", "")),
            str(args.get("output_key", "")),
            str(step.get("save_as", "")),
            str(args.get("pattern", "")),
            " ".join(str(key) for key in fields.keys()) if isinstance(fields, dict) else "",
        ]
    ).casefold()
    selector_like_pattern = looks_like_css_selector(args.get("pattern", ""))
    fields_are_selector_like = isinstance(fields, dict) and any(
        isinstance(value, str) and looks_like_css_selector(value)
        for value in fields.values()
    )
    has_structural_hint = selector_like_pattern or fields_are_selector_like or bool(args.get("item_selector"))
    if not has_structural_hint and not any(
        token in hint
        for token in ("article", "articles", "news", "paper", "repository", "repo", "product", "card", "catalog", "listing", "table")
    ):
        return None
    if any(token in hint for token in ("product", "products", "product_cards")):
        return "product_cards"
    if any(token in hint for token in ("card", "cards", "card_items", "catalog", "listing", "listings")):
        return "card_items"
    if any(token in hint for token in ("repository", "repositories", "repo")):
        return "repository_results"
    if any(token in hint for token in ("paper", "papers", "preprint")):
        return "paper_results"
    if any(token in hint for token in ("news", "news_items")):
        return "news_items"
    if any(token in hint for token in ("article", "articles")):
        return "article_results"
    if "table" in hint:
        return "table_rows"
    return None


def item_type_args_for_intent(intent: str) -> dict[str, str]:
    if intent in {"article_results", "paper_results", "repository_results"}:
        return {"item_type": intent.replace("_results", "")}
    if intent == "news_items":
        return {"item_type": "news"}
    return {}


def canonical_structured_intent(value: str) -> str | None:
    normalized = normalize_intent_alias(value)
    if normalized == "package_metadata":
        return "package_metadata"
    if normalized in {"card", "cards", "card_item", "card_items", "catalog", "catalog_items", "listing", "listings"}:
        return "card_items"
    if normalized in {"product", "products", "product_card", "product_cards"}:
        return "product_cards"
    if normalized in {"repository", "repositories", "repo", "repo_results", "repository_results"}:
        return "repository_results"
    if normalized in {"paper", "papers", "preprint", "paper_results"}:
        return "paper_results"
    if normalized in {"article", "articles", "article_results"}:
        return "article_results"
    if normalized in {"news", "news_item", "news_items"}:
        return "news_items"
    if normalized in {"table", "table_row", "table_rows", "row", "rows"}:
        return "table_rows"
    if normalized in {"currency", "currency_row", "currency_rows", "currency_table_rows"}:
        return "table_rows"
    if normalized in {"search", "search_result", "search_results", "results"}:
        return "search_results"
    return normalized if normalized.endswith("_results") else None


def default_output_key_for_intent(intent: str) -> str:
    return {
        "product_cards": "products",
        "card_items": "cards",
        "repository_results": "repositories",
        "paper_results": "papers",
        "article_results": "articles",
        "news_items": "news",
        "currency_table_rows": "rows",
        "table_rows": "rows",
        "package_metadata": "package_metadata",
        "package_info": "package_metadata",
        "library_metadata": "package_metadata",
        "package": "package_metadata",
        "visible_links": "links",
        "extract_visible_links": "links",
        "links": "links",
        "search_results": "results",
        "result_list": "results",
    }.get(intent, "results")


def coalesce_package_metadata_steps(
    steps: list[dict[str, Any]],
    *,
    goal: str,
    required_fields: list[str],
) -> list[dict[str, Any]]:
    field_set = {normalize_required_field_alias(str(field)).strip().lower() for field in required_fields}
    goal_hint = str(goal or "").casefold()
    package_requested = (
        len(field_set & PACKAGE_METADATA_FIELDS) >= 2
        or ("package" in goal_hint and any(token in goal_hint for token in ("version", "description", "summary")))
    )
    if not package_requested:
        return steps

    package_step_indices: list[int] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        save_as = str(step.get("save_as", "") or "").strip().lower()
        if save_as not in PACKAGE_METADATA_FIELDS:
            continue
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        intent = str(args.get("intent", "") or "").strip().casefold()
        fragile_package_action = action in {
            "extract_text",
            "extract_html",
            "extract_pattern_from_page_text",
            "extract_text_near_text",
            "extract_value_near_anchor",
        } or (action == "extract_by_intent" and intent in {"value_near_anchor", "text_near_anchor"})
        if fragile_package_action:
            package_step_indices.append(index)
    if not package_step_indices:
        return steps

    insert_at = package_step_indices[0]
    skip = set(package_step_indices)
    coalesced: list[dict[str, Any]] = []
    inserted = False
    for index, step in enumerate(steps):
        if index == insert_at and not inserted:
            coalesced.append(
                {
                    "step_id": 0,
                    "action": "extract_by_intent",
                    "args": {"intent": "package_metadata", "output_key": "package_metadata"},
                    "save_as": "package_metadata",
                }
            )
            inserted = True
        if index in skip:
            continue
        coalesced.append(step)

    for index, step in enumerate(coalesced, start=1):
        if isinstance(step, dict):
            step["step_id"] = index
    return coalesced


def normalize_plan_action_aliases(payload: dict[str, Any] | list[Any]) -> tuple[dict[str, Any], bool]:
    if isinstance(payload, list):
        plan = {"steps": payload}
    else:
        plan = dict(payload) if isinstance(payload, dict) else {}
    steps = plan.get("steps")
    if not isinstance(steps, list) and isinstance(plan.get("actions"), list):
        steps = plan.pop("actions")
        plan["steps"] = steps
    if not isinstance(steps, list) and isinstance(plan.get("tasks"), list):
        steps = plan.pop("tasks")
        plan["steps"] = steps
    if not isinstance(steps, list) and isinstance(plan.get("order"), list):
        steps = plan.pop("order")
        plan["steps"] = steps
    if not isinstance(steps, list):
        return plan, False

    action_oov_detected = False
    invalid_actions: list[dict[str, Any]] = []
    normalized_aliases: list[dict[str, str]] = []
    normalized_steps: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            normalized_steps.append(raw_step)
            continue
        step = dict(raw_step)
        action = step.get("action")
        if isinstance(action, str):
            original_action = action
            normalized_action = action.strip()
            canonical = ACTION_ALIASES.get(normalized_action, normalized_action)
            step["action"] = canonical
            if canonical != original_action:
                normalized_aliases.append({"from": original_action, "to": canonical})
            if canonical == "extract_by_intent":
                args = step.get("args")
                if not isinstance(args, dict):
                    args = {}
                    step["args"] = args
                alias_intent = ACTION_ALIAS_INTENTS.get(normalized_action)
                if alias_intent and not str(args.get("intent", "")).strip():
                    args["intent"] = alias_intent
                if str(args.get("intent", "")).strip():
                    args["intent"] = normalize_intent_alias(args.get("intent"))
            if normalized_action == "submit_search" and canonical == "click_by_semantic_target":
                args = step.get("args")
                if not isinstance(args, dict):
                    args = {}
                    step["args"] = args
                args.setdefault("target_text", "search")
            if canonical not in CANONICAL_ACTIONS:
                action_oov_detected = True
                invalid_actions.append(
                    {
                        "step_id": step.get("step_id"),
                        "invalid_action": canonical,
                        "original_action": original_action,
                    }
                )
        normalized_steps.append(step)

    plan["steps"] = normalized_steps
    if normalized_aliases:
        plan["_normalized_action_aliases"] = normalized_aliases
    if invalid_actions:
        plan["_invalid_actions"] = invalid_actions
    return plan, action_oov_detected


class PlannerValidationFailed(ValueError):
    def __init__(self, diagnostics: dict[str, Any]):
        super().__init__(f"planner_validation_failed: {json.dumps(diagnostics, ensure_ascii=False)}")
        self.diagnostics = diagnostics


def raise_for_invalid_plan_actions(
    plan: dict[str, Any],
    *,
    profile_diagnostics: dict[str, Any] | None = None,
    allowed_actions: list[str] | set[str] | None = None,
) -> None:
    invalid_actions = plan.get("_invalid_actions")
    if not isinstance(invalid_actions, list) or not invalid_actions:
        return
    first = invalid_actions[0] if isinstance(invalid_actions[0], dict) else {}
    diagnostics = {
        "failure_stage": "planner_validation_failed",
        "invalid_action": first.get("invalid_action"),
        "invalid_actions": invalid_actions,
        "allowed_actions": sorted(allowed_actions or CANONICAL_ACTIONS),
    }
    if isinstance(profile_diagnostics, dict):
        diagnostics.update(
            {
                "task_type": profile_diagnostics.get("task_type"),
                "preferred_runtime_intents": profile_diagnostics.get("preferred_runtime_intents", []),
                "conceptual_profile_intents": profile_diagnostics.get("conceptual_profile_intents", []),
                "full_vocabulary_was_used": profile_diagnostics.get("full_vocabulary_was_used", False),
            }
        )
    raise PlannerValidationFailed(
        diagnostics
    )
