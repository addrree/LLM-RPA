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
    "extract_card_items": "extract_by_intent",
    "extract_cards": "extract_by_intent",
    "extract_fields_from_region": "extract_by_intent",
}


ACTION_ALIAS_INTENTS: dict[str, str] = {
    "extract_card_items": "card_items",
    "extract_cards": "card_items",
    "extract_fields_from_region": "field_schema",
}

INTENT_ALIASES: dict[str, str] = {
    "card": "card_items",
    "cards": "card_items",
    "catalog": "card_items",
    "catalog_items": "card_items",
    "listing": "card_items",
    "listings": "card_items",
    "semantic_region_fields": "field_schema",
    "fields_schema": "field_schema",
    "region_fields": "field_schema",
    "extract_fields_from_region": "field_schema",
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


def normalize_required_field_alias(field: str) -> str:
    normalized_field = str(field or "").strip()
    if normalized_field.endswith("[]"):
        normalized_field = normalized_field[:-2]
    return REQUIRED_FIELD_ALIASES.get(normalized_field.lower(), normalized_field)


_FIELD_REQUEST_VERB = re.compile(
    r"\b(?:extract|export|return|collect|get|provide|list|выгрузи|извлеки|верни|получи|собери|перечисли)\b",
    flags=re.IGNORECASE,
)
_FIELD_LIST_SPLITTER = re.compile(
    r"\s*(?:,|;|\band\b|\bplus\b|\bas\s+well\s+as\b|\bи\b|\bа\s+также\b)\s*",
    flags=re.IGNORECASE,
)
_FIELD_LIST_INTRO = re.compile(
    r"\b(?:with|including|fields?|с\s+полями|включая)\b",
    flags=re.IGNORECASE,
)
_FIELD_QUALIFIER = re.compile(
    r"\b(?:if|when)\s+(?:available|present)\b|\bпри\s+наличии\b",
    flags=re.IGNORECASE,
)
_FIELD_LEADING_MODIFIERS = {
    "a",
    "an",
    "the",
    "all",
    "any",
    "visible",
    "requested",
    "following",
    "short",
    "brief",
    "general",
    "common",
    "все",
    "видимые",
    "следующие",
    "краткое",
    "краткую",
    "общие",
    "общую",
}


def _field_names_from_explicit_list(segment: str, *, explicit_intro: bool) -> list[str]:
    value = str(segment or "").strip()
    if not value:
        return []
    parts = [part.strip() for part in _FIELD_LIST_SPLITTER.split(value) if part.strip()]
    if not explicit_intro:
        has_list_punctuation = "," in value or ";" in value
        short_conjunction_list = len(parts) >= 2 and all(len(part.split()) <= 3 for part in parts)
        if not has_list_punctuation and not short_conjunction_list:
            return []

    fields: list[str] = []
    for part in parts:
        cleaned = _FIELD_QUALIFIER.sub("", part).strip(" .:()[]{}'\"")
        words = [word for word in re.findall(r"\w+", cleaned, flags=re.UNICODE) if word]
        while words and words[0].casefold() in _FIELD_LEADING_MODIFIERS:
            words.pop(0)
        if not words or len(words) > 6:
            continue
        field = "_".join(word.casefold() for word in words).strip("_")
        if field and field not in fields:
            fields.append(field)
    return fields


def _infer_explicit_goal_fields(goal: object) -> list[str]:
    text = str(goal or "").strip()
    if not text:
        return []

    candidates: list[tuple[str, bool]] = []
    if ":" in text:
        candidates.append((text.rsplit(":", 1)[1], True))

    matches = list(_FIELD_REQUEST_VERB.finditer(text))
    if matches:
        tail = text[matches[-1].end() :].strip()
        intro_matches = list(_FIELD_LIST_INTRO.finditer(tail))
        if intro_matches:
            candidates.append((tail[intro_matches[-1].end() :], True))
        candidates.append((tail, False))

    for segment, explicit_intro in candidates:
        fields = _field_names_from_explicit_list(segment, explicit_intro=explicit_intro)
        if fields:
            return fields
    return []


def infer_schema_required_fields(goal: object, required_fields: list[str] | None = None) -> list[str]:
    inferred: list[str] = []

    def add(field: str) -> None:
        if field not in inferred:
            inferred.append(field)

    for field in required_fields or []:
        normalized = str(field or "").strip()
        if normalized and normalized not in {"page_snapshot", "clicked_text"}:
            add(normalized)

    for field in _infer_explicit_goal_fields(goal):
        add(field)
    return inferred


def goal_requests_schema_fields(goal: object, required_fields: list[str] | None = None) -> bool:
    fields = infer_schema_required_fields(goal, required_fields)
    collection_keys = {
        "items",
        "results",
        "rows",
        "links",
        "cards",
    }
    return bool(fields) and not all(str(field).casefold() in collection_keys for field in fields)


def _field_rule_for_name(field: object) -> dict[str, Any]:
    name = str(field or "").strip()
    normalized = normalize_required_field_alias(name).casefold()
    if normalized in {"url", "final_url", "current_url", "source_url"} or normalized.endswith("_url"):
        return {"type": "current_url"}
    if normalized in {"title", "page_title", "current_title"}:
        return {"type": "page_title"}
    if normalized in {"description", "summary", "snippet", "meta_description"}:
        return {"type": "meta_description", "anchors": [name.replace("_", " ")]}
    if normalized in {"email", "phone"}:
        return {"type": normalized}
    return {"type": "text", "anchors": [name.replace("_", " ")]}


def build_field_schema_args(
    goal: object,
    required_fields: list[str] | None = None,
    output_key: str = "extracted_fields",
) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for field in infer_schema_required_fields(goal, required_fields):
        normalized = str(field or "").strip()
        if normalized:
            fields[normalized] = _field_rule_for_name(normalized)
    return {
        "intent": "field_schema",
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
    intent = normalize_intent_alias(args.get("intent"))
    if intent in {"field_schema", "card_items", "table_rows"}:
        return intent
    typed_field_schema = isinstance(fields, dict) and bool(fields) and all(
        isinstance(rule, dict)
        and any(key in rule for key in ("type", "anchors", "anchor_text", "anchor_candidates", "value_pattern"))
        and not any(key in rule for key in ("selector", "attr", "attribute"))
        for rule in fields.values()
    )
    if typed_field_schema or bool(args.get("region_hint")) or bool(args.get("region_candidates")):
        return "field_schema"
    selector_like_pattern = looks_like_css_selector(args.get("pattern", ""))
    fields_are_selector_like = isinstance(fields, dict) and any(
        isinstance(value, str) and looks_like_css_selector(value)
        for value in fields.values()
    )
    fields_have_selector_rules = isinstance(fields, dict) and any(
        isinstance(value, dict) and any(key in value for key in ("selector", "attr", "attribute"))
        for value in fields.values()
    )
    if fields_have_selector_rules:
        return None
    shape = str(args.get("shape", "") or "").strip().casefold()
    if shape in {"table", "rows", "grid"}:
        return "table_rows"
    if selector_like_pattern or fields_are_selector_like or bool(args.get("item_selector")) or isinstance(fields, dict):
        return "card_items"
    return None


def normalize_schema_fields_step(
    step: dict[str, Any],
    *,
    goal: object,
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    current = dict(step) if isinstance(step, dict) else {}
    args = current.get("args")
    current["args"] = dict(args) if isinstance(args, dict) else {}
    args = current["args"]
    action = str(current.get("action", "") or "").strip()

    if action == "extract_fields_from_region":
        action = "extract_by_intent"
        current["action"] = action
        args.setdefault("intent", "field_schema")

    if action == "click_by_semantic_target":
        target_candidates = args.get("target_candidates")
        candidates = [
            str(item).strip()
            for item in target_candidates
            if str(item).strip()
        ] if isinstance(target_candidates, list) else []
        target_text = str(args.get("target_text") or args.get("text") or args.get("target") or "").strip()
        if not target_text and candidates:
            args["target_text"] = candidates[0]
        return current

    intent = normalize_intent_alias(args.get("intent"))
    schema_hint = semantic_intent_for_structured_step(current) == "field_schema"
    has_anchor = bool(str(args.get("anchor_text") or args.get("anchor") or "").strip()) or (
        isinstance(args.get("anchor_candidates"), list)
        and any(str(item or "").strip() for item in args.get("anchor_candidates", []))
    )
    malformed_schema_extraction = goal_requests_schema_fields(goal, required_fields) and (
        action == "extract_value_near_anchor" and not has_anchor
    )
    explicit_schema = action == "extract_by_intent" and intent == "field_schema"
    if not (malformed_schema_extraction or schema_hint or explicit_schema):
        return current

    output_key = str(args.get("output_key") or current.get("save_as") or "extracted_fields").strip() or "extracted_fields"
    normalized_args = build_field_schema_args(goal, required_fields, output_key=output_key)
    if isinstance(args.get("fields"), dict) and args["fields"]:
        normalized_args["fields"] = dict(args["fields"])
    if isinstance(args.get("region_candidates"), list) and args["region_candidates"]:
        normalized_args["region_candidates"] = list(args["region_candidates"])
    if str(args.get("region_hint", "") or "").strip():
        normalized_args["region_hint"] = str(args["region_hint"]).strip()
    current["action"] = "extract_by_intent"
    current["args"] = normalized_args
    current["save_as"] = output_key
    return current


def canonical_structured_intent(value: str) -> str | None:
    normalized = normalize_intent_alias(value)
    if normalized == "field_schema":
        return "field_schema"
    if normalized in {"card", "cards", "card_item", "card_items", "catalog", "catalog_items", "listing", "listings"}:
        return "card_items"
    if normalized in {"table", "table_row", "table_rows", "row", "rows"}:
        return "table_rows"
    return None


def default_output_key_for_intent(intent: str) -> str:
    normalized = normalize_intent_alias(intent)
    return {
        "card_items": "items",
        "table_rows": "rows",
        "visible_links": "links",
        "extract_visible_links": "links",
        "links": "links",
        "field_schema": "extracted_fields",
    }.get(normalized, "items")


def coalesce_field_schema_steps(
    steps: list[dict[str, Any]],
    *,
    goal: str,
    required_fields: list[str],
) -> list[dict[str, Any]]:
    field_set = {
        str(field or "").strip()
        for field in required_fields
        if str(field or "").strip() and str(field or "").strip() not in {"page_snapshot", "clicked_text"}
    }
    if len(field_set) < 2 or not goal_requests_schema_fields(goal, list(field_set)):
        return steps

    field_step_indices: list[int] = []
    schema_fields: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        save_as = str(step.get("save_as", "") or "").strip()
        if save_as not in field_set:
            continue
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        fragile_field_action = action in {
            "extract_text",
            "extract_html",
            "extract_pattern_from_page_text",
            "extract_text_near_text",
            "extract_value_near_anchor",
        }
        if fragile_field_action:
            field_step_indices.append(index)
            schema_fields[save_as] = _field_rule_for_name(save_as)
    if len(field_step_indices) < 2:
        return steps

    insert_at = field_step_indices[0]
    skip = set(field_step_indices)
    coalesced: list[dict[str, Any]] = []
    inserted = False
    for index, step in enumerate(steps):
        if index == insert_at and not inserted:
            coalesced.append(
                {
                    "step_id": 0,
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "field_schema",
                        "fields": schema_fields,
                        "output_key": "extracted_fields",
                    },
                    "save_as": "extracted_fields",
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
