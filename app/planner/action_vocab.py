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
    "result": "search_results",
    "results": "search_results",
    "search_result": "search_results",
    "search_results": "search_results",
    "page_summary": "text_block",
    "section_summary": "text_block",
    "text_block": "text_block",
    "first_paragraph": "text_block",
    "anchor_object": "anchor_object",
    "label_value_object": "anchor_object",
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
# The older regex literals above intentionally remain for compatibility with
# already mojibaked prompts; these overrides add normal Cyrillic spellings for
# generic field-list parsing.
_FIELD_REQUEST_VERB = re.compile(
    r"\b(?:extract|export|return|collect|get|provide|list|find|"
    r"выгрузи|извлеки|верни|получи|собери|перечисли|найди|"
    r"РІС‹РіСЂСѓР·Рё|РёР·РІР»РµРєРё|РІРµСЂРЅРё|РїРѕР»СѓС‡Рё|СЃРѕР±РµСЂРё|РїРµСЂРµС‡РёСЃР»Рё)\b",
    flags=re.IGNORECASE,
)
_FIELD_LIST_SPLITTER = re.compile(
    r"\s*(?:,|;|\band\b|\bplus\b|\bas\s+well\s+as\b|\bи\b|\bа\s+также\b|\bРё\b|\bР°\s+С‚Р°РєР¶Рµ\b)\s*",
    flags=re.IGNORECASE,
)
_FIELD_LIST_INTRO = re.compile(
    r"\b(?:with|including|fields?|с\s+полями|включая|СЃ\s+РїРѕР»СЏРјРё|РІРєР»СЋС‡Р°СЏ)\b",
    flags=re.IGNORECASE,
)
_FIELD_QUALIFIER = re.compile(
    r"\b(?:if|when)\s+(?:available|present)\b|\bпри\s+наличии\b|\bРїСЂРё\s+РЅР°Р»РёС‡РёРё\b",
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
    if (
        normalized in {"description", "summary", "snippet", "meta_description"}
        or any(token in normalized for token in ("description", "summary", "snippet", "overview"))
        or any(token in normalized for token in ("описан", "аннотац", "предложен", "абзац", "кратк"))
    ):
        return {"type": "meta_description", "anchors": [name.replace("_", " ")]}
    if (
        normalized in {"email", "e_mail"}
        or "email" in normalized
        or "e_mail" in normalized
        or ("почт" in normalized and "адрес" not in normalized)
        or ("электрон" in normalized and "почт" in normalized)
    ):
        return {"type": "email"}
    if normalized in {"phone", "tel", "telephone"} or any(token in normalized for token in ("phone", "telephone", "телефон", "тел")):
        return {"type": "phone"}
    if any(token in normalized for token in ("count", "number", "total", "числ", "количеств")):
        return {"type": "number", "anchors": [name.replace("_", " ")]}
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
    if intent in {"field_schema", "card_items", "table_rows", "text_block", "search_results", "anchor_object"}:
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

    if action == "extract_items":
        return current

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
    if normalized in {"text_block", "page_summary", "section_summary", "first_paragraph"}:
        return "text_block"
    if normalized in {"search_result", "search_results", "result", "results"}:
        return "search_results"
    if normalized in {"anchor_object", "label_value_object"}:
        return "anchor_object"
    if normalized in {"card", "cards", "card_item", "card_items", "catalog", "catalog_items", "listing", "listings"}:
        return "card_items"
    if normalized in {"table", "table_row", "table_rows", "row", "rows"}:
        return "table_rows"
    return None


def default_output_key_for_intent(intent: str) -> str:
    normalized = normalize_intent_alias(intent)
    return {
        "card_items": "items",
        "search_results": "search_results",
        "table_rows": "rows",
        "text_block": "summary",
        "anchor_object": "metadata",
        "visible_links": "links",
        "extract_visible_links": "links",
        "links": "links",
        "field_schema": "extracted_fields",
    }.get(normalized, "items")


TECHNICAL_PLAN_FIELDS = {
    "page_snapshot",
    "clicked_text",
    "final_url",
    "page_title",
    "current_url",
    "url",
    "title",
    "visible_links",
    "links",
    "search_results",
    "results",
}

_COLLECTION_PARENT_HINTS = {
    "items",
    "item",
    "cards",
    "card_items",
    "stories",
    "articles",
    "modules",
    "products",
    "listings",
    "catalog_items",
    "remaining_items",
    "list_items",
    "rows",
    "table_rows",
    "results",
    "search_results",
}

_BROAD_SEMANTIC_CLICK_TARGETS = {
    "a",
    "an",
    "the",
    "link",
    "links",
    "button",
    "buttons",
    "item",
    "items",
    "card",
    "cards",
    "result",
    "results",
    "entry",
    "entries",
    "article",
    "articles",
    "story",
    "stories",
    "\u0441\u0441\u044b\u043b\u043a\u0430",
    "\u0441\u0441\u044b\u043b\u043a\u0443",
    "\u0441\u0441\u044b\u043b\u043a\u0438",
    "\u043a\u043d\u043e\u043f\u043a\u0430",
    "\u043a\u043d\u043e\u043f\u043a\u0443",
    "\u043a\u043d\u043e\u043f\u043a\u0438",
    "\u044d\u043b\u0435\u043c\u0435\u043d\u0442",
    "\u044d\u043b\u0435\u043c\u0435\u043d\u0442\u044b",
    "\u043f\u0443\u043d\u043a\u0442",
    "\u043f\u0443\u043d\u043a\u0442\u044b",
    "\u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0430",
    "\u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0438",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b",
}


def is_broad_semantic_click_target(target_text: object) -> bool:
    text = str(target_text or "").strip().casefold()
    if not text:
        return False
    compact = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()
    compact = re.sub(
        r"^(?:click|open|follow|press|go to|navigate to|"
        r"\u043d\u0430\u0436\u043c\u0438|\u043a\u043b\u0438\u043a\u043d\u0438|\u043e\u0442\u043a\u0440\u043e\u0439|\u043f\u0435\u0440\u0435\u0439\u0434\u0438)\s+",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    compact = re.sub(r"^(?:the|a|an)\s+", "", compact).strip()
    if compact in _BROAD_SEMANTIC_CLICK_TARGETS:
        return True
    return bool(re.fullmatch(r"(?:visible\s+)?(?:link|button|item|card|result|story|article)s?", compact))


def _text_variants_with_mojibake_repairs(value: object) -> list[str]:
    text = str(value or "")
    variants = [text]
    for encoding in ("cp1251", "latin1"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired and repaired != text and repaired not in variants:
            variants.append(repaired)
    return variants


def _casefold_with_mojibake_repairs(value: object) -> str:
    return " ".join(item.casefold() for item in _text_variants_with_mojibake_repairs(value))


def casefold_with_mojibake_repairs(value: object) -> str:
    return _casefold_with_mojibake_repairs(value)


def anchor_object_args_for_goal(goal: object, required_fields: list[str] | None = None) -> dict[str, Any]:
    folded_goal = _casefold_with_mojibake_repairs(goal)
    normalized_fields = [
        _casefold_with_mojibake_repairs(normalize_required_field_alias(str(field or "")))
        for field in (required_fields or [])
        if str(field or "").strip()
    ]
    combined = " ".join([folded_goal, *normalized_fields])
    wants_language = any(token in combined for token in ("language", "\u044f\u0437\u044b\u043a", "\u043d\u0430\u0437\u0432\u0430\u043d"))
    wants_count = any(token in combined for token in ("count", "number", "articles", "article_count", "\u0447\u0438\u0441\u043b", "\u0441\u0442\u0430\u0442", "\u043a\u043e\u043b\u0438\u0447"))
    wants_article_count = any(token in combined for token in ("article", "articles", "article_count", "\u0441\u0442\u0430\u0442"))
    wants_near = any(token in combined for token in ("near", "next to", "beside", "\u0440\u044f\u0434\u043e\u043c", "\u0432\u043e\u0437\u043b\u0435", "\u043e\u043a\u043e\u043b\u043e"))
    if not (wants_language and wants_article_count and wants_count and (wants_near or "article_count" in combined)):
        return {}

    anchor_text = ""
    for variant in _text_variants_with_mojibake_repairs(goal):
        for pattern in (
            r"(?:language|langue|idioma)\s+([^\s,.;:]+)",
            "(?:\u044f\u0437\u044b\u043a\\w*|\u044f\u0437\u044b\u043a\u0430)\\s+([^\\s,.;:]+)",
            r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’_-]{2,})\b",
        ):
            match = re.search(pattern, variant, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" :;,.\"'")
            if candidate and not candidate.lower().startswith(("http", "www")):
                anchor_text = candidate
                break
        if anchor_text:
            break

    fields: dict[str, dict[str, Any]] = {}
    for raw_field in required_fields or []:
        field = str(raw_field or "").strip()
        if not field:
            continue
        normalized = _casefold_with_mojibake_repairs(normalize_required_field_alias(field))
        if normalized in TECHNICAL_PLAN_FIELDS:
            continue
        if any(token in normalized for token in ("language", "name", "title", "label", "\u044f\u0437\u044b\u043a", "\u043d\u0430\u0437\u0432\u0430\u043d")):
            fields["language_name"] = {"type": "text"}
        elif any(token in normalized for token in ("count", "number", "total", "article", "\u0447\u0438\u0441\u043b", "\u0441\u0442\u0430\u0442", "\u043a\u043e\u043b\u0438\u0447")):
            fields["article_count"] = {"type": "number"}
    if wants_language:
        fields.setdefault("language_name", {"type": "text"})
    if wants_count:
        fields.setdefault("article_count", {"type": "number"})
    if not fields:
        return {}
    return {
        "intent": "anchor_object",
        "fields": fields,
        **({"anchor_text": anchor_text} if anchor_text else {}),
        "output_key": "metadata",
    }


def repair_anchor_object_plan_steps(
    steps: list[dict[str, Any]],
    *,
    goal: object,
    required_fields: list[str],
    preferred_intents: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    anchor_args = anchor_object_args_for_goal(goal, required_fields)
    if not anchor_args:
        return steps, required_fields
    normalized_preferred = {normalize_intent_alias(item) for item in (preferred_intents or set()) if str(item).strip()}
    if normalized_preferred and "anchor_object" not in normalized_preferred:
        return steps, required_fields

    output_key = str(anchor_args.get("output_key") or default_output_key_for_intent("anchor_object")).strip()
    anchor_fields = [str(field).strip() for field in (anchor_args.get("fields") or {}).keys() if str(field).strip()]

    def is_anchor_candidate_step(step: dict[str, Any]) -> bool:
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        if action == "extract_by_intent":
            intent = normalize_intent_alias(args.get("intent", ""))
            return intent in {
                "",
                "field_schema",
                "value_near_anchor",
                "anchor_object",
                "text_block",
                "card_items",
                "cards",
                "table_rows",
                "rows",
                "search_results",
            }
        return action in {
            "extract_text",
            "extract_html",
            "extract_items",
            "extract_structured_items",
            "extract_pattern_from_page_text",
            "extract_text_near_text",
            "extract_value_near_anchor",
        }

    def is_anchor_projection(step: dict[str, Any]) -> bool:
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        save_as = normalize_required_field_alias(str(step.get("save_as", "") or args.get("output_key", "") or "").strip())
        if save_as and save_as in anchor_fields:
            return True
        if action == "extract_by_intent" and normalize_intent_alias(args.get("intent", "")) in {
            "field_schema",
            "value_near_anchor",
            "anchor_object",
            "text_block",
        }:
            fields = args.get("fields")
            if isinstance(fields, dict):
                return any(normalize_required_field_alias(str(field).strip()) in anchor_fields for field in fields)
        return False

    repaired: list[dict[str, Any]] = []
    inserted = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        if action == "click_by_semantic_target" and is_broad_semantic_click_target(
            args.get("target_text") or args.get("target") or args.get("text")
        ):
            continue
        if not inserted and is_anchor_candidate_step(step):
            current = dict(step)
            current["action"] = "extract_by_intent"
            current["args"] = dict(anchor_args)
            limit = args.get("limit")
            if isinstance(limit, int) and limit > 0:
                current["args"]["limit"] = limit
            if output_key:
                current["args"]["output_key"] = output_key
            current.pop("save_as", None)
            repaired.append(current)
            inserted = True
            continue
        if inserted and is_anchor_projection(step):
            continue
        repaired.append(step)

    if not inserted:
        insert_index = next((idx for idx, step in enumerate(repaired) if step.get("action") == "finish"), len(repaired))
        args = dict(anchor_args)
        if output_key:
            args["output_key"] = output_key
        repaired.insert(
            insert_index,
            {
                "step_id": 0,
                "action": "extract_by_intent",
                "args": args,
            },
        )

    preserved_technical = [
        normalize_required_field_alias(str(field or "").strip())
        for field in required_fields
        if normalize_required_field_alias(str(field or "").strip()) in {"final_url", "page_title"}
    ]
    updated_required: list[str] = []
    for field in [*preserved_technical, *anchor_fields]:
        if field and field not in updated_required:
            updated_required.append(field)
    return repaired, (updated_required or required_fields)


def collection_intent_for_goal(goal: object, required_fields: list[str] | None = None) -> str:
    text = _casefold_with_mojibake_repairs(goal)
    normalized_required = " ".join(
        _casefold_with_mojibake_repairs(normalize_required_field_alias(str(field or "")))
        for field in (required_fields or [])
        if str(field or "").strip()
    )
    combined = f"{text} {normalized_required}"
    if any(marker in combined for marker in ("visible link", "visible_links", "link list", "\u0441\u0441\u044b\u043b\u043a")):
        return "visible_links"
    if any(marker in combined for marker in ("table", "table row", "table_rows", "row ", " rows", "\u0442\u0430\u0431\u043b\u0438\u0446", "\u0441\u0442\u0440\u043e\u043a")):
        return "table_rows"
    markers = (
        "card",
        "cards",
        "catalog",
        "list",
        "items",
        "several",
        "multiple",
        "all ",
        "top ",
        "stories",
        "story cards",
        "articles",
        "modules",
        "products",
        "listings",
        "\u043a\u0430\u0440\u0442\u043e\u0447",
        "\u0441\u043f\u0438\u0441\u043e\u043a",
        "\u044d\u043b\u0435\u043c\u0435\u043d\u0442",
        "\u043f\u0443\u043d\u043a\u0442",
        "\u043d\u0435\u0441\u043a\u043e\u043b\u044c\u043a",
        "\u0432\u0441\u0435 ",
        "\u0438\u0441\u0442\u043e\u0440",
        "\u0441\u0442\u0430\u0442",
        "\u043c\u043e\u0434\u0443\u043b",
        "\u0442\u043e\u0432\u0430\u0440",
    )
    if any(marker in combined for marker in markers):
        return "card_items"
    return ""


def collection_condition_for_goal(goal: object) -> dict[str, Any]:
    text_variants = [item.strip() for item in _text_variants_with_mojibake_repairs(goal) if item.strip()]
    if not text_variants:
        return {}
    patterns = [
        ("title", r"\b(?:whose|where|with)\s+(?:title|name|heading)\s+(?:contains?|includes?|has)\s+(.{1,80}?)(?:,|\.|:|\bwith\b|\bthen\b|$)"),
        ("title", r"\b(?:title|name|heading)\s+(?:contains?|includes?|has)\s+(.{1,80}?)(?:,|\.|:|\bwith\b|\bthen\b|$)"),
        (
            "title",
            "(?:\u0432\\s+(?:\u043d\u0430\u0437\u0432\u0430\u043d\u0438\u0438|\u0437\u0430\u0433\u043e\u043b\u043e\u0432\u043a\u0435)[^,.:]{0,70}?"
            "\\s+(?:\u0435\u0441\u0442\u044c|\u0441\u043e\u0434\u0435\u0440\u0436\\w*)\\s+(.{1,80}?)(?:,|\\.|:|\\s+\u0441\\s+|$))",
        ),
        (
            "title",
            "(?:\u043d\u0430\u0437\u0432\u0430\u043d\u0438[е\u044f]|\u0437\u0430\u0433\u043e\u043b\u043e\u0432\\w*)\\s+"
            "(?:\u0441\u043e\u0434\u0435\u0440\u0436\\w*|\u0432\u043a\u043b\u044e\u0447\u0430\\w*|\u0438\u043c\u0435\\w*)\\s+(.{1,80}?)(?:,|\\.|:|\\s+\u0441\\s+|$)",
        ),
        ("contains", r"\b(?:contains?|includes?|matching)\s+(.{1,80}?)(?:,|\.|:|\bwith\b|\bthen\b|$)"),
        (
            "contains",
            "(?:\u0441\u043e\u0434\u0435\u0440\u0436\\w*|\u0432\u043a\u043b\u044e\u0447\u0430\\w*)\\s+(.{1,80}?)(?:,|\\.|:|\\s+\u0441\\s+|$)",
        ),
    ]
    for text in text_variants:
        for key, pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.\"'")
            if not raw:
                continue
            terms = [
                item.strip(" '\"\u201c\u201d\u00ab\u00bb")
                for item in re.split(r"\s+(?:or|или)\s+|\|", raw, flags=re.IGNORECASE)
                if item.strip(" '\"\u201c\u201d\u00ab\u00bb")
            ]
            if terms:
                return {key: terms if len(terms) > 1 else terms[0]}
    return {}


def collection_item_fields_for_goal(goal: object, required_fields: list[str] | None = None) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for raw_field in required_fields or []:
        field = str(raw_field or "").strip()
        if not field:
            continue
        normalized = _casefold_with_mojibake_repairs(normalize_required_field_alias(field))
        if normalized in TECHNICAL_PLAN_FIELDS or normalized in _COLLECTION_PARENT_HINTS:
            continue
        if any(token in normalized for token in ("title", "name", "heading", "\u043d\u0430\u0437\u0432\u0430\u043d", "\u0437\u0430\u0433\u043e\u043b\u043e\u0432")):
            fields[field] = {"type": "title"}
        elif any(token in normalized for token in ("description", "summary", "snippet", "overview", "brief", "\u043e\u043f\u0438\u0441\u0430\u043d", "\u043a\u0440\u0430\u0442\u043a")):
            fields[field] = {"type": "description"}
        elif any(token in normalized for token in ("href", "link", "url", "\u0441\u0441\u044b\u043b\u043a")):
            fields[field] = {"type": "url"}
        elif any(token in normalized for token in ("count", "number", "total", "\u0447\u0438\u0441\u043b", "\u043a\u043e\u043b\u0438\u0447")):
            fields[field] = {"type": "number"}
        else:
            fields[field] = {"type": "text"}
    if fields:
        return fields

    text = _casefold_with_mojibake_repairs(goal)
    if any(token in text for token in ("title", "name", "heading", "\u043d\u0430\u0437\u0432\u0430\u043d", "\u0437\u0430\u0433\u043e\u043b\u043e\u0432")):
        fields["title"] = {"type": "title"}
    if any(token in text for token in ("description", "summary", "snippet", "brief", "\u043e\u043f\u0438\u0441\u0430\u043d", "\u043a\u0440\u0430\u0442\u043a")):
        fields["description"] = {"type": "description"}
    return fields


def collection_output_key_for_goal(
    *,
    goal: object,
    required_fields: list[str] | None,
    intent: str,
) -> str:
    normalized_intent = normalize_intent_alias(intent)
    for field in required_fields or []:
        candidate = normalize_required_field_alias(str(field or "").strip())
        if candidate and candidate.casefold() in _COLLECTION_PARENT_HINTS:
            return candidate
    text = str(goal or "").casefold()
    if "stories" in text or "\u0438\u0441\u0442\u043e\u0440" in text:
        return "items"
    return default_output_key_for_intent(normalized_intent)


def repair_collection_plan_steps(
    steps: list[dict[str, Any]],
    *,
    goal: object,
    required_fields: list[str],
    preferred_intents: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if anchor_object_args_for_goal(goal, required_fields):
        return steps, required_fields
    intent = collection_intent_for_goal(goal, required_fields)
    if not intent or intent == "visible_links":
        return steps, required_fields
    if any(
        isinstance(step, dict) and str(step.get("action", "") or "").strip() in {"find_row_by_condition", "click_row_action"}
        for step in steps
    ):
        return steps, required_fields
    normalized_preferred = {normalize_intent_alias(item) for item in (preferred_intents or set()) if str(item).strip()}
    if normalized_preferred and intent not in normalized_preferred:
        return steps, required_fields

    item_fields = collection_item_fields_for_goal(goal, required_fields)
    condition = collection_condition_for_goal(goal)
    business_required = [
        normalize_required_field_alias(str(field or "").strip())
        for field in required_fields
        if str(field or "").strip()
    ]
    business_required = [
        field for field in business_required if field and field.casefold() not in TECHNICAL_PLAN_FIELDS
    ]
    item_field_names = {name.casefold() for name in item_fields}

    if intent == "table_rows":
        for step in steps:
            if not isinstance(step, dict) or str(step.get("action", "") or "").strip() != "extract_by_intent":
                continue
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            if normalize_intent_alias(args.get("intent", "")) != "field_schema":
                continue
            fields = args.get("fields")
            save_as = str(step.get("save_as", "") or args.get("output_key", "") or "").strip()
            if not save_as or not isinstance(fields, dict):
                continue
            field_names = {
                normalize_required_field_alias(str(field or "").strip()).casefold()
                for field in fields.keys()
                if str(field or "").strip()
            }
            if business_required and all(field.casefold() in field_names for field in business_required):
                return steps, [save_as]

    def is_collection_step(step: dict[str, Any]) -> bool:
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        step_intent = normalize_intent_alias(args.get("intent", ""))
        return action in {"extract_items", "extract_structured_items"} or (
            action == "extract_by_intent" and step_intent in {"card_items", "table_rows", "cards", "rows"}
        )

    existing_output_key = ""
    for step in steps:
        if not isinstance(step, dict) or not is_collection_step(step):
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        existing_output_key = str(args.get("output_key", "") or step.get("save_as", "") or "").strip()
        if existing_output_key:
            break
    output_key = existing_output_key or collection_output_key_for_goal(
        goal=goal,
        required_fields=required_fields,
        intent=intent,
    )

    def is_scalar_item_projection(step: dict[str, Any]) -> bool:
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        save_as = normalize_required_field_alias(str(step.get("save_as", "") or args.get("output_key", "") or "").strip())
        if not save_as:
            return False
        if save_as.casefold() not in item_field_names and save_as not in business_required:
            return False
        if action in {"extract_text", "extract_html", "extract_pattern_from_page_text", "extract_text_near_text", "extract_value_near_anchor"}:
            return True
        if action == "extract_by_intent":
            step_intent = normalize_intent_alias(args.get("intent", ""))
            return step_intent in {"text_block", "field_schema", "anchor_object", "value_near_anchor"}
        return False

    repaired: list[dict[str, Any]] = []
    removed_broad_click = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        if action == "click_by_semantic_target" and is_broad_semantic_click_target(
            args.get("target_text") or args.get("target") or args.get("text")
        ):
            removed_broad_click = True
            continue
        if is_scalar_item_projection(step):
            continue
        current = dict(step)
        current["args"] = dict(args)
        if is_collection_step(current):
            current_args = current["args"]
            current_args["intent"] = intent
            current_args.setdefault("output_key", output_key)
            if item_fields:
                current_args["fields"] = item_fields
            if condition:
                current_args["condition"] = condition
                current_args.pop("filter", None)
                current_args.pop("where", None)
            if not str(current.get("save_as", "") or "").strip():
                current["save_as"] = str(current_args.get("output_key") or output_key)
        repaired.append(current)

    has_collection = any(is_collection_step(step) for step in repaired if isinstance(step, dict))
    if not has_collection:
        insert_index = next((idx for idx, step in enumerate(repaired) if step.get("action") == "finish"), len(repaired))
        args: dict[str, Any] = {"intent": intent, "output_key": output_key, "limit": 20}
        if item_fields:
            args["fields"] = item_fields
        if condition:
            args["condition"] = condition
        repaired.insert(
            insert_index,
            {
                "step_id": 0,
                "action": "extract_by_intent",
                "args": args,
                "save_as": output_key,
            },
        )

    updated_required = list(required_fields)
    if business_required and output_key not in updated_required:
        if item_fields and (
            removed_broad_click
            or all(field.casefold() in item_field_names for field in business_required)
            or not any(field.casefold() in _COLLECTION_PARENT_HINTS for field in business_required)
        ):
            updated_required = [output_key]
    return repaired, updated_required


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
