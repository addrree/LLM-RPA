from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import re
from difflib import SequenceMatcher
from typing import Any


@dataclass
class MiniWoBGroundingResult:
    action: str
    mapping_error: str | None = None
    selected_candidate: dict[str, Any] | None = None
    repeated_warning: str | None = None
    mapping_strategy: str | None = None
    mapping_diagnostics: dict[str, Any] | None = None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def normalize_candidate_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("value", "name", "text", "label"):
            if key in value and value.get(key) not in (None, ""):
                return normalize_candidate_value(value.get(key))
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return normalize_candidate_value(parsed)
        return re.sub(r"\s+", " ", stripped)
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_candidate_text(value: Any) -> str:
    return normalize_text(normalize_candidate_value(value))


def _candidate_normalized_text(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("value", "name", "text", "label"):
        if key in candidate and candidate.get(key) not in (None, ""):
            text = normalize_candidate_value(candidate.get(key))
            if text:
                return text
    return ""


REAL_BID_KEYS = ("bid", "data-testid", "data_testid", "browsergym_id", "data-bid", "data_bid", "ref")
REAL_BID_SOURCES = {"bid", "data-testid", "data_testid", "browsergym_id", "data-bid", "data_bid", "ref"}
FAKE_BID_SOURCES = {"id", "dom_id", "element_id", "index", "candidate_index", "node_id", "backend_node_id"}
SUBMIT_BUTTON_NAMES = {"submit", "login", "ok", "done"}
SUBMIT_BUTTON_ALIASES = SUBMIT_BUTTON_NAMES | {"go"}
TEXT_INPUT_INTENTS = {"fill", "type", "enter", "input", "text", "username", "password"}
SELECT_INTENT_WORDS = {"choose", "select", "pick"}
SELECT_CONTAINER_WORDS = {"list", "dropdown", "drop-down", "combo", "combobox", "select", "option", "menu"}


def real_candidate_bid(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    source = str(candidate.get("bid_source") or "").strip()
    if source in FAKE_BID_SOURCES:
        return ""
    if source and source not in REAL_BID_SOURCES:
        return ""
    for key in REAL_BID_KEYS:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            if key == "bid" and not source:
                return str(value).strip()
            if source in REAL_BID_SOURCES or key != "bid":
                return str(value).strip()
    return ""


def _candidate_id(candidate: dict[str, Any]) -> str:
    return real_candidate_bid(candidate)


def _candidate_text_values(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("name", "text", "value", "selected_value", "current_value", "label", "visible_label", "ariaLabel", "aria_label", "aria-label", "title", "innerText"):
        value = candidate.get(key)
        if isinstance(value, dict):
            value = value.get("value") or value.get("name")
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    return values


def _candidate_text(candidate: dict[str, Any]) -> str:
    values = _candidate_text_values(candidate)
    return values[0] if values else ""


def _is_textbox(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    role = _norm(candidate.get("role"))
    tag = _norm(candidate.get("tag"))
    typ = _norm(candidate.get("type"))
    return role == "textbox" or tag == "textarea" or (tag == "input" and typ not in {"button", "submit", "checkbox", "radio"})


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def browsergym_click_action(candidate_id: str, action_syntax: list[str] | None = None) -> str:
    return f'click("{_escape(candidate_id)}", "left")'


def browsergym_fill_action(candidate_id: str, text: str) -> str:
    return f'fill("{_escape(candidate_id)}", "{_escape(text)}")'


def _action_syntax_prefers_option_list(action_syntax: list[str] | None) -> bool:
    for item in action_syntax or []:
        text = str(item).casefold()
        if "select_option" in text and ("[" in text or "list" in text or "options" in text):
            return True
    return False


def browsergym_select_option_action(candidate_id: str, option_text: str, action_syntax: list[str] | None = None) -> str:
    if _action_syntax_prefers_option_list(action_syntax):
        return f'select_option("{_escape(candidate_id)}", ["{_escape(option_text)}"])'
    return f'select_option("{_escape(candidate_id)}", "{_escape(option_text)}")'


def _parse_call(action: str) -> tuple[str, list[Any]] | None:
    text = str(action or "").strip()
    match = re.match(r"^\s*([A-Za-z_][\w]*)\s*\((.*)\)\s*$", text, flags=re.DOTALL)
    if not match:
        return None
    name = match.group(1).lower()
    args_raw = match.group(2).strip()
    if not args_raw:
        return name, []
    try:
        parsed = json.loads(f"[{args_raw}]")
        if isinstance(parsed, list):
            return name, parsed
    except Exception:
        pass
    # Minimal fallback for bare click(submit)-style calls.
    parts = [part.strip() for part in args_raw.split(",")]
    cleaned: list[Any] = []
    for part in parts:
        if (part.startswith('"') and part.endswith('"')) or (part.startswith("'") and part.endswith("'")):
            cleaned.append(part[1:-1])
        else:
            cleaned.append(part)
    return name, cleaned


def _extract_click_target(action: str) -> str:
    parsed = _parse_call(action)
    if not parsed or parsed[0] != "click" or not parsed[1]:
        return ""
    return str(parsed[1][0]).strip()


def _visible_enabled_bonus(candidate: dict[str, Any]) -> int:
    enabled_bonus = 12 if candidate.get("enabled", True) is not False and candidate.get("disabled") is not True else -30
    visible_bonus = 12 if candidate.get("visible", True) is not False else -30
    clickable_bonus = 4 if candidate.get("clickable", True) is not False else -8
    return enabled_bonus + visible_bonus + clickable_bonus


def _score_candidate(candidate: dict[str, Any], target: str) -> tuple[int, float, int]:
    target_n = _norm(target)
    if not target_n:
        return (0, 0.0, 0)
    role = _norm(candidate.get("role"))
    tag = _norm(candidate.get("tag"))
    buttonish_bonus = 25 if role == "button" or tag in {"button", "input"} else 0
    best_base = 0
    best_ratio = 0.0
    best_len = 9999
    for text_value in _candidate_text_values(candidate):
        text = _norm(text_value)
        if not text:
            continue
        ratio = SequenceMatcher(None, text, target_n).ratio()
        if text == target_n:
            base = 120
        elif target_n in text or text in target_n:
            base = 80
        elif ratio >= 0.72:
            base = 45
        else:
            base = 0
        if base > best_base or (base == best_base and ratio > best_ratio):
            best_base = base
            best_ratio = ratio
            best_len = len(text)
    return (best_base + buttonish_bonus + _visible_enabled_bonus(candidate), best_ratio, -best_len)


def find_click_candidate(candidates: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    scored: list[tuple[tuple[int, float, int], int, dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict):
            continue
        score = _score_candidate(candidate, target)
        if score[0] > 0:
            scored.append((score, -idx, candidate))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    best_score, _, best = scored[0]
    return best if best_score[0] >= 45 else None


def textbox_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in candidates or [] if _is_textbox(c)]



def _role_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or value.get("name") or "").strip()
    return str(value or "").strip()


def _attrs_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) % 2 == 0 and all(not isinstance(item, (dict, list, tuple)) for item in value):
            return {str(value[i]): value[i + 1] for i in range(0, len(value), 2)}
        out: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("key") or item.get("attribute")
                if name is not None:
                    out[str(name)] = item.get("value")
        return out
    return {}


def _first_scalar(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _copy_real_bid(candidate: dict[str, Any], source: dict[str, Any], attrs: dict[str, Any] | None = None) -> None:
    attrs = attrs or {}
    for key in REAL_BID_KEYS:
        value = source.get(key)
        if value in (None, ""):
            value = attrs.get(key)
        if value not in (None, ""):
            candidate["bid"] = str(value).strip()
            candidate["bid_source"] = key
            candidate[key] = str(value).strip()
            return


def _looks_editable(node: dict[str, Any], attrs: dict[str, Any]) -> bool:
    role = _norm(_role_value(node.get("role")))
    node_name = _norm(node.get("nodeName") or node.get("tagName") or node.get("tag"))
    typ = _norm(node.get("type") or attrs.get("type"))
    if role == "textbox" or node_name == "textarea":
        return True
    if node_name == "input":
        return typ in {"", "text", "password", "search", "email", "tel", "url"}
    for key in ("editable", "focusable", "isEditable", "isTextInput", "input"):
        value = node.get(key)
        if value is True or _norm(value) in {"true", "1", "yes"}:
            return True
    return False


def _textbox_candidate_from_axtree_node(node: dict[str, Any]) -> dict[str, Any] | None:
    attrs = _attrs_dict(node.get("attributes"))
    if not _looks_editable(node, attrs):
        return None
    candidate: dict[str, Any] = {"role": "textbox", "source": "axtree_object"}
    _copy_real_bid(candidate, node, attrs)
    for out_key, *in_keys in (
        ("name", "name", "accessibleName", "label"),
        ("value", "value", "inputValue"),
        ("backendDOMNodeId", "backendDOMNodeId", "backend_node_id"),
        ("tag", "tag", "tagName", "nodeName"),
        ("type", "type"),
    ):
        value = _first_scalar(node, *in_keys)
        if value not in (None, ""):
            candidate[out_key] = _role_value(value) if out_key in {"name", "value"} else str(value).strip()
    for key in ("bbox", "bounding_box", "browsergym_bbox", "action_bbox"):
        if node.get(key) not in (None, ""):
            candidate[key] = node.get(key)
    for key in ("editable", "focusable", "enabled", "visible", "disabled"):
        if key in node:
            candidate[key] = node.get(key)
    return candidate if real_candidate_bid(candidate) or candidate.get("backendDOMNodeId") else None


def _textbox_candidate_from_dom_node(node: dict[str, Any]) -> dict[str, Any] | None:
    attrs = _attrs_dict(node.get("attributes"))
    node_name = _norm(node.get("nodeName") or node.get("tagName") or node.get("tag"))
    typ = _norm(node.get("type") or attrs.get("type"))
    if node_name not in {"input", "textarea"}:
        return None
    if node_name == "input" and typ not in {"", "text", "password", "search", "email", "tel", "url"}:
        return None
    candidate: dict[str, Any] = {"role": "textbox", "tag": node_name, "source": "dom_object", "input_type": typ or "text"}
    _copy_real_bid(candidate, node, attrs)
    for out_key, *in_keys in (
        ("name", "name", "aria-label", "ariaLabel", "placeholder"),
        ("value", "value", "inputValue"),
        ("backendDOMNodeId", "backendDOMNodeId", "backend_node_id"),
        ("type", "type"),
    ):
        value = _first_scalar(node, *in_keys)
        if value in (None, ""):
            value = _first_scalar(attrs, *in_keys)
        if value not in (None, ""):
            candidate[out_key] = str(value).strip()
    for key in ("bbox", "bounding_box", "browsergym_bbox", "action_bbox"):
        if node.get(key) not in (None, ""):
            candidate[key] = node.get(key)
    return candidate if real_candidate_bid(candidate) or candidate.get("backendDOMNodeId") else None


def _walk_textbox_nodes(value: Any, *, source: str, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        candidate = _textbox_candidate_from_axtree_node(value) if source == "axtree_object" else _textbox_candidate_from_dom_node(value)
        if candidate:
            out.append(candidate)
        for child in value.values():
            if isinstance(child, (dict, list, tuple)):
                _walk_textbox_nodes(child, source=source, out=out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_textbox_nodes(item, source=source, out=out)


def merge_textbox_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        bid = real_candidate_bid(candidate)
        backend_id = str(candidate.get("backendDOMNodeId") or candidate.get("backend_node_id") or "").strip()
        keys = []
        if bid:
            keys.append(("bid", bid))
        if backend_id:
            keys.append(("backendDOMNodeId", backend_id))
        existing = next((index[key] for key in keys if key in index), None)
        if existing is None:
            existing = dict(candidate)
            merged.append(existing)
        else:
            for key, value in candidate.items():
                if value not in (None, "") and existing.get(key) in (None, ""):
                    existing[key] = value
            if existing.get("source") != candidate.get("source") and candidate.get("source"):
                existing["source"] = "+".join(sorted(set(str(existing.get("source") or "").split("+")) | {str(candidate.get("source"))}))
        for key in keys:
            index[key] = existing
    return merged


def extract_textbox_candidates_from_observation(obs: dict[str, Any] | None, info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for source_map in (obs if isinstance(obs, dict) else {}, info if isinstance(info, dict) else {}):
        if source_map.get("axtree_object") is not None:
            _walk_textbox_nodes(source_map.get("axtree_object"), source="axtree_object", out=found)
        if source_map.get("dom_object") is not None:
            _walk_textbox_nodes(source_map.get("dom_object"), source="dom_object", out=found)
    return merge_textbox_candidates(found)


def parse_quoted_strings(instruction: str) -> list[str]:
    return [m.group(1) or m.group(2) for m in re.finditer(r'"([^"]*)"|\'([^\']*)\'', str(instruction or ""))]


def parse_username_password_instruction(instruction: str) -> dict[str, str]:
    text = str(instruction or "")
    result: dict[str, str] = {}
    user_match = re.search(r"username\s+['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
    pass_match = re.search(r"password\s+['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
    if user_match:
        result["username"] = user_match.group(1)
    if pass_match:
        result["password"] = pass_match.group(1)
    return result


def map_login_textboxes(instruction: str, candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values = parse_username_password_instruction(instruction)
    boxes = textbox_candidates(candidates)
    mapped: dict[str, dict[str, Any]] = {}
    if "username" in values and len(boxes) >= 1:
        mapped["username"] = boxes[0]
    if "password" in values and len(boxes) >= 2:
        mapped["password"] = boxes[1]
    return mapped



def _text_from_instruction_for_fill(instruction: str, selected: dict[str, Any] | None = None) -> str:
    values = parse_username_password_instruction(instruction)
    if values:
        input_type = _norm((selected or {}).get("input_type") or (selected or {}).get("type"))
        name_text = _norm(" ".join(_candidate_text_values(selected or {})))
        if input_type == "password" or "password" in name_text:
            return values.get("password", "")
        return values.get("username", "")
    quoted = parse_quoted_strings(instruction)
    if quoted:
        return quoted[0]
    return ""


def _instruction_requires_text_entry(instruction: str) -> bool:
    text = _norm(instruction)
    return bool(parse_quoted_strings(instruction)) and any(word in text for word in ("enter", "type", "input", "text field", "password", "username"))



def _is_select_control(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    role = _norm(_role_value(candidate.get("role")) or candidate.get("kind"))
    tag = _norm(candidate.get("tag"))
    return role in {"combobox", "listbox", "select"} or tag == "select"


def _is_option_candidate(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    role = _norm(_role_value(candidate.get("role")) or candidate.get("kind"))
    tag = _norm(candidate.get("tag"))
    return role in {"option", "listitem", "menuitem", "radio"} or tag in {"option", "li"}


def _instruction_has_select_intent(instruction: str) -> bool:
    text = _norm(instruction)
    if not text:
        return False
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in SELECT_INTENT_WORDS) and any(word in text for word in SELECT_CONTAINER_WORDS)


def extract_select_target_from_instruction(instruction: str, candidates: list[dict[str, Any]] | None = None) -> str:
    text = str(instruction or "")
    quoted = parse_quoted_strings(text)
    if quoted:
        return quoted[0].strip()
    patterns = (
        r"\b(?:choose|select|pick)\s+(?:the\s+)?(?:option|value|item)?\s*([^.,;:]+?)\s+(?:from|in|on)\s+(?:the\s+)?(?:list|dropdown|drop-down|combobox|combo|select|menu)\b",
        r"\b(?:choose|select|pick)\s+(?:the\s+)?(?:option|value|item)\s+([^.,;:]+)",
        r"\b(?:from|in)\s+(?:the\s+)?(?:list|dropdown|drop-down|combobox|combo|select|menu)[^.,;:]*?\b(?:choose|select|pick)\s+([^.,;:]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\b(?:option|value|item)\b", " ", match.group(1), flags=re.IGNORECASE)
            value = re.sub(r"\s+", " ", value).strip(" .,:;\t")
            if value:
                return value
    option_candidates = [c for c in candidates or [] if _is_option_candidate(c)]
    instruction_n = _norm(text)
    matches: list[tuple[int, int, str]] = []
    for idx, candidate in enumerate(option_candidates):
        for option_text in _candidate_text_values(candidate):
            norm = _norm(option_text)
            if norm and re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", instruction_n):
                matches.append((len(norm), -idx, option_text.strip()))
    if matches:
        matches.sort(reverse=True)
        return matches[0][2]
    return ""


def _find_select_control(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    controls = [c for c in candidates or [] if _is_select_control(c) and c.get("disabled") is not True and c.get("enabled", True) is not False]
    return controls[0] if controls else None


def _candidate_parent_refs(candidate: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("parent_bid", "owner_bid", "controlled_by", "controls", "listbox_bid", "select_bid"):
        value = candidate.get(key)
        if value not in (None, ""):
            refs.add(str(value).strip())
    return refs


def _find_option_candidate(candidates: list[dict[str, Any]], target_text: str, control: dict[str, Any] | None = None) -> dict[str, Any] | None:
    control_bid = real_candidate_bid(control) if control else ""
    scored: list[tuple[tuple[int, float, int], int, dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates or []):
        if not _is_option_candidate(candidate):
            continue
        if candidate.get("disabled") is True or candidate.get("enabled", True) is False:
            continue
        score = _score_candidate(candidate, target_text)
        text_match_score = score[0] - _visible_enabled_bonus(candidate)
        if text_match_score <= 0:
            continue
        if control_bid and control_bid in _candidate_parent_refs(candidate):
            score = (score[0] + 20, score[1], score[2])
        scored.append((score, -idx, candidate))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    best_score, _, best = scored[0]
    return best if best_score[0] >= 45 else None


def _select_option_supported(action_syntax: list[str] | None) -> bool:
    return any("select_option" in str(item) for item in action_syntax or [])


def _select_control_current_value(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("value", "selected_value", "current_value", "selected", "name", "text", "label"):
        value = candidate.get(key)
        if value not in (None, "") and not isinstance(value, bool):
            return str(value).strip()
    return ""


def _select_control_has_target_value(candidate: dict[str, Any] | None, target_text: str) -> bool:
    if not isinstance(candidate, dict) or not target_text:
        return False
    target_n = normalize_text(target_text)
    for key in ("value", "selected_value", "current_value", "name", "text", "label"):
        value = candidate.get(key)
        if value not in (None, "") and not isinstance(value, bool) and normalize_text(normalize_candidate_value(value)) == target_n:
            return True
    return False


def _selected_target_option(candidates: list[dict[str, Any]], target_text: str) -> dict[str, Any] | None:
    for candidate in candidates or []:
        if _is_option_candidate(candidate) and candidate.get("selected") is True and _candidate_matches_text(candidate, target_text):
            return candidate
    control = _find_select_control(candidates)
    return control if _select_control_has_target_value(control, target_text) else None


def _select_option_no_progress_error(mapped_action: str, control: dict[str, Any] | None, target_text: str, history: list[dict]) -> str | None:
    if not mapped_action or not _is_select_control(control) or not target_text:
        return None
    if _select_control_has_target_value(control, target_text):
        return None
    attempts = 0
    saw_attempt = False
    for item in reversed(history or []):
        previous = str(item.get("action") or "").strip()
        if previous == mapped_action and float(item.get("reward") or 0) <= 0:
            saw_attempt = True
            attempts += 1
            continue
        if saw_attempt:
            break
        if previous and not previous.startswith("noop"):
            break
    if attempts >= 2:
        return "action_mapping_failure: select_option_no_progress"
    return None


def is_submit_like_candidate(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    role = normalize_candidate_text(candidate.get("role"))
    kind = normalize_candidate_text(candidate.get("kind"))
    tag = normalize_candidate_text(candidate.get("tag"))
    typ = normalize_candidate_text(candidate.get("type"))
    if role != "button" and kind != "button" and tag != "button" and typ not in {"button", "submit"}:
        return False
    for key in ("name", "text", "value", "label", "visible_label", "ariaLabel", "aria_label", "aria-label", "title", "innerText"):
        if normalize_candidate_text(candidate.get(key)) in SUBMIT_BUTTON_ALIASES:
            return True
    return False


def _is_submit_like(candidate: dict[str, Any] | None) -> bool:
    return is_submit_like_candidate(candidate)


def _is_explicit_left_click(parsed: tuple[str, list[Any]] | None) -> bool:
    if not parsed or parsed[0] != "click" or not parsed[1]:
        return False
    if len(parsed[1]) == 1:
        return True
    return _norm(parsed[1][1]) in {"", "left"}


def _looks_like_bid_literal(value: str, parsed_response: dict[str, Any] | None = None, target_text: str = "") -> bool:
    value = str(value or "").strip()
    if not value:
        return False
    target_bid = str((parsed_response or {}).get("target_bid") or "").strip()
    if target_bid and value == target_bid:
        return True
    if target_text and _norm(value) == normalize_text(target_text):
        return False
    return bool(re.search(r"\d", value) or (not re.search(r"\s", value) and re.search(r"[_.:-]", value)))


def _candidate_matches_text(candidate: dict[str, Any] | None, target_text: str) -> bool:
    if not isinstance(candidate, dict) or not target_text:
        return False
    target_n = normalize_text(target_text)
    for text_value in _candidate_text_values(candidate):
        text_n = _norm(text_value)
        if text_n and (text_n == target_n or target_n in text_n or text_n in target_n):
            return True
    return False


def _candidate_text_summary(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    return {
        "bid": real_candidate_bid(candidate) or candidate.get("bid"),
        "bid_source": candidate.get("bid_source"),
        "text": candidate.get("text"),
        "name": candidate.get("name"),
        "value": candidate.get("value"),
        "label": candidate.get("label") or candidate.get("visible_label") or candidate.get("ariaLabel") or candidate.get("aria_label"),
        "role": candidate.get("role"),
        "kind": candidate.get("kind"),
        "tag": candidate.get("tag"),
        "selected": candidate.get("selected"),
        "visible": candidate.get("visible"),
    }


def _option_candidate_texts(candidates: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for candidate in candidates or []:
        if _is_option_candidate(candidate):
            text = _candidate_text(candidate)
            if text:
                texts.append(text)
    return texts[:30]


def _select_diagnostics(
    *,
    target_text: str,
    candidates: list[dict[str, Any]],
    clicked_bid: str = "",
    clicked_candidate: dict[str, Any] | None = None,
    decision: str = "",
) -> dict[str, Any]:
    return {
        "target_option": target_text or "",
        "option_candidates_count": sum(1 for candidate in candidates or [] if _is_option_candidate(candidate)),
        "option_candidate_texts": _option_candidate_texts(candidates),
        "clicked_bid": clicked_bid or "",
        "clicked_bid_candidate": _candidate_text_summary(clicked_candidate),
        "select_guard_decision": decision or "",
    }



def _target_option_visible_enough(option: dict[str, Any] | None) -> bool:
    return isinstance(option, dict) and option.get("visible", True) is not False


def _same_select_control_repeat_error(mapped_action: str, candidate: dict[str, Any] | None, history: list[dict]) -> str | None:
    if not _is_select_control(candidate):
        return None
    repeats = 0
    for item in reversed(history or []):
        if str(item.get("action") or "").strip() == mapped_action and float(item.get("reward") or 0) <= 0:
            repeats += 1
        else:
            break
    if repeats >= 2:
        return "action_mapping_failure: no_progress_repeated_select repeated combobox/select click without progress"
    return None


def _rationale_links_bid_to_target(parsed_response: dict[str, Any], clicked_bid: str, target_text: str) -> bool:
    haystack = _norm(" ".join(str(parsed_response.get(key) or "") for key in ("rationale", "reason", "target_text", "option_text", "option_value")))
    return bool(clicked_bid and target_text and _norm(clicked_bid) in haystack and normalize_text(target_text) in haystack)


def _history_action_target(action: str) -> str:
    parsed = _parse_call(action)
    if parsed and parsed[0] in {"click", "select_option"} and parsed[1]:
        return str(parsed[1][0]).strip()
    return ""


def _should_block_repeated_select_loop(mapped_action: str, candidates: list[dict[str, Any]], history: list[dict]) -> str | None:
    current_target = _history_action_target(mapped_action)
    if not current_target:
        return None
    current_candidate = _find_by_real_bid(candidates, current_target)
    if not (_is_select_control(current_candidate) or _is_option_candidate(current_candidate)):
        return None
    targets = [current_target]
    for item in reversed(history):
        if float(item.get("reward") or 0) > 0:
            break
        target = _history_action_target(str(item.get("action") or ""))
        if not target:
            break
        candidate = _find_by_real_bid(candidates, target)
        if not (_is_select_control(candidate) or _is_option_candidate(candidate)):
            break
        targets.append(target)
    unique = set(targets)
    if len(targets) >= 5 and len(unique) == 2:
        kinds = {_norm((_find_by_real_bid(candidates, target) or {}).get("role") or (_find_by_real_bid(candidates, target) or {}).get("tag")) for target in unique}
        if any(kind in {"combobox", "listbox", "select"} for kind in kinds) and "option" in kinds:
            return "action_mapping_failure: no_progress_repeated_select combobox-option pair repeated without reward/progress"
    return None


def _ground_select_intent(
    *,
    proposed_action: str,
    instruction: str,
    parsed_response: dict[str, Any],
    candidates: list[dict[str, Any]],
    history: list[dict],
    action_syntax: list[str] | None,
) -> MiniWoBGroundingResult | None:
    explicit_intent = _norm(parsed_response.get("intent") or parsed_response.get("action_intent") or parsed_response.get("rationale") or "")
    target_text = str(parsed_response.get("option_text") or parsed_response.get("option_value") or "").strip()
    if not target_text:
        target_text = extract_select_target_from_instruction(instruction, candidates)
    if not target_text:
        proposed = str(parsed_response.get("target_text") or parsed_response.get("text") or parsed_response.get("value") or "").strip()
        if proposed and _find_option_candidate(candidates, proposed):
            target_text = proposed
    has_select_intent = _instruction_has_select_intent(instruction) or any(word in explicit_intent for word in ("select", "choose", "pick", "combobox", "dropdown", "list"))
    instruction_n = normalize_text(instruction)
    requires_submit_click = any(token in instruction_n for token in ("submit", "login", "done", "click submit", "press submit"))
    if not has_select_intent:
        return None

    parsed_proposed = _parse_call(proposed_action)
    clicked_raw = str(parsed_proposed[1][0]).strip() if _is_explicit_left_click(parsed_proposed) else ""
    clicked_raw_candidate = _find_by_real_bid(candidates, clicked_raw) if clicked_raw else None
    clicked_bid = clicked_raw if clicked_raw_candidate is not None or _looks_like_bid_literal(clicked_raw, parsed_response, target_text) else ""
    clicked_candidate = clicked_raw_candidate if clicked_bid else None
    control = _find_select_control(candidates)
    control_bid = real_candidate_bid(control) if control else ""
    option = _find_option_candidate(candidates, target_text, control) if target_text else None
    option_text = _candidate_text(option) if option else target_text
    submit_candidate = find_submit_button(candidates)
    current_value = _select_control_current_value(control)
    normalized_current_value = normalize_candidate_value(current_value)
    normalized_target_text = normalize_text(target_text)
    selected_value_matches_target = normalize_text(normalized_current_value) == normalized_target_text if normalized_current_value and normalized_target_text else False
    selected = _selected_target_option(candidates, target_text) if selected_value_matches_target else None
    diagnostics = _select_diagnostics(
        target_text=target_text,
        candidates=candidates,
        clicked_bid=clicked_bid,
        clicked_candidate=clicked_candidate,
        decision="evaluating_select_guard",
    )
    diagnostics.update(
        {
            "select_control_bid": control_bid,
            "current_select_value_before": current_value,
            "normalized_current_select_value": normalized_current_value,
            "selected_value_matches_target": bool(selected_value_matches_target),
            "submit_candidate": _candidate_text_summary(submit_candidate),
        }
    )

    if clicked_bid and clicked_candidate is None:
        diagnostics["select_guard_decision"] = "blocked_unknown_bid"
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_error=f"action_mapping_failure: clicked bid {clicked_bid!r} not found in candidates",
            mapping_strategy="none",
            mapping_diagnostics=diagnostics,
        )

    if not target_text:
        diagnostics["select_guard_decision"] = "blocked_missing_target_option"
        return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: missing_target_option", mapping_strategy="none", mapping_diagnostics=diagnostics)

    proposed_target = str(parsed_proposed[1][0]).strip() if parsed_proposed and parsed_proposed[1] else str(parsed_response.get("target_text") or "").strip()
    proposed_candidate = (_find_by_real_bid(candidates, proposed_target) or find_click_candidate(candidates, proposed_target)) if proposed_target else None
    is_submit_attempt = bool(clicked_candidate is not None and _is_submit_like(clicked_candidate)) or bool(find_submit_button([proposed_candidate] if isinstance(proposed_candidate, dict) else []))
    if is_submit_attempt and requires_submit_click:
        diagnostics["clicked_bid_candidate_text"] = _candidate_normalized_text(clicked_candidate)
        diagnostics["clicked_bid_candidate_role"] = normalize_candidate_value((clicked_candidate or {}).get("role"))
        if selected_value_matches_target and clicked_bid:
            diagnostics["submit_allowed"] = True
            diagnostics["submit_source"] = "clicked_bid_candidate"
            diagnostics["selected_candidate_bid"] = clicked_bid
            diagnostics["select_guard_decision"] = "allow_submit_after_match"
            return MiniWoBGroundingResult(action=browsergym_click_action(clicked_bid, action_syntax=action_syntax), mapping_strategy="select_submit_after_match", mapping_diagnostics=diagnostics)
        diagnostics["submit_allowed"] = False
        candidate_action = browsergym_select_option_action(control_bid, option_text or target_text, action_syntax=action_syntax) if control_bid and _select_option_supported(action_syntax) else ""
        if candidate_action and any(str(item.get("action") or "").strip() == candidate_action and float(item.get("reward") or 0) <= 0 for item in reversed(history or [])):
            diagnostics["select_guard_decision"] = "blocked_submit_after_select_option_no_state_change"
            return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: select_option_no_state_change", selected_candidate=control, repeated_warning="select_option did not change selected value", mapping_strategy="select_option_control", mapping_diagnostics=diagnostics)
        diagnostics["select_guard_decision"] = "blocked_submit_before_select"
        return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: submit_before_select", selected_candidate=control, mapping_strategy="select_option_control", mapping_diagnostics=diagnostics)

    if not control_bid:
        diagnostics["select_guard_decision"] = "blocked_missing_select_control"
        return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: missing_select_control", selected_candidate=control, mapping_strategy="none", mapping_diagnostics=diagnostics)

    if not _select_option_supported(action_syntax):
        diagnostics["select_guard_decision"] = "blocked_select_option_unsupported"
        return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: select_option_unsupported", selected_candidate=control, mapping_strategy="none", mapping_diagnostics=diagnostics)

    mapped = browsergym_select_option_action(control_bid, option_text or target_text, action_syntax=action_syntax)
    if selected_value_matches_target:
        if clicked_bid and is_submit_like_candidate(clicked_candidate):
            diagnostics["submit_allowed"] = True
            diagnostics["submit_source"] = "clicked_bid_candidate"
            diagnostics["selected_candidate_bid"] = clicked_bid
            diagnostics["select_guard_decision"] = "allow_submit_after_match"
            return MiniWoBGroundingResult(
                action=browsergym_click_action(clicked_bid, action_syntax=action_syntax),
                selected_candidate=clicked_candidate,
                mapping_strategy="select_submit_after_match",
                mapping_diagnostics=diagnostics,
            )
        diagnostics["submit_allowed"] = bool(submit_candidate)
        if submit_candidate:
            submit_bid = real_candidate_bid(submit_candidate)
            if submit_bid:
                diagnostics["submit_source"] = "submit_candidate"
                diagnostics["select_guard_decision"] = "map_redundant_select_to_submit"
                return MiniWoBGroundingResult(action=browsergym_click_action(submit_bid, action_syntax=action_syntax), mapping_strategy="select_submit_after_match", mapping_diagnostics=diagnostics)
        diagnostics["select_guard_decision"] = "target_selected_waiting_for_submit"
        return MiniWoBGroundingResult(action="noop()", mapping_error="action_mapping_failure: target_selected_waiting_for_submit", selected_candidate=control, mapping_strategy="select_option_control", mapping_diagnostics=diagnostics)
    progress_error = _select_option_no_progress_error(mapped, control, target_text, history)
    if progress_error:
        diagnostics["select_guard_decision"] = "blocked_select_option_no_progress"
        return MiniWoBGroundingResult(action="noop()", mapping_error=progress_error, selected_candidate=control, repeated_warning="select_option did not change selected value", mapping_strategy="select_option_control", mapping_diagnostics=diagnostics)

    # Override option-bid clicks, combobox clicks, and premature Submit clicks with
    # the deterministic BrowserGym select_option backend on the owning control bid.
    diagnostics["select_guard_decision"] = "override_to_select_option_control"
    return MiniWoBGroundingResult(
        action=mapped,
        selected_candidate=control,
        mapping_strategy="select_option_control",
        mapping_diagnostics=diagnostics,
    )

def find_submit_button(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        role = normalize_candidate_text(candidate.get("role"))
        tag = normalize_candidate_text(candidate.get("tag"))
        typ = normalize_candidate_text(candidate.get("type"))
        if role != "button" and tag != "button" and typ not in {"button", "submit"}:
            continue
        names = {normalize_candidate_text(value) for value in _candidate_text_values(candidate)}
        if names & SUBMIT_BUTTON_ALIASES:
            return candidate
    return None


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if isinstance(bbox, dict):
        x = _numeric(bbox.get("x")) or 0.0
        y = _numeric(bbox.get("y")) or 0.0
        width = _numeric(bbox.get("width"))
        height = _numeric(bbox.get("height"))
        if width is not None and height is not None:
            return x + width / 2.0, y + height / 2.0
        left = _numeric(bbox.get("left"))
        right = _numeric(bbox.get("right"))
        top = _numeric(bbox.get("top"))
        bottom = _numeric(bbox.get("bottom"))
        if None not in (left, right, top, bottom):
            return (left + right) / 2.0, (top + bottom) / 2.0
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        nums = [_numeric(v) for v in bbox[:4]]
        if all(v is not None for v in nums):
            x, y, w, h = nums  # type: ignore[misc]
            return float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
    return None


def candidate_center_with_strategy(candidate: dict[str, Any]) -> tuple[float, float, str] | None:
    coordinate_pairs = (
        ("action_x", "action_y", "coordinate_scaled"),
        ("action_center_x", "action_center_y", "coordinate_scaled"),
        ("browsergym_center_x", "browsergym_center_y", "coordinate_scaled"),
        ("click_x", "click_y", "coordinate_scaled" if str(candidate.get("action_coordinate_space") or "").lower() == "browsergym_scaled" else "coordinate"),
        ("center_x", "center_y", "coordinate_raw"),
    )
    for x_key, y_key, strategy in coordinate_pairs:
        cx = _numeric(candidate.get(x_key))
        cy = _numeric(candidate.get(y_key))
        if cx is not None and cy is not None:
            return cx, cy, strategy
    for bbox_key, strategy in (("action_bbox", "coordinate_scaled"), ("browsergym_bbox", "coordinate_scaled"), ("bbox", "coordinate_raw"), ("bounding_box", "coordinate_raw")):
        center = _bbox_center(candidate.get(bbox_key))
        if center is not None:
            return center[0], center[1], strategy
    return None


def candidate_center(candidate: dict[str, Any]) -> tuple[float, float] | None:
    center = candidate_center_with_strategy(candidate)
    if center is None:
        return None
    return center[0], center[1]


def browsergym_mouse_click_action(x: float, y: float) -> str:
    def fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")

    return f'mouse_click({fmt(x)}, {fmt(y)}, "left")'


def _find_by_real_bid(candidates: list[dict[str, Any]], bid: str) -> dict[str, Any] | None:
    return next((c for c in candidates or [] if real_candidate_bid(c) == bid), None)


def _first_text_arg(parsed: tuple[str, list[Any]] | None) -> str:
    if not parsed or len(parsed[1]) < 2:
        return ""
    return str(parsed[1][1])


def _select_textbox_for_fill(parsed: tuple[str, list[Any]] | None, parsed_response: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    args = parsed[1] if parsed else []
    target_bid = str(parsed_response.get("target_bid") or "").strip()
    first = str(args[0]).strip() if args else ""
    for bid in (target_bid, first):
        selected = _find_by_real_bid(candidates, bid)
        if selected is not None:
            return selected
    boxes = textbox_candidates(candidates)
    if len(boxes) == 1:
        return boxes[0]
    return None


def _should_block_repeated_textbox_click(before: str, candidates: list[dict[str, Any]], history: list[dict]) -> str | None:
    parsed = _parse_call(before)
    if not parsed or parsed[0] != "click" or not parsed[1]:
        return None
    clicked_bid = str(parsed[1][0]).strip()
    selected = _find_by_real_bid(candidates, clicked_bid)
    if not _is_textbox(selected):
        return None
    repeats = 0
    for item in reversed(history):
        if str(item.get("action") or "").strip() == before and float(item.get("reward") or 0) <= 0:
            repeats += 1
        else:
            break
    if repeats >= 2:
        return f"action_mapping_failure: no_progress repeated textbox click {before!r} without text input"
    return None


def _passthrough_call(parsed: tuple[str, list[Any]] | None, before: str) -> MiniWoBGroundingResult | None:
    if not parsed:
        return None
    name, args = parsed
    if name == "fill" and len(args) >= 2:
        return MiniWoBGroundingResult(action=browsergym_fill_action(str(args[0]), str(args[1])), mapping_strategy="bid_fill")
    if name == "clear" and len(args) >= 1:
        return MiniWoBGroundingResult(action=f'clear("{_escape(args[0])}")', mapping_strategy="bid_clear")
    if name == "focus" and len(args) >= 1:
        return MiniWoBGroundingResult(action=f'focus("{_escape(args[0])}")', mapping_strategy="bid_focus")
    if name == "press" and len(args) >= 2:
        return MiniWoBGroundingResult(action=f'press("{_escape(args[0])}", "{_escape(args[1])}")', mapping_strategy="bid_press")
    if name == "select_option" and len(args) >= 2:
        option_arg = args[1]
        if isinstance(option_arg, list):
            option_arg = option_arg[0] if option_arg else ""
        return MiniWoBGroundingResult(action=browsergym_select_option_action(str(args[0]), str(option_arg)), mapping_strategy="bid_select_option")
    if name == "keyboard_type" and len(args) >= 1:
        return MiniWoBGroundingResult(action=f'keyboard_type("{_escape(args[0])}")', mapping_strategy="keyboard_type")
    if name == "keyboard_insert_text" and len(args) >= 1:
        return MiniWoBGroundingResult(action=f'keyboard_insert_text("{_escape(args[0])}")', mapping_strategy="keyboard_insert_text")
    return None


def ground_miniwob_action(
    *,
    action: str,
    parsed_response: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    history: list[dict] | None = None,
    action_syntax: list[str] | None = None,
) -> MiniWoBGroundingResult:
    parsed_response = parsed_response or {}
    before = " ".join(str(action or "").strip().split())
    parsed = _parse_call(before)
    target = str(parsed_response.get("target_text") or "").strip() or _extract_click_target(before)
    target_bid = str(parsed_response.get("target_bid") or "").strip()
    history = history or []

    textbox_repeat_error = _should_block_repeated_textbox_click(before, candidates, history)
    if textbox_repeat_error:
        return MiniWoBGroundingResult(action="noop()", mapping_error=textbox_repeat_error, repeated_warning="repeated textbox click blocked", mapping_strategy="none")

    repeats = 0
    for item in reversed(history):
        if str(item.get("action") or "").strip() == before and float(item.get("reward") or 0) <= 0:
            repeats += 1
        else:
            break
    if repeats >= 2:
        repeated_candidate = None
        repeated_target = str(parsed[1][0]).strip() if parsed and parsed[0] == "click" and parsed[1] else ""
        if repeated_target:
            repeated_candidate = _find_by_real_bid(candidates, repeated_target)
        if _is_select_control(repeated_candidate) or _is_option_candidate(repeated_candidate):
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error="action_mapping_failure: no_progress_repeated_select repeated select/list click without progress",
                repeated_warning="repeated select loop blocked",
                mapping_strategy="none",
            )
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_error=f"action_mapping_failure: repeated ineffective action {before!r} without progress",
            repeated_warning="previous action had no effect; exact repeat blocked",
            mapping_strategy="none",
        )

    instruction = str(parsed_response.get("instruction") or parsed_response.get("miniwob_instruction") or parsed_response.get("task_instruction") or "")
    select_grounding = _ground_select_intent(
        proposed_action=before,
        instruction=instruction,
        parsed_response=parsed_response,
        candidates=candidates,
        history=history,
        action_syntax=action_syntax,
    )
    if select_grounding is not None:
        return select_grounding

    if parsed and parsed[0] in {"fill", "type"}:
        selected = _select_textbox_for_fill(parsed, parsed_response, candidates)
        text = _first_text_arg(parsed) or str(parsed_response.get("text") or parsed_response.get("value") or "")
        if selected is not None:
            bid = real_candidate_bid(selected)
            if bid:
                return MiniWoBGroundingResult(action=browsergym_fill_action(bid, text), selected_candidate=selected, mapping_strategy="bid_fill")
        passthrough = _passthrough_call(parsed, before)
        if passthrough is not None:
            return passthrough

    passthrough = _passthrough_call(parsed, before)
    if passthrough is not None:
        if parsed and parsed[0] == "fill":
            selected = _select_textbox_for_fill(parsed, parsed_response, candidates)
            passthrough.selected_candidate = selected
        return passthrough

    intent = _norm(parsed_response.get("intent") or parsed_response.get("action_intent") or parsed_response.get("rationale") or "")
    if target_bid and any(word in intent for word in TEXT_INPUT_INTENTS):
        selected = _find_by_real_bid(candidates, target_bid)
        if _is_textbox(selected):
            text = str(parsed_response.get("target_text") or parsed_response.get("text") or parsed_response.get("value") or "")
            return MiniWoBGroundingResult(action=browsergym_fill_action(target_bid, text), selected_candidate=selected, mapping_strategy="bid_fill")

    if before.lower().startswith("click") or before.lower().startswith("mouse_click") or target or target_bid:
        selected = None
        click_arg = str(parsed[1][0]).strip() if parsed and parsed[1] else ""
        for bid in (target_bid, click_arg):
            if bid:
                selected = _find_by_real_bid(candidates, bid)
                if selected is not None:
                    break
        if selected is None and target:
            selected = find_click_candidate(candidates, target)
        if selected is not None:
            candidate_id = real_candidate_bid(selected)
            instruction = str(parsed_response.get("instruction") or parsed_response.get("miniwob_instruction") or parsed_response.get("task_instruction") or "")
            text = str(parsed_response.get("text") or parsed_response.get("value") or "")
            if _is_textbox(selected) and candidate_id:
                if not text and parsed_response.get("target_text") and str(parsed_response.get("target_text")).strip() != candidate_id:
                    text = str(parsed_response.get("target_text") or "")
                if not text and _instruction_requires_text_entry(instruction):
                    text = _text_from_instruction_for_fill(instruction, selected)
                if text:
                    return MiniWoBGroundingResult(action=browsergym_fill_action(candidate_id, text), selected_candidate=selected, mapping_strategy="bid_fill")
            if candidate_id:
                return MiniWoBGroundingResult(action=browsergym_click_action(candidate_id, action_syntax=action_syntax), selected_candidate=selected, mapping_strategy="bid_click")
            center = candidate_center_with_strategy(selected)
            if center is not None:
                return MiniWoBGroundingResult(action=browsergym_mouse_click_action(center[0], center[1]), selected_candidate=selected, mapping_strategy=center[2])
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error=f"action_mapping_failure: no grounded bid or bbox for target_text={target!r}",
                selected_candidate=selected,
                mapping_strategy="none",
            )
        if target and not re.search(r"[\d_.:-]", target):
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error=f"action_mapping_failure: no clickable candidate matched target_text={target!r}",
                mapping_strategy="none",
            )
    return MiniWoBGroundingResult(action=before or "noop()")
