from __future__ import annotations

from dataclasses import dataclass
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


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


REAL_BID_KEYS = ("bid", "data-testid", "data_testid", "browsergym_id", "data-bid", "data_bid", "ref")
REAL_BID_SOURCES = {"bid", "data-testid", "data_testid", "browsergym_id", "data-bid", "data_bid", "ref"}
FAKE_BID_SOURCES = {"id", "dom_id", "element_id", "index", "candidate_index", "node_id", "backend_node_id"}
SUBMIT_BUTTON_NAMES = {"submit", "login", "ok", "done"}
TEXT_INPUT_INTENTS = {"fill", "type", "enter", "input", "text", "username", "password"}


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
    for key in ("name", "text", "value", "label", "ariaLabel", "aria_label", "title"):
        value = candidate.get(key)
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


def find_submit_button(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        role = _norm(candidate.get("role"))
        tag = _norm(candidate.get("tag"))
        typ = _norm(candidate.get("type"))
        if role != "button" and tag != "button" and typ not in {"button", "submit"}:
            continue
        names = {_norm(value) for value in _candidate_text_values(candidate)}
        if names & SUBMIT_BUTTON_NAMES:
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
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_error=f"action_mapping_failure: repeated ineffective action {before!r} without progress",
            repeated_warning="previous action had no effect; exact repeat blocked",
            mapping_strategy="none",
        )

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
