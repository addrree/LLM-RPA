from __future__ import annotations

import html
import re
from typing import Any

CLICKABLE_ROLES = {"button", "link", "checkbox", "radio", "textbox", "combobox", "option", "menuitem"}
CLICKABLE_TAGS = {"button", "a", "input", "select", "textarea", "label"}
ID_KEYS = ("bid", "element_id", "node_id", "backend_node_id", "id")
REAL_BID_FIELD_SOURCES = (
    ("bid", "bid"),
    ("data-testid", "data-testid"),
    ("data_testid", "data_testid"),
    ("dataTestId", "data-testid"),
    ("browsergym_id", "browsergym_id"),
    ("browsergymId", "browsergym_id"),
    ("data-bid", "data-bid"),
    ("data_bid", "data_bid"),
    ("dataBid", "data-bid"),
    ("ref", "ref"),
)
TEXT_KEYS = ("name", "text", "label", "ariaLabel", "aria_label", "title", "value", "content", "inner_text", "innerText")


def get_first_not_none(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _serialize_field_summary(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"kind": type(value).__name__, "shape": tuple(shape), "dtype": str(dtype) if dtype is not None else None}
    if isinstance(value, str):
        return {"kind": "str", "length": len(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "size": len(value), "keys": sorted(list(value.keys()))[:20]}
    if isinstance(value, list):
        return {"kind": "list", "size": len(value)}
    return {"kind": type(value).__name__}


def _stringify_instruction(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("goal", "instruction", "intent", "task", "utterance"):
            if value.get(key):
                return _stringify_instruction(value.get(key))
        return " ".join(f"{k}: {_stringify_instruction(v)}" for k, v in list(value.items())[:5] if _stringify_instruction(v)).strip()
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value[:5]:
            if isinstance(item, dict):
                text = get_first_not_none(item, "content", "text", "message", "utterance")
                if text is not None:
                    parts.append(_stringify_instruction(text))
            else:
                parts.append(_stringify_instruction(item))
        return " ".join(part for part in parts if part).strip()
    if getattr(value, "shape", None) is not None:
        return ""
    return str(value).strip()


def _safe_text(value: Any, limit: int = 4000) -> str:
    if value is None or getattr(value, "shape", None) is not None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (dict, list, tuple)):
        return str(value)[:limit]
    return str(value)[:limit]


def extract_goal_instruction(obs: dict[str, Any], info: dict[str, Any]) -> str:
    for source in (obs, info):
        for key in ("goal", "instruction", "intent", "task_goal", "utterance", "task_info", "chat_messages", "goal_object"):
            text = _stringify_instruction(source.get(key))
            if text:
                return text[:1200]
    return ""


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no", "disabled"}:
            return False
    return None


PRESERVED_CANDIDATE_FIELDS = (
    "page_center_x",
    "page_center_y",
    "browsergym_center_x",
    "browsergym_center_y",
    "browsergym_scale_factor",
    "coordinate_space",
    "action_coordinate_space",
    "action_x",
    "action_y",
    "action_center_x",
    "action_center_y",
    "click_x",
    "click_y",
    "center_x",
    "center_y",
)

PRESERVED_BBOX_FIELDS = ("browsergym_bbox", "action_bbox", "bbox", "bounding_box")
DANGEROUS_CANDIDATE_FIELD_PATTERNS = ("screenshot", "image", "dom", "html", "axtree", "raw")


def _is_safe_candidate_scalar(key: str, value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return value is not None
    if isinstance(value, str):
        key_l = key.lower()
        if any(pattern in key_l for pattern in DANGEROUS_CANDIDATE_FIELD_PATTERNS):
            return False
        return len(value) <= 500
    return False


def _candidate_from_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    existing_bid_source = str(item.get("bid_source") or "").strip()
    for key, source in REAL_BID_FIELD_SOURCES:
        value = item.get(key)
        if value is not None and str(value).strip():
            out["bid"] = str(value).strip()
            out["bid_source"] = existing_bid_source or source
            out[key] = str(value).strip()
            break
    for key in ("element_id", "node_id", "backend_node_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            out[key] = str(value).strip()
    for key in ("role", "kind", "name", "text", "label", "tag", "type", "value", "ariaLabel", "aria_label", "title", "parent_bid", "owner_bid", "select_bid", "parent_name"):
        value = item.get(key)
        if value not in (None, ""):
            out[key] = str(value).strip() if isinstance(value, str) else value
    for key in ("enabled", "visible", "clickable", "disabled", "selected"):
        if key in item:
            bool_value = _as_bool(item.get(key))
            out[key] = bool_value if bool_value is not None else item.get(key)
    for key in PRESERVED_BBOX_FIELDS:
        value = item.get(key)
        if value not in (None, ""):
            out[key] = value
    for key in PRESERVED_CANDIDATE_FIELDS:
        if item.get(key) is not None:
            out[key] = item.get(key)
    for key, value in item.items():
        if key in out or key in PRESERVED_BBOX_FIELDS or key in PRESERVED_CANDIDATE_FIELDS:
            continue
        if _is_safe_candidate_scalar(str(key), value):
            out[str(key)] = value
    if not any(k in out and str(out[k]).strip() for k in ("name", "text", "label", "value", "ariaLabel", "aria_label")):
        text = get_first_not_none(item, "content", "inner_text", "innerText", "aria-label")
        if text not in (None, ""):
            out["text"] = str(text).strip()
    return out if out else None


def _candidate_is_clickable(candidate: dict[str, Any]) -> bool:
    role = str(candidate.get("role") or "").strip().lower()
    tag = str(candidate.get("tag") or "").strip().lower()
    if role in CLICKABLE_ROLES or tag in CLICKABLE_TAGS:
        return True
    if candidate.get("clickable") is True or candidate.get("enabled") is True:
        return True
    has_id = any(candidate.get(key) for key in ID_KEYS)
    has_text = any(str(candidate.get(key) or "").strip() for key in ("name", "text", "label", "value", "ariaLabel", "aria_label"))
    return has_id and has_text


def _dedupe_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(str(candidate.get(k) or "").strip().lower() for k in ("bid", "element_id", "node_id", "backend_node_id", "id", "role", "tag", "name", "text", "label", "value"))


def _append_candidate(candidates: list[dict[str, Any]], candidate: dict[str, Any] | None, seen: set[tuple[Any, ...]]) -> None:
    if not candidate or not _candidate_is_clickable(candidate):
        return
    key = _dedupe_key(candidate)
    if key not in seen:
        seen.add(key)
        candidates.append(candidate)


def _walk_candidates(value: Any, candidates: list[dict[str, Any]], seen: set[tuple[Any, ...]]) -> None:
    if isinstance(value, dict):
        _append_candidate(candidates, _candidate_from_dict(value), seen)
        for child_key in ("children", "childNodes", "nodes", "items", "elements"):
            child = value.get(child_key)
            if child is not None:
                _walk_candidates(child, candidates, seen)
        # Some BrowserGym objects use arbitrary nested fields.
        for key, child in value.items():
            if key not in {"children", "childNodes", "nodes", "items", "elements"} and isinstance(child, (dict, list, tuple)):
                _walk_candidates(child, candidates, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_candidates(item, candidates, seen)


def _parse_axtree_clickables(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for line in str(text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        bid_match = re.search(r"\bbid\s*[=:]\s*['\"]?([A-Za-z0-9_.:-]+)", raw, flags=re.IGNORECASE)
        if not bid_match:
            bid_match = re.search(r"^\s*\[?([A-Za-z0-9_.:-]+)\]?\s+(?:button|link|input|textbox|checkbox|radio|combobox|option|menuitem)\b", raw, flags=re.IGNORECASE)
        role_match = re.search(r"\b(button|link|input|textbox|checkbox|radio|combobox|menuitem|option)\b", raw, flags=re.IGNORECASE)
        name_match = re.search(r"['\"]([^'\"]{1,120})['\"]", raw)
        if not (bid_match or role_match or name_match):
            continue
        candidate: dict[str, Any] = {"raw": raw[:240]}
        if bid_match:
            candidate["bid"] = bid_match.group(1)
            candidate["bid_source"] = "bid"
        if role_match:
            role = role_match.group(1).lower()
            candidate["role"] = "textbox" if role == "input" else role
        if name_match:
            candidate["name"] = name_match.group(1).strip()
        elif role_match:
            tail = raw[role_match.end():].strip(" :-\t")
            if tail:
                candidate["text"] = tail[:120]
        _append_candidate(candidates, candidate, seen)
    return candidates


def _parse_html_clickables(markup: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    pattern = re.compile(r"<(?P<tag>button|a|input|select|textarea|label)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>|<(?P<selftag>input)\b(?P<selfattrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
    attr_re = re.compile(r"([:\w-]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+)))?", re.IGNORECASE)
    for match in pattern.finditer(str(markup or "")[:60000]):
        tag = (match.group("tag") or match.group("selftag") or "").lower()
        attrs_raw = match.group("attrs") or match.group("selfattrs") or ""
        attrs = {m.group(1).lower(): html.unescape(m.group(2) or m.group(3) or m.group(4) or "") for m in attr_re.finditer(attrs_raw)}
        body = re.sub(r"<[^>]+>", " ", match.group("body") or "")
        text = html.unescape(re.sub(r"\s+", " ", body).strip()) or attrs.get("value") or attrs.get("aria-label") or attrs.get("title") or attrs.get("name") or attrs.get("id")
        candidate = {
            "tag": tag,
            "role": attrs.get("role") or ("button" if tag == "button" or attrs.get("type") in {"button", "submit"} else "link" if tag == "a" else "textbox" if tag in {"input", "textarea"} else "combobox" if tag == "select" else ""),
            "text": text,
            "value": attrs.get("value"),
            "label": attrs.get("aria-label"),
            "id": attrs.get("id"),
            "name": attrs.get("name") or text,
            "visible": True,
            "enabled": "disabled" not in attrs,
            "disabled": "disabled" in attrs,
        }
        if attrs.get("bid"):
            candidate["bid"] = attrs["bid"]
            candidate["bid_source"] = "bid"
        elif attrs.get("data-testid"):
            candidate["bid"] = attrs["data-testid"]
            candidate["bid_source"] = "data-testid"
        elif attrs.get("browsergym_id"):
            candidate["bid"] = attrs["browsergym_id"]
            candidate["bid_source"] = "browsergym_id"
        elif attrs.get("data-bid"):
            candidate["bid"] = attrs["data-bid"]
            candidate["bid_source"] = "data-bid"
        elif attrs.get("ref"):
            candidate["bid"] = attrs["ref"]
            candidate["bid_source"] = "ref"
        if attrs.get("id"):
            candidate["element_id"] = attrs["id"]
        normalized_candidate = _candidate_from_dict(candidate)
        _append_candidate(candidates, normalized_candidate, seen)
        if tag == "select":
            parent_bid = (normalized_candidate or {}).get("bid") or ""
            option_re = re.compile(r"<option\b(?P<attrs>[^>]*)>(?P<body>.*?)</option>|<option\b(?P<selfattrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
            for option_match in option_re.finditer(match.group("body") or ""):
                option_attrs_raw = option_match.group("attrs") or option_match.group("selfattrs") or ""
                option_attrs = {m.group(1).lower(): html.unescape(m.group(2) or m.group(3) or m.group(4) or "") for m in attr_re.finditer(option_attrs_raw)}
                option_body = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", option_match.group("body") or "")).strip())
                option_candidate = {
                    "tag": "option",
                    "role": "option",
                    "text": option_body or option_attrs.get("label") or option_attrs.get("value"),
                    "value": option_attrs.get("value") or option_body,
                    "label": option_attrs.get("label"),
                    "name": option_attrs.get("label") or option_body or option_attrs.get("value"),
                    "parent_bid": parent_bid,
                    "owner_bid": parent_bid,
                    "parent_name": (normalized_candidate or {}).get("name") or (normalized_candidate or {}).get("text") or attrs.get("name") or attrs.get("id"),
                    "selected": "selected" in option_attrs,
                    "disabled": "disabled" in option_attrs,
                    "enabled": "disabled" not in option_attrs,
                    "visible": True,
                    "clickable": True,
                }
                for bid_key, source_name in (("bid", "bid"), ("data-testid", "data-testid"), ("browsergym_id", "browsergym_id"), ("data-bid", "data-bid"), ("ref", "ref")):
                    if option_attrs.get(bid_key):
                        option_candidate["bid"] = option_attrs[bid_key]
                        option_candidate["bid_source"] = source_name
                        break
                _append_candidate(candidates, _candidate_from_dict(option_candidate), seen)
    return candidates



def _role_text(candidate: dict[str, Any]) -> str:
    role = candidate.get("role")
    if isinstance(role, dict):
        return str(role.get("value") or role.get("name") or "").strip().lower()
    return str(role or "").strip().lower()


def _candidate_tag(candidate: dict[str, Any]) -> str:
    return str(candidate.get("tag") or candidate.get("kind") or "").strip().lower()


def _is_select_control_candidate(candidate: dict[str, Any]) -> bool:
    return _role_text(candidate) in {"combobox", "listbox", "select"} or _candidate_tag(candidate) == "select"


def _is_option_candidate(candidate: dict[str, Any]) -> bool:
    return _role_text(candidate) in {"option", "listitem", "menuitem", "radio"} or _candidate_tag(candidate) in {"option", "li"}


def _is_submit_candidate(candidate: dict[str, Any]) -> bool:
    role = _role_text(candidate)
    tag = _candidate_tag(candidate)
    typ = str(candidate.get("type") or "").strip().lower()
    if role != "button" and tag != "button" and typ not in {"button", "submit"}:
        return False
    names = {str(candidate.get(key) or "").strip().lower() for key in ("name", "text", "label", "value", "ariaLabel", "aria_label")}
    return bool(names & {"submit", "login", "ok", "done"})


def _candidate_public_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("bid", "bid_source", "role", "kind", "tag", "value", "selected_value", "current_value", "name", "text", "label", "enabled", "visible", "selected", "parent_bid", "owner_bid", "select_bid", "parent_name")
    return {key: candidate.get(key) for key in keys if key in candidate}

def extract_clickable_candidates(obs: dict[str, Any], info: dict[str, Any], *, limit: int = 30) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in (obs, info):
        for key in ("clickable_candidates", "page_clickable_candidates", "clickable_elements", "buttons", "links", "elements", "interactive_elements"):
            value = source.get(key)
            if value:
                _walk_candidates(value, candidates, seen)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            _append_candidate(candidates, {"text": item.strip()[:120], "clickable": True}, seen)
    for source in (obs, info):
        for key in ("axtree_object", "dom_object"):
            value = source.get(key)
            if value:
                _walk_candidates(value, candidates, seen)
    axtree_text = _safe_text(get_first_not_none(obs, "axtree_txt", "axtree", "accessibility_tree", "text_tree") or get_first_not_none(info, "axtree_txt", "axtree", "accessibility_tree"), 12000)
    for candidate in _parse_axtree_clickables(axtree_text):
        _append_candidate(candidates, candidate, seen)
    html_text = _safe_text(get_first_not_none(obs, "pruned_html", "html") or get_first_not_none(info, "pruned_html", "html"), 60000)
    for candidate in _parse_html_clickables(html_text):
        _append_candidate(candidates, candidate, seen)
    return candidates[:limit]


def browsergym_obs_to_page_context(obs: dict, info: dict | None = None) -> dict:
    obs = obs if isinstance(obs, dict) else {"raw": obs}
    info = info if isinstance(info, dict) else {}

    text = get_first_not_none(obs, "text", "textual_observation", "observation", "utterance")
    if text is None:
        text = get_first_not_none(info, "text")
    if text is None:
        text = ""

    pruned_html = get_first_not_none(obs, "pruned_html", "dom", "html")
    if pruned_html is None:
        pruned_html = get_first_not_none(info, "pruned_html", "dom", "html")

    raw_screenshot = get_first_not_none(obs, "screenshot")
    if raw_screenshot is None:
        raw_screenshot = get_first_not_none(info, "screenshot")
    raw_image = get_first_not_none(obs, "image")
    if raw_image is None:
        raw_image = get_first_not_none(info, "image")

    axtree = get_first_not_none(obs, "axtree_txt", "axtree", "accessibility_tree", "text_tree")
    if axtree is None:
        axtree = get_first_not_none(info, "axtree_txt", "axtree", "accessibility_tree", "text_tree")

    goal_instruction = extract_goal_instruction(obs, info)
    clickable_candidates = extract_clickable_candidates(obs, info, limit=30)
    select_control_candidates = [_candidate_public_summary(candidate) for candidate in clickable_candidates if _is_select_control_candidate(candidate)]
    option_candidates = [_candidate_public_summary(candidate) for candidate in clickable_candidates if _is_option_candidate(candidate)]
    submit_candidates = [_candidate_public_summary(candidate) for candidate in clickable_candidates if _is_submit_candidate(candidate)]

    context = {
        "url": get_first_not_none(obs, "url") if get_first_not_none(obs, "url") is not None else (get_first_not_none(info, "url") or ""),
        "title": get_first_not_none(obs, "title") if get_first_not_none(obs, "title") is not None else (get_first_not_none(info, "title") or ""),
        "open_pages_titles": get_first_not_none(obs, "open_pages_titles") or get_first_not_none(info, "open_pages_titles") or [],
        "goal_instruction": goal_instruction,
        "instruction": goal_instruction,
        "text": text,
        "text_excerpt": str(text)[:1200],
        "visible_text_excerpt": str(text)[:1200],
        "axtree_excerpt": _safe_text(axtree, 1200),
        "pruned_html_excerpt": _safe_text(pruned_html, 1200),
        "dom_excerpt": _safe_text(pruned_html, 1200),
        "clickable_candidates": clickable_candidates,
        "clickable_candidates_count": len(clickable_candidates),
        "select_control_candidates": select_control_candidates,
        "option_candidates": option_candidates,
        "submit_candidates": submit_candidates,
        "screenshot": None,
        "image": None,
        "screenshot_summary": _serialize_field_summary(raw_screenshot),
        "image_summary": _serialize_field_summary(raw_image),
        "links": get_first_not_none(obs, "links") if get_first_not_none(obs, "links") is not None else (get_first_not_none(info, "links") or []),
        "buttons": get_first_not_none(obs, "buttons") if get_first_not_none(obs, "buttons") is not None else (get_first_not_none(info, "buttons") or []),
        "obs_keys": sorted(list(obs.keys())),
        "info_keys": sorted(list(info.keys())),
        "observation_summary": {
            "screenshot": _serialize_field_summary(raw_screenshot),
            "image": _serialize_field_summary(raw_image),
            "axtree": _serialize_field_summary(axtree),
            "pruned_html": _serialize_field_summary(pruned_html),
        },
    }
    return context


def page_context_to_snapshot_like(context: dict) -> dict:
    raw_text = context.get("text", "")
    text = str(raw_text) if raw_text is not None else ""
    headings: list[str] = []
    for line in text.splitlines()[:30]:
        raw = line.strip()
        if raw and len(raw) <= 100 and raw[0].isupper():
            headings.append(raw)
    buttons = context.get("buttons") or []
    links = context.get("links") or []
    return {
        "url": context.get("url", ""),
        "title": context.get("title", ""),
        "page_text": text or context.get("text_excerpt", ""),
        "visible_headings": headings[:10],
        "buttons": buttons if isinstance(buttons, list) else [],
        "links": links if isinstance(links, list) else [],
        "goal_instruction": context.get("goal_instruction", ""),
        "source": "browsergym",
        "clickable_candidates": context.get("clickable_candidates", []),
        "select_control_candidates": context.get("select_control_candidates", []),
        "option_candidates": context.get("option_candidates", []),
        "submit_candidates": context.get("submit_candidates", []),
    }
