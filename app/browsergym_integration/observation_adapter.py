from __future__ import annotations

import re
from typing import Any


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
        return {
            "kind": type(value).__name__,
            "shape": tuple(shape),
            "dtype": str(dtype) if dtype is not None else None,
        }
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
        for key in ("goal", "instruction", "intent", "task_goal", "utterance", "task_info", "chat_messages"):
            text = _stringify_instruction(source.get(key))
            if text:
                return text[:1200]
    return ""


def _candidate_from_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for key in ("bid", "element_id", "backend_node_id", "node_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            out[key] = str(value).strip()
            break
    for key in ("role", "name", "text", "label", "tag", "enabled", "visible", "clickable"):
        value = item.get(key)
        if value not in (None, ""):
            out[key] = value
    if not any(k in out for k in ("name", "text", "label")):
        text = get_first_not_none(item, "content", "inner_text", "aria_label", "title", "value")
        if text not in (None, ""):
            out["text"] = str(text).strip()
    return out if out else None


def _parse_axtree_clickables(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        # Common BrowserGym axtree formats include bid/id, role and quoted accessible name.
        bid_match = re.search(r"\b(?:bid|id|node_id|backend_node_id)\s*[=:]\s*['\"]?([A-Za-z0-9_.:-]+)", raw, flags=re.IGNORECASE)
        if not bid_match:
            bid_match = re.search(r"^\s*\[?([A-Za-z0-9_.:-]+)\]?\s+(?:button|link|input|textbox|checkbox|radio)\b", raw, flags=re.IGNORECASE)
        role_match = re.search(r"\b(button|link|input|textbox|checkbox|radio|menuitem|option)\b", raw, flags=re.IGNORECASE)
        name_match = re.search(r"['\"]([^'\"]{1,120})['\"]", raw)
        if not (bid_match or role_match or name_match):
            continue
        candidate: dict[str, Any] = {"raw": raw[:240]}
        if bid_match:
            candidate["bid"] = bid_match.group(1)
        if role_match:
            candidate["role"] = role_match.group(1).lower()
        if name_match:
            candidate["name"] = name_match.group(1).strip()
        elif role_match:
            tail = raw[role_match.end():].strip(" :-\t")
            if tail:
                candidate["text"] = tail[:120]
        if candidate.get("bid") or candidate.get("name") or candidate.get("text"):
            candidates.append(candidate)
    return candidates


def extract_clickable_candidates(obs: dict[str, Any], info: dict[str, Any], *, limit: int = 30) -> list[dict[str, Any]]:
    pools: list[Any] = []
    for source in (obs, info):
        for key in ("clickable_candidates", "clickable_elements", "buttons", "links", "elements", "interactive_elements"):
            value = source.get(key)
            if value:
                pools.append(value)
    candidates: list[dict[str, Any]] = []
    for pool in pools:
        values = pool if isinstance(pool, list) else [pool]
        for item in values:
            candidate = None
            if isinstance(item, dict):
                candidate = _candidate_from_dict(item)
            elif isinstance(item, str) and item.strip():
                candidate = {"text": item.strip()[:120]}
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    axtree_text = _safe_text(get_first_not_none(obs, "axtree_txt", "axtree", "accessibility_tree", "text_tree") or get_first_not_none(info, "axtree_txt", "axtree", "accessibility_tree"), 12000)
    for candidate in _parse_axtree_clickables(axtree_text):
        if candidate not in candidates:
            candidates.append(candidate)
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
    return {
        "url": context.get("url", ""),
        "title": context.get("title", ""),
        "page_text": text,
        "visible_headings": headings[:10],
        "links": _normalize_clickables(context.get("links", [])),
        "buttons": _normalize_clickables(context.get("buttons", [])),
        "clickable_candidates": context.get("clickable_candidates", [])[:30],
        "source": "browsergym",
    }


def _normalize_clickables(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict):
            label_value = get_first_not_none(item, "text", "name", "label")
            label = str(label_value).strip() if label_value is not None else ""
            if label:
                result.append(label)
    return result[:20]
