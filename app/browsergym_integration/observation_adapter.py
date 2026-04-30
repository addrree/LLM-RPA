from __future__ import annotations

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


def browsergym_obs_to_page_context(obs: dict, info: dict | None = None) -> dict:
    obs = obs if isinstance(obs, dict) else {"raw": obs}
    info = info if isinstance(info, dict) else {}

    text = get_first_not_none(obs, "text", "textual_observation", "observation", "utterance")
    if text is None:
        text = get_first_not_none(info, "text")
    if text is None:
        text = str(obs)[:2000]

    raw_screenshot = get_first_not_none(obs, "screenshot")
    if raw_screenshot is None:
        raw_screenshot = get_first_not_none(info, "screenshot")
    raw_image = get_first_not_none(obs, "image")
    if raw_image is None:
        raw_image = get_first_not_none(info, "image")

    axtree = get_first_not_none(obs, "axtree", "accessibility_tree")
    if axtree is None:
        axtree = get_first_not_none(info, "axtree")

    context = {
        "url": get_first_not_none(obs, "url") if get_first_not_none(obs, "url") is not None else (get_first_not_none(info, "url") or ""),
        "title": get_first_not_none(obs, "title") if get_first_not_none(obs, "title") is not None else (get_first_not_none(info, "title") or ""),
        "text": text,
        "axtree": axtree,
        "screenshot": None,
        "image": None,
        "screenshot_summary": _serialize_field_summary(raw_screenshot),
        "image_summary": _serialize_field_summary(raw_image),
        "visible_text_excerpt": str(text)[:1000],
        "links": get_first_not_none(obs, "links") if get_first_not_none(obs, "links") is not None else (get_first_not_none(info, "links") or []),
        "buttons": get_first_not_none(obs, "buttons") if get_first_not_none(obs, "buttons") is not None else (get_first_not_none(info, "buttons") or []),
        "clickable_elements": get_first_not_none(obs, "clickable_elements") if get_first_not_none(obs, "clickable_elements") is not None else (get_first_not_none(info, "clickable_elements") or []),
        "obs_keys": sorted(list(obs.keys())),
        "info_keys": sorted(list(info.keys())),
        "observation_summary": {
            "screenshot": _serialize_field_summary(raw_screenshot),
            "image": _serialize_field_summary(raw_image),
            "axtree": _serialize_field_summary(axtree),
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
