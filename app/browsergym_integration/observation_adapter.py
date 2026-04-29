from __future__ import annotations

from typing import Any


def browsergym_obs_to_page_context(obs: dict, info: dict | None = None) -> dict:
    obs = obs if isinstance(obs, dict) else {"raw": obs}
    info = info if isinstance(info, dict) else {}

    text = (
        obs.get("text")
        or obs.get("textual_observation")
        or obs.get("observation")
        or obs.get("utterance")
        or info.get("text")
    )
    if text is None:
        text = str(obs)[:2000]

    context = {
        "url": obs.get("url") or info.get("url") or "",
        "title": obs.get("title") or info.get("title") or "",
        "text": text,
        "axtree": obs.get("axtree") or obs.get("accessibility_tree") or info.get("axtree"),
        "screenshot": obs.get("screenshot") or info.get("screenshot"),
        "visible_text_excerpt": str(text)[:1000],
        "links": obs.get("links") or info.get("links") or [],
        "buttons": obs.get("buttons") or info.get("buttons") or [],
        "clickable_elements": obs.get("clickable_elements") or info.get("clickable_elements") or [],
        "obs_keys": sorted(list(obs.keys())),
        "info_keys": sorted(list(info.keys())),
    }
    return context


def page_context_to_snapshot_like(context: dict) -> dict:
    text = str(context.get("text", "") or "")
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
            label = str(item.get("text") or item.get("name") or item.get("label") or "").strip()
            if label:
                result.append(label)
    return result[:20]
