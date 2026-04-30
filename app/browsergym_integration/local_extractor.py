from __future__ import annotations

import re
from typing import Any


def _as_text(value: Any, limit: int = 3000) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:limit]


def extract_heading_from_observation(obs_summary: dict) -> str:
    axtree = _as_text(obs_summary.get("axtree_excerpt"))
    text = _as_text(obs_summary.get("text_excerpt"))
    title = _as_text(obs_summary.get("title"))
    open_titles = obs_summary.get("open_pages_titles") or []

    for source in (axtree, text):
        for line in source.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("h1"):
                parts = re.split(r"[:\-]\s*", line, maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()[:200]
            if 3 <= len(line) <= 120 and line[0].isupper():
                return line[:200]
    if title:
        return title[:200]
    for t in open_titles:
        if str(t).strip():
            return str(t).strip()[:200]
    return ""


def extract_text_from_observation(obs_summary: dict, selector: str | None = None, goal: str | None = None) -> str:
    if selector and selector.strip().lower() == "h1":
        heading = extract_heading_from_observation(obs_summary)
        if heading:
            return heading
    if goal and "heading" in goal.lower():
        heading = extract_heading_from_observation(obs_summary)
        if heading:
            return heading
    text = _as_text(obs_summary.get("text_excerpt") or obs_summary.get("text"))
    return text.splitlines()[0].strip()[:200] if text.strip() else ""


def extract_pattern_from_observation(obs_summary: dict, pattern: str, case_insensitive: bool = False) -> str:
    flags = re.IGNORECASE if case_insensitive else 0
    text = "\n".join([
        _as_text(obs_summary.get("text_excerpt")),
        _as_text(obs_summary.get("axtree_excerpt")),
        _as_text(obs_summary.get("title")),
    ])
    m = re.search(pattern, text, flags=flags)
    return m.group(0).strip()[:200] if m else ""


def extract_structured_items_from_observation(obs_summary: dict, limit: int = 10) -> list[str]:
    values: list[str] = []
    for key in ("links", "buttons", "visible_headings"):
        for item in obs_summary.get(key) or []:
            s = str(item).strip()
            if s and s not in values:
                values.append(s)
            if len(values) >= limit:
                return values
    return values


def extract_value_near_anchor_from_observation(obs_summary: dict, anchor_candidates: list[str] | None = None, value_type: str | None = None) -> str:
    text = _as_text(obs_summary.get("text_excerpt") or obs_summary.get("text"))
    anchors = anchor_candidates or []
    for line in text.splitlines():
        low = line.lower()
        if any(a.lower() in low for a in anchors):
            return line.strip()[:200]
    return ""
