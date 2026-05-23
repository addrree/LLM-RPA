from __future__ import annotations

import re
from typing import Any


def _candidate_text(c: dict[str, Any]) -> str:
    for k in ("text", "innerText", "name", "label", "value"):
        v = c.get(k)
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_numbers(text: str) -> list[dict[str, Any]]:
    out = []
    for m in re.finditer(r"[-+]?\d+(?:[\.,]\d+)?", text or ""):
        raw = m.group(0)
        try:
            num = float(raw.replace(",", "."))
        except Exception:
            continue
        out.append({"raw": raw, "value": num, "start": m.start(), "end": m.end(), "line": text.strip(), "context": text[max(0, m.start()-15):m.end()+15]})
    return out


def extract_labeled_items(candidates: list[dict[str, Any]], text_lines: list[str] | None = None) -> list[dict[str, Any]]:
    items = []
    for c in candidates or []:
        text = _candidate_text(c)
        if not text:
            continue
        items.append({
            "label": c.get("label") or c.get("name") or text,
            "value": c.get("value"),
            "text": text,
            "role": c.get("role"),
            "tag": c.get("tag"),
            "class": c.get("className"),
            "bbox": c.get("bbox"),
            "position": {"x": c.get("browsergym_center_x") or c.get("center_x"), "y": c.get("browsergym_center_y") or c.get("center_y")},
            "source": c.get("source", "candidate"),
            "candidate": c,
        })
    return items


def extract_list_items(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for c in candidates or []:
        role = str(c.get("role") or "").lower()
        if role in {"listitem", "option", "menuitem", "row"} or any(k in str(c.get("className") or "").lower() for k in ["list", "item", "result", "row"]):
            t = _candidate_text(c)
            if t:
                items.append({"text": t, "candidate": c})
    return items


def extract_grid_items(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates or []:
        role = str(c.get("role") or "").lower()
        cls = str(c.get("className") or "").lower()
        if role in {"gridcell", "cell", "row", "columnheader"} or any(k in cls for k in ["grid", "table", "calendar", "cell"]):
            t = _candidate_text(c)
            if t:
                out.append({"text": t, "row": c.get("row"), "column": c.get("column"), "candidate": c})
    return out


def extract_email_items(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates or []:
        txt = " ".join(str(c.get(k) or "") for k in ["text", "innerText", "name", "label", "ariaLabel"])
        low = txt.lower()
        if any(k in low for k in ["subject", "from", "inbox", "email", "message", "@"]):
            out.append({"sender": c.get("sender"), "subject": c.get("subject") or _candidate_text(c), "snippet": c.get("snippet"), "important": any(k in low for k in ["important", "starred", "★", "!"]), "candidate": c, "text": txt.strip()})
    return out


def extract_calendar_items(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates or []:
        txt = _candidate_text(c)
        low = txt.lower()
        if any(k in low for k in ["am", "pm", ":"]) or any(k in str(c.get("className") or "").lower() for k in ["calendar", "event", "slot"]):
            tm = re.search(r"\b\d{1,2}:\d{2}(?:\s?[ap]m)?\b", low)
            out.append({"time": tm.group(0) if tm else None, "title": txt, "text": txt, "position": c.get("bbox") or {}, "candidate": c})
    return out


def extract_tree_items(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates or []:
        role = str(c.get("role") or "").lower()
        cls = str(c.get("className") or "").lower()
        if role in {"treeitem", "node"} or any(k in cls for k in ["tree", "node"]):
            txt = _candidate_text(c)
            out.append({"node_text": txt, "depth": c.get("depth"), "parent": c.get("parent"), "expanded": c.get("expanded"), "candidate": c})
    return out


def build_extraction_context(obs: dict[str, Any] | None, context: dict[str, Any] | None, candidates: list[dict[str, Any]] | None) -> dict[str, Any]:
    context = context or {}
    candidates = list(candidates or [])
    visible_text = str(context.get("axtree_excerpt") or context.get("goal_instruction") or "")
    if isinstance(obs, dict):
        visible_text = str(obs.get("visible_text") or visible_text)
    text_lines = [ln.strip() for ln in visible_text.splitlines() if ln.strip()]
    nums = []
    for ln in text_lines:
        nums.extend(extract_numbers(ln))
    cand_texts = [_candidate_text(c) for c in candidates if _candidate_text(c)]
    clickable = [t for c, t in ((c, _candidate_text(c)) for c in candidates) if t and (c.get("bid") or c.get("browsergym_center_x") is not None)]
    return {
        "visible_text": visible_text,
        "text_lines": text_lines,
        "numeric_values": nums,
        "candidate_texts": cand_texts,
        "clickable_texts": clickable,
        "tables_or_grid_like_blocks": extract_grid_items(candidates),
        "email_like_items": extract_email_items(candidates),
        "calendar_like_items": extract_calendar_items(candidates),
        "tree_like_items": extract_tree_items(candidates),
        "shape_like_items": [i for i in extract_labeled_items(candidates) if any(k in str(i.get("text", "")).lower() for k in ["triangle", "square", "circle", "shape"])],
        "list_like_items": extract_list_items(candidates),
        "raw_candidates_summary": [{"text": _candidate_text(c), "role": c.get("role"), "tag": c.get("tag"), "class": c.get("className"), "bid": c.get("bid")} for c in candidates],
    }
