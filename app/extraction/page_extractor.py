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


def _candidate_all_text(c: dict[str, Any]) -> dict[str, str]:
    fields = ("text", "innerText", "textContent", "value", "name", "ariaLabel", "title", "parent_text")
    out: dict[str, str] = {}
    for f in fields:
        v = c.get(f)
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, str) and v.strip():
            out[f] = v.strip()
    return out


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
        txt = " ".join(str(c.get(k) or "") for k in ["text", "innerText", "textContent", "name", "label", "ariaLabel"]).strip()
        low = txt.lower()
        cls = str(c.get("className") or "").lower()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        is_thread = "email-thread" in cls or "mail" in cls or "thread" in cls or "inbox" in cls or (2 <= len(lines) <= 4 and str(c.get("tag") or "").lower() == "div" and any(k in cls for k in ["mail", "inbox", "thread"]))
        if (len(lines) > 8 and "email-thread" not in cls) or (len(txt) > 600 and "email-thread" not in cls):
            continue
        if is_thread or any(k in low for k in ["subject", "from", "inbox", "email", "message"]):
            sender = lines[0] if len(lines) >= 1 else c.get("sender")
            subject = lines[1] if len(lines) >= 2 else (c.get("subject") or _candidate_text(c))
            snippet = lines[2] if len(lines) >= 3 else c.get("snippet")
            important = any(k in (low + " " + cls + " " + str(c.get("title") or "").lower()) for k in ["important", "starred", "★"])
            out.append({"sender": sender, "subject": subject, "snippet": snippet, "important": important, "candidate": c, "text": txt.strip()})
    return out


def extract_numbers_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates or []:
        for field, txt in _candidate_all_text(c).items():
            for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", txt):
                raw = m.group(0)
                try:
                    val = float(raw)
                except Exception:
                    continue
                out.append({"value": int(val) if val.is_integer() else val, "raw_text": raw, "candidate": c, "candidate_bid": c.get("bid"), "bbox": c.get("bbox"), "position": {"x": c.get("browsergym_center_x") or c.get("center_x"), "y": c.get("browsergym_center_y") or c.get("center_y")}, "line": txt, "context": txt[max(0, m.start()-12):m.end()+12], "source_field": field})
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
    numeric_values = nums + extract_numbers_from_candidates(candidates)
    card_like_items = [c for c in candidates if _candidate_text(c) and (any(k in str(c.get("className") or "").lower() for k in ["card", "item", "tile", "option", "choice"]) or re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", _candidate_text(c) or ""))]
    card_like_items = [c for c in card_like_items if str(c.get("id") or "").lower() != "wrap" and len((_candidate_text(c) or "").splitlines()) <= 8]
    return {
        "visible_text": visible_text,
        "text_lines": text_lines,
        "numeric_values": numeric_values,
        "numeric_values_visible_text": nums,
        "numeric_values_candidates": extract_numbers_from_candidates(candidates),
        "card_like_items": card_like_items,
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
