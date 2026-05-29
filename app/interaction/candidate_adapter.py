from __future__ import annotations

from typing import Any


def normalize_candidate_for_extraction(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a BrowserGym/extraction-controller compatible candidate shape."""

    item = dict(candidate or {})
    candidate_id = item.get("candidate_id") or item.get("bid") or item.get("id")
    class_name = item.get("className") or item.get("class_name") or item.get("class") or ""
    inner_text = item.get("innerText", item.get("inner_text", item.get("text", "")))
    text_content = item.get("textContent", item.get("text_content", item.get("text", "")))
    aria_label = item.get("ariaLabel", item.get("aria_label", ""))
    bbox = item.get("bbox") or {}

    normalized = {
        **item,
        "bid": item.get("bid") or candidate_id,
        "candidate_id": candidate_id,
        "text": item.get("text") or inner_text or text_content or item.get("name") or aria_label or "",
        "innerText": inner_text or "",
        "inner_text": inner_text or "",
        "textContent": text_content or "",
        "text_content": text_content or "",
        "ariaLabel": aria_label or "",
        "aria_label": aria_label or "",
        "className": class_name,
        "class": class_name,
        "id": item.get("id", ""),
        "role": item.get("role", ""),
        "tag": item.get("tag", ""),
        "href": item.get("href", ""),
        "selector": item.get("selector") or item.get("css_path") or "",
        "bbox": bbox,
        "visible": item.get("visible", True),
    }
    return normalized


def normalize_candidates_for_extraction(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_candidate_for_extraction(candidate) for candidate in candidates or []]


def compact_text_lines(text: str, *, limit: int = 250) -> list[str]:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def split_candidate_groups(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    normalized = normalize_candidates_for_extraction(candidates)
    buttons = []
    links = []
    inputs = []
    for candidate in normalized:
        kind = str(candidate.get("kind") or "").lower()
        tag = str(candidate.get("tag") or "").lower()
        role = str(candidate.get("role") or "").lower()
        input_type = str(candidate.get("input_type") or candidate.get("type") or "").lower()
        if kind == "link" or tag == "a" or role == "link":
            links.append(candidate)
        if kind == "button" or tag == "button" or role == "button" or input_type in {"button", "submit"}:
            buttons.append(candidate)
        if kind in {"textbox", "select", "checkbox", "radio", "date"} or tag in {"input", "textarea", "select"}:
            inputs.append(candidate)
    return {"candidates": normalized, "buttons": buttons, "links": links, "inputs": inputs}
