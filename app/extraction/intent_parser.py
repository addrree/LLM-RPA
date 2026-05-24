from __future__ import annotations

import re
from typing import Any


def parse_extraction_intent(instruction: str) -> dict[str, Any]:
    text = " ".join(str(instruction or "").strip().lower().split())
    signals: list[str] = []
    intent = "unknown"

    mark_important_request = bool(re.search(r"(click the star|star icon|mark (it )?as important)", text))
    if re.search(r"\b(important|starred)\b.*\b(email|inbox|message)\b|\b(email|inbox|message)\b.*\b(important|starred)\b", text) and not mark_important_request:
        intent = "find_important_email"; signals.append("important_email")
    elif re.search(r"\b(email|inbox|message|sender|subject)\b", text):
        intent = "find_email"; signals.append("email")
    elif re.search(r"\b(calendar|meeting|event|slot|appointment)\b", text):
        intent = "find_calendar_event"; signals.append("calendar")
    elif re.search(r"\b(\d+(?:st|nd|rd|th))\s+word\b", text):
        intent = "ordinal_word_extraction"; signals.append("ordinal_word")
    elif re.search(r"\b(tree|node|parent|child|expand|collapse)\b", text):
        intent = "find_tree_node"; signals.append("tree")
    elif re.search(r"\b(odd|even|parity)\b", text):
        intent = "parity_check"; signals.append("parity")
    elif re.search(r"\b(row|column|col|cell|grid|table|coordinate)\b", text):
        intent = "grid_lookup"; signals.append("grid")
    elif re.search(r"\b(identify|classify|what shape|which shape)\b", text):
        intent = "classify_object"; signals.append("classify")
    elif re.search(r"\b(how many|count|number of visible)\b", text):
        intent = "count_objects"; signals.append("count")
    elif re.search(r"\b(midpoint|middle|median)\b", text):
        intent = "find_midpoint_or_middle_value"; signals.append("middle")
    elif re.search(r"\b(greatest|largest|highest|max(?:imum)?|cheapest|lowest|minimum|smallest)\b", text):
        intent = "find_max_numeric"; signals.append("extreme_numeric")
    elif re.search(r"\b(find|extract|locate)\b", text):
        intent = "find_text"; signals.append("find_text")

    quoted = re.findall(r'"([^"]+)"', str(instruction or ""))
    if re.search(r"\b(press|click)\b[^.]*\"[^\"]+\"", str(instruction or "").lower()):
        quoted = [q for q in quoted if q.lower() not in {"submit", "ok", "confirm"}]
    ord_match = re.search(r"\b(\d+)(st|nd|rd|th)\s+word\b", text)
    numbers = [int(n) for n in re.findall(r"\b\d+\b", text)]
    requested_email_action = "open"
    if re.search(r"\breply\b", text):
        requested_email_action = "reply"
    elif re.search(r"\bforward\b", text):
        requested_email_action = "forward"
    elif re.search(r"\b(trash|delete)\b", text):
        requested_email_action = "trash"
    elif re.search(r"\b(star|important|mark as important)\b|click the star|star icon|mark (it )?as important", text):
        requested_email_action = "star"
    sender_match = re.search(r"(?:email by|from)\s+([a-zA-Z]+)", text, flags=re.I)
    target_sender = sender_match.group(1).strip() if sender_match else None
    target_keywords = [w for w in re.findall(r"[a-z]{3,}", text) if w not in {"find", "click", "extract", "open", "with", "from", "that", "this", "there", "visible", "item", "page"}][:8]
    confidence = 0.9 if intent != "unknown" else 0.2
    return {
        "intent": intent,
        "confidence": confidence,
        "signals": signals,
        "normalized_instruction": text,
        "constraints": {
            "quoted_targets": quoted,
            "numbers": numbers,
            "keywords": target_keywords,
            "ordinal_index": int(ord_match.group(1)) if ord_match else None,
            "requested_email_action": requested_email_action,
            "reply_text": quoted[0] if quoted and requested_email_action == "reply" else None,
            "target_sender": target_sender,
        },
    }
