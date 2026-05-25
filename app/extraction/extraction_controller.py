from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


TEXT_FIELDS = ("text", "innerText", "textContent", "name", "ariaLabel", "title", "label", "value")


def _text_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("value", "text", "name", "label"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _real_candidate_bid(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("bid") or "").strip()


def _browsergym_click_action(bid: str, action_syntax: list[str] | None = None) -> str:
    _ = action_syntax
    return f'click("{bid}", "left")'


@dataclass
class ExtractionDecision:
    answer: str | None = None
    extracted_data: Any = None
    action: str | None = None
    selected_candidate: dict[str, Any] | None = None
    confidence: float = 0.0
    strategy: str = ""
    diagnostics: dict[str, Any] | None = None


def _candidate_text(c: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        text = _text_value(c.get(field))
        if text:
            return text
    return ""


def _candidate_text_from(c: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        text = _text_value(c.get(field))
        if text:
            return text
    return ""


def _candidate_haystack(c: dict[str, Any], *, include_parent: bool = False) -> str:
    fields = TEXT_FIELDS + (("id", "className", "parent_class", "parent_text") if include_parent else ("id", "className"))
    return " ".join(_text_value(c.get(field)) for field in fields).lower()


def _find_click_candidate_by_text(candidates: list[dict[str, Any]], target_text: str) -> dict[str, Any] | None:
    return find_action_candidate(candidates, target_text)


def find_action_candidate(candidates: list[dict[str, Any]], target_text: str, preferred_roles: set[str] | None = None) -> dict[str, Any] | None:
    t = str(target_text or "").strip().lower()
    preferred_roles = preferred_roles or {"button", "link", "menuitem", "option"}
    exact: list[dict[str, Any]] = []
    contains: list[dict[str, Any]] = []
    for c in candidates:
        txt = _candidate_text(c)
        low = txt.lower()
        if not t or t not in low:
            continue
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if _is_wrapper_candidate(c) or len(lines) > 3:
            continue
        if low == t:
            exact.append(c)
        elif len(txt) <= 40:
            contains.append(c)

    def score(c: dict[str, Any]) -> tuple[int, int, int]:
        tag = str(c.get("tag") or "").lower()
        role = str(c.get("role") or "").lower()
        cid_cls = f"{str(c.get('id') or '').lower()} {str(c.get('className') or '').lower()}"
        txt = _candidate_text(c)
        return (
            int(tag in {"button", "input"}),
            int(role in preferred_roles or any(k in cid_cls for k in ["button", "action", "submit", "reply", "send", "star", "trash", "delete"])),
            -len(txt),
        )

    pool = sorted(exact or contains, key=score, reverse=True)
    return pool[0] if pool else None


def _is_wrapper_candidate(c: dict[str, Any]) -> bool:
    cid = str(c.get("id") or "").lower()
    tag = str(c.get("tag") or "").lower()
    cls = str(c.get("className") or "").lower()
    text = _candidate_text(c).lower()
    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else {}
    w = float(bbox.get("width") or 0)
    h = float(bbox.get("height") or 0)
    return cid in {"wrap", "area"} or tag in {"body", "html", "main"} or "wrapper" in cls or (w > 140 and h > 180) or ("submit" in text and len(text.splitlines()) >= 3)


def _is_textbox_candidate(c: dict[str, Any]) -> bool:
    tag = str(c.get("tag") or "").lower()
    role = str(c.get("role") or "").lower()
    typ = str(c.get("type") or "").lower()
    if c.get("visible") is False or c.get("disabled") is True:
        return False
    if tag == "textarea":
        return True
    if tag == "input" and typ in {"", "text", "search", "email", "tel", "url"}:
        return True
    return role in {"textbox", "input"} and typ not in {"submit", "button", "checkbox", "radio"}


def _is_instruction_like(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in ("find the", "type that", "press", "click the", "mark it as"))


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text or "")


def _split_before_submit(text: str) -> str:
    lines = [ln.rstrip() for ln in str(text or "").splitlines()]
    for idx, line in enumerate(lines):
        if line.strip().lower() == "submit":
            return "\n".join(lines[:idx]).strip()
    return re.split(r"\bsubmit\b", str(text or ""), maxsplit=1, flags=re.I)[0].strip()


def _history_has_fill(history: list[Any], bid: str, value: str) -> bool:
    for item in history:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        if "fill(" not in action:
            continue
        if bid and bid not in action:
            continue
        if value in action:
            return True
    return False


def _candidate_center_y(c: dict[str, Any]) -> float:
    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else {}
    y = float(bbox.get("y") or 0)
    h = float(bbox.get("height") or 0)
    return y + (h / 2)


def _candidate_center_x(c: dict[str, Any]) -> float:
    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else {}
    x = float(bbox.get("x") or bbox.get("left") or 0)
    w = float(bbox.get("width") or 0)
    return x + (w / 2)


def _build_parity_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in candidates:
        txt = _candidate_text(c).strip()
        low = txt.lower()
        if low not in {"odd", "even"}:
            continue
        parent_text = str(c.get("parent_text") or "")
        parent_class = str(c.get("parent_class") or "").lower()
        cls = str(c.get("className") or "").lower()
        number_match = re.search(r"\b(\d+)\b", parent_text) or re.search(r"\b(\d+)\b", str(c.get("sibling_text") or ""))
        if parent_class != "row" and cls != "row" and not re.search(r"odd\s*\n?\s*\d+\s*\n?\s*even", parent_text.lower()):
            continue
        if not number_match:
            continue
        number = int(number_match.group(1))
        row = {
            "row_id": f"{number}-{round(_candidate_center_y(c), 1)}",
            "number": number,
            "odd_candidate": c if low == "odd" else None,
            "even_candidate": c if low == "even" else None,
            "y_center": _candidate_center_y(c),
            "parent_text": re.sub(r"\s+", " ", parent_text.strip().lower()),
        }
        rows.append(row)

    if not rows:
        y_groups: dict[int, list[dict[str, Any]]] = {}
        for c in candidates:
            if c.get("visible") is False:
                continue
            y_groups.setdefault(int(round(_candidate_center_y(c) / 8.0)), []).append(c)
        for group in y_groups.values():
            odd = next((c for c in group if _candidate_text(c).strip().lower() == "odd"), None)
            even = next((c for c in group if _candidate_text(c).strip().lower() == "even"), None)
            number_cand = next((c for c in group if re.fullmatch(r"\s*\d+\s*", _candidate_text(c) or "")), None)
            if not (odd and even and number_cand):
                continue
            number = int(_candidate_text(number_cand).strip())
            y_center = (_candidate_center_y(odd) + _candidate_center_y(even) + _candidate_center_y(number_cand)) / 3.0
            rows.append({
                "row_id": f"{number}-{round(y_center, 1)}",
                "number": number,
                "odd_candidate": odd,
                "even_candidate": even,
                "y_center": y_center,
                "parent_text": "",
            })

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["number"], int(round(row["y_center"] / 10.0)))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = row
        else:
            existing["odd_candidate"] = existing["odd_candidate"] or row["odd_candidate"]
            existing["even_candidate"] = existing["even_candidate"] or row["even_candidate"]

    dedup = [r for r in grouped.values() if r.get("odd_candidate") and r.get("even_candidate")]
    dedup.sort(key=lambda r: (r["y_center"], r["number"]))
    return dedup


def _email_zone(c: dict[str, Any]) -> str:
    return " ".join([
        str(c.get("className") or ""),
        str(c.get("id") or ""),
        str(c.get("role") or ""),
        str(c.get("parent_class") or ""),
        str(c.get("parent_tag") or ""),
    ]).lower()


def _is_opened_email_content(c: dict[str, Any]) -> bool:
    zone = _email_zone(c)
    return any(k in zone for k in ("email-header", "email-body", "email-left", "email-right", "email-content", "email-sender"))


def _is_email_thread_candidate(c: dict[str, Any]) -> bool:
    zone = _email_zone(c)
    if _is_opened_email_content(c):
        return False
    txt = _candidate_text(c)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if "email-thread" in zone:
        return bool(lines)
    return str(c.get("role") or "").lower() == "row" and bool(lines)


def _email_sender_from_candidate(c: dict[str, Any]) -> str:
    txt = _candidate_text(c)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines:
        from_match = re.search(r"from:\s*([A-Za-z]+)", lines[0], flags=re.I)
        return from_match.group(1) if from_match else lines[0]
    from_match = re.search(r"from:\s*([A-Za-z]+)", txt, flags=re.I)
    return from_match.group(1) if from_match else txt


def _is_same_email_row_area(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    row_bid = _real_candidate_bid(row)
    if row_bid and str(candidate.get("parent_bid") or "") == row_bid:
        return True
    row_text = _candidate_text(row)
    sender = _email_sender_from_candidate(row).lower()
    parent_text = _text_value(candidate.get("parent_text")).lower()
    if sender and sender in parent_text and _candidate_text(row).splitlines()[0].lower() in parent_text:
        return True
    row_bbox = row.get("bbox") if isinstance(row.get("bbox"), dict) else {}
    cand_bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    if row_bbox and cand_bbox:
        row_top = float(row_bbox.get("top", row_bbox.get("y", 0)) or 0)
        row_bottom = float(row_bbox.get("bottom", row_top + float(row_bbox.get("height") or 0)) or 0)
        cy = _candidate_center_y(candidate)
        if row_top <= cy <= row_bottom:
            return True
    return bool(row_text and row_text in parent_text)


def _email_action_haystack(c: dict[str, Any]) -> str:
    return " ".join(
        _text_value(c.get(field))
        for field in ("text", "innerText", "textContent", "name", "ariaLabel", "title", "id", "className", "parent_class", "role")
    ).lower()


def _has_action_marker(hay: str, marker: str) -> bool:
    return bool(re.search(rf"(^|[^a-z0-9]){re.escape(marker)}([^a-z0-9]|$)", hay))


def _is_email_action_candidate(c: dict[str, Any], keys: list[str]) -> bool:
    if not _real_candidate_bid(c) or c.get("visible") is False or c.get("disabled") is True:
        return False
    if _is_textbox_candidate(c):
        return False
    if _is_wrapper_candidate(c):
        return False
    tag = str(c.get("tag") or "").lower()
    role = str(c.get("role") or "").lower()
    text = _candidate_text(c).strip().lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cid = str(c.get("id") or "").lower()
    cls = str(c.get("className") or "").lower()
    parent_cls = str(c.get("parent_class") or "").lower()
    hay = _email_action_haystack(c)
    keyset = set(keys)
    exact_text = text in keyset and len(text) <= 20
    if cid.startswith("close-") and "close" not in keyset:
        return False
    if tag in {"div", "section", "form"} and cid in {"reply", "forward", "email", "main", "area", "wrap"}:
        return False
    if tag in {"div", "section", "form"} and not exact_text and (len(lines) > 2 or len(text) > 80):
        return False
    if not exact_text and re.search(r"\b(to|subject):", text):
        return False
    keyed = any(k in hay for k in keys)
    if not keyed and not exact_text:
        return False
    if exact_text:
        return True
    if "reply" in keyset:
        return (
            "email-reply" in cls
            or "email-reply" in parent_cls
            or cid in {"reply-button", "reply-btn"}
            or (tag in {"button", "a", "span"} and "reply" in hay and cid != "reply")
            or role in {"button", "link", "menuitem"} and "reply" in hay
        )
    if "forward" in keyset:
        return (
            "email-forward" in cls
            or "email-forward" in parent_cls
            or cid in {"forward-button", "forward-btn"}
            or (tag in {"button", "a", "span"} and "forward" in hay and cid != "forward")
            or role in {"button", "link", "menuitem"} and "forward" in hay
        )
    if "send" in keyset or "submit" in keyset:
        sendish = _has_action_marker(hay, "send") or _has_action_marker(hay, "submit")
        return (
            cid.startswith("send")
            or cid in {"submit", "subbtn"}
            or tag in {"button", "input", "a", "span"} and sendish
            or role in {"button", "link", "menuitem"} and sendish
        )
    if "star" in keyset or "important" in keyset:
        return (
            "star" in cid
            or "star" in cls
            or "important" in cid
            or "important" in cls
            or "email-actions" in parent_cls and tag in {"span", "button", "a"}
            or role in {"button", "link", "menuitem"} and any(k in hay for k in ("star", "important"))
        )
    if "trash" in keyset or "delete" in keyset:
        return (
            "trash" in cid
            or "trash" in cls
            or "delete" in cid
            or "delete" in cls
            or role in {"button", "link", "menuitem"} and any(k in hay for k in ("trash", "delete"))
        )
    return tag in {"button", "input", "a", "span"} or role in {"button", "link", "menuitem"}


def _find_email_action(candidates: list[dict[str, Any]], keys: list[str]) -> dict[str, Any] | None:
    matches = [c for c in candidates if _is_email_action_candidate(c, keys)]
    if not matches:
        return None

    def score(c: dict[str, Any]) -> tuple[int, int, int]:
        tag = str(c.get("tag") or "").lower()
        role = str(c.get("role") or "").lower()
        text = _candidate_text(c).strip().lower()
        hay = _email_action_haystack(c)
        return (
            int(text in set(keys)),
            int(tag in {"button", "input", "span"} or role == "button"),
            int(any(k in hay for k in ("icon", "action", "toolbar", "star", "reply", "send", "trash", "delete"))),
        )

    return sorted(matches, key=score, reverse=True)[0]


def _find_inline_email_star(candidates: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    matches = []
    for c in candidates:
        if not _is_email_action_candidate(c, ["star", "important"]):
            continue
        if _is_same_email_row_area(row, c):
            matches.append(c)
    if not matches:
        return None
    matches.sort(key=lambda c: (_candidate_center_x(c), -len(_candidate_text(c))), reverse=True)
    return matches[0]


def _find_forward_recipient_textbox(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = []
    for c in candidates:
        if not _is_textbox_candidate(c) or not _real_candidate_bid(c):
            continue
        tag = str(c.get("tag") or "").lower()
        hay = _email_action_haystack(c)
        if tag == "input" and any(k in hay for k in ("forward-sender", "recipient", "to:")):
            matches.append(c)
    if matches:
        matches.sort(key=lambda c: (int("forward-sender" in _email_action_haystack(c)), -len(_candidate_text(c))), reverse=True)
        return matches[0]
    return None


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _find_tree_expand_candidate(candidates: list[dict[str, Any]], target: str, history: list[Any]) -> dict[str, Any] | None:
    target_compact = _compact_text(target)
    if not target_compact:
        return None
    clicked_bids = {str(h.get("selected_candidate_bid") or "") for h in history if isinstance(h, dict)}
    matches = []
    for c in candidates:
        bid = _real_candidate_bid(c)
        if not bid or bid in clicked_bids or c.get("visible") is False or _is_wrapper_candidate(c):
            continue
        visible_text = _candidate_text(c).strip()
        if visible_text.lower() == str(target or "").strip().lower():
            continue
        searchable = " ".join(_text_value(c.get(field)) for field in ("textContent", "innerText", "text", "parent_text"))
        if target_compact not in _compact_text(searchable):
            continue
        cls = str(c.get("className") or "").lower()
        parent_cls = str(c.get("parent_class") or "").lower()
        tag = str(c.get("tag") or "").lower()
        if not (tag in {"li", "span", "div", "a"} or any(k in f"{cls} {parent_cls}" for k in ("tree", "folder", "file", "expandable", "collapsable"))):
            continue
        matches.append(c)
    if not matches:
        return None

    def score(c: dict[str, Any]) -> tuple[int, int, int]:
        cls = str(c.get("className") or "").lower()
        text_blob = " ".join(_text_value(c.get(field)) for field in ("textContent", "innerText", "text"))
        return (
            int(any(k in cls for k in ("expandable", "folder", "tree"))),
            int(str(c.get("tag") or "").lower() == "li"),
            -len(text_blob),
        )

    return sorted(matches, key=score, reverse=True)[0]


def solve_extraction_task(intent: dict[str, Any] | str, extraction_context: dict[str, Any], candidates: list[dict[str, Any]], action_syntax: list[str] | None = None) -> ExtractionDecision | None:
    intent_name = intent.get("intent") if isinstance(intent, dict) else str(intent)
    constraints = intent.get("constraints", {}) if isinstance(intent, dict) else {}
    keywords = [str(k).lower() for k in (constraints.get("keywords") or [])]
    candidates = list(candidates or [])
    nums = [n for n in extraction_context.get("numeric_values", []) if isinstance(n, dict) and n.get("value") is not None]
    history = constraints.get("history") or []

    if intent_name == "ordinal_word_extraction":
        ord_idx = constraints.get("ordinal_index")
        if not ord_idx:
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "ordinal_not_found"})
        ord_idx = int(ord_idx)

        paragraph_candidates: list[tuple[dict[str, Any], str, list[str]]] = []
        for c in candidates:
            tag = str(c.get("tag") or "").lower()
            role = str(c.get("role") or "").lower()
            ctext = _candidate_text_from(c, ("text", "innerText", "textContent"))
            low = ctext.lower()
            if not ctext or low == "submit" or _is_instruction_like(ctext) or "submit" in low:
                continue
            if role in {"button", "textbox", "input"} or tag in {"input", "textarea", "button"}:
                continue
            words = _word_tokens(ctext)
            if len(words) >= ord_idx and tag in {"p", "div", "span", "td", "li"} and not _is_wrapper_candidate(c):
                paragraph_candidates.append((c, ctext, words))

        paragraph_candidate_bid = None
        paragraph_text = ""
        words: list[str] = []
        if paragraph_candidates:
            def paragraph_score(item: tuple[dict[str, Any], str, list[str]]) -> tuple[int, int, int]:
                cand, text, tokens = item
                tag = str(cand.get("tag") or "").lower()
                return (
                    int(tag == "p"),
                    int(tag in {"div", "span"}),
                    -abs(len(tokens) - ord_idx),
                )

            paragraph_candidates.sort(key=paragraph_score, reverse=True)
            pc, paragraph_text, words = paragraph_candidates[0]
            paragraph_candidate_bid = _real_candidate_bid(pc)
        else:
            fallback_candidates: list[tuple[dict[str, Any], str, list[str]]] = []
            for c in candidates:
                tag = str(c.get("tag") or "").lower()
                role = str(c.get("role") or "").lower()
                if role == "button" or tag == "button":
                    continue
                for field in ("text", "innerText", "textContent", "parent_text"):
                    txt = _text_value(c.get(field))
                    if not txt or _is_instruction_like(txt):
                        continue
                    split = _split_before_submit(txt) if "submit" in txt.lower() else txt.strip()
                    split_words = _word_tokens(split)
                    if len(split_words) >= ord_idx:
                        fallback_candidates.append((c, split, split_words))
            if fallback_candidates:
                fallback_candidates.sort(key=lambda x: (int(str(x[0].get("tag") or "").lower() in {"p", "div", "span"}), -abs(len(x[2]) - ord_idx)), reverse=True)
                pc, paragraph_text, words = fallback_candidates[0]
                paragraph_candidate_bid = _real_candidate_bid(pc)

        if len(words) < ord_idx:
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "ordinal_out_of_range", "ordinal": ord_idx})
        ans = words[ord_idx - 1]

        textbox = next((c for c in candidates if _is_textbox_candidate(c) and _real_candidate_bid(c)), None)
        textbox_bid = _real_candidate_bid(textbox)
        fill_already_done = _history_has_fill(history, textbox_bid, ans)
        if textbox and textbox_bid and not fill_already_done:
            return ExtractionDecision(answer=ans, action=f'fill("{textbox_bid}", "{ans}")', selected_candidate=textbox, confidence=0.9, strategy="ordinal_word_fill", diagnostics={"ordinal": ord_idx, "paragraph_text": paragraph_text, "words_count": len(words), "answer": ans, "textbox_bid": textbox_bid, "submit_bid": "", "paragraph_candidate_bid": paragraph_candidate_bid, "fill_already_done": False})

        submit = find_action_candidate(candidates, "submit")
        if submit and _real_candidate_bid(submit):
            return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="ordinal_word_submit", diagnostics={"ordinal": ord_idx, "paragraph_text": paragraph_text, "words_count": len(words), "answer": ans, "textbox_bid": textbox_bid, "submit_bid": _real_candidate_bid(submit), "paragraph_candidate_bid": paragraph_candidate_bid, "fill_already_done": True})
        return ExtractionDecision(answer=ans, strategy="no_decision", diagnostics={"reason": "submit_not_found"})

    if intent_name == "find_max_numeric" and nums:
        mx = max(nums, key=lambda n: float(n.get("value", 0)))
        ans = str(int(mx["value"])) if float(mx["value"]).is_integer() else str(mx["value"])
        num_cands = [c for c in candidates if re.fullmatch(rf"\s*{re.escape(ans)}\s*", str(c.get("text") or c.get("innerText") or ""))]
        precise = [c for c in num_cands if not _is_wrapper_candidate(c)]
        cand = precise[0] if precise else None
        if cand is None:
            return ExtractionDecision(answer=ans, strategy="no_decision", diagnostics={"reason": "precise_numeric_candidate_not_found"})
        max_action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else ""
        max_clicked = any(str(h.get("selected_candidate_bid") or "") == _real_candidate_bid(cand) or str(h.get("action") or "").strip() == max_action for h in history if isinstance(h, dict))
        if max_clicked:
            submit = find_action_candidate(candidates, "submit")
            if submit and _real_candidate_bid(submit):
                return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="max_numeric_submit")
        return ExtractionDecision(answer=ans, action=max_action, selected_candidate=cand, confidence=0.9, strategy="max_numeric_from_visible_text")

    if intent_name == "parity_check":
        rows = _build_parity_rows(candidates)
        clicked_bids = {str(h.get("selected_candidate_bid") or "") for h in history if isinstance(h, dict)}
        unanswered = [r for r in rows if _real_candidate_bid(r["odd_candidate"]) not in clicked_bids and _real_candidate_bid(r["even_candidate"]) not in clicked_bids]
        if not rows:
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "rows_not_found"})
        if not unanswered:
            submit = find_action_candidate(candidates, "submit")
            if submit and _real_candidate_bid(submit):
                return ExtractionDecision(answer="submit", action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.8, strategy="parity_submit", diagnostics={"rows_total": len(rows), "answered_rows": len(rows), "submit_ready": True})
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "submit_not_found", "rows_total": len(rows)})
        row = unanswered[0]
        ans = "even" if row["number"] % 2 == 0 else "odd"
        cand = row["even_candidate"] if ans == "even" else row["odd_candidate"]
        return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(cand)), selected_candidate=cand, confidence=0.8, strategy="parity_rowwise", diagnostics={"rows_total": len(rows), "answered_rows": len(rows) - len(unanswered), "current_row_number": row["number"], "selected_parity": ans})

    if intent_name in {"find_email", "find_important_email"}:
        emails = []
        for item in extraction_context.get("email_like_items") or []:
            if not isinstance(item, dict):
                continue
            cand = item.get("candidate")
            if isinstance(cand, dict) and _is_email_thread_candidate(cand):
                emails.append(item)
        if not emails:
            for c in candidates:
                if not _is_email_thread_candidate(c):
                    continue
                sender = _email_sender_from_candidate(c)
                emails.append({"sender": sender, "candidate": c, "text": _candidate_text(c), "important": False})
        target_sender = str(constraints.get("target_sender") or "").strip().lower()
        requested_email_action = str(constraints.get("requested_email_action") or "open").lower()
        reply_text = constraints.get("reply_text")
        forward_to = constraints.get("forward_to")
        opened_email_detected = any(_is_opened_email_content(c) for c in candidates)

        row_match = next((e for e in emails if (not target_sender or target_sender == str(e.get("sender") or "").strip().lower() or target_sender in str(e.get("sender") or "").strip().lower())), None)
        if row_match is None and intent_name == "find_important_email":
            row_match = next((e for e in emails if bool(e.get("important"))), None)
        row_cand = row_match.get("candidate") if isinstance(row_match, dict) else None
        row_opened = any(str(h.get("selected_candidate_bid") or "") == _real_candidate_bid(row_cand) for h in history if isinstance(h, dict)) if row_cand else False

        if requested_email_action == "star" and row_cand and not row_opened:
            star_inline = _find_inline_email_star(candidates, row_cand)
            if star_inline:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(star_inline)), selected_candidate=star_inline, strategy="email_star_click", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "forward_to": forward_to, "opened_email_detected": opened_email_detected, "matched_row_bid": _real_candidate_bid(row_cand), "star_button_bid": _real_candidate_bid(star_inline)})

        if row_cand and not row_opened:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(row_cand)), selected_candidate=row_cand, strategy="email_open_row", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "forward_to": forward_to, "opened_email_detected": opened_email_detected, "matched_row_bid": _real_candidate_bid(row_cand)})

        reply_button = _find_email_action(candidates, ["reply"])
        forward_button = _find_email_action(candidates, ["forward"])
        send_button = _find_email_action(candidates, ["send", "submit"])
        delete_button = _find_email_action(candidates, ["trash", "delete"])
        star_button = _find_email_action(candidates, ["star", "important"])
        reply_textbox = next((c for c in candidates if _is_textbox_candidate(c) and not _is_wrapper_candidate(c) and _real_candidate_bid(c)), None)
        forward_textbox = _find_forward_recipient_textbox(candidates)
        reply_filled = bool(reply_text) and _history_has_fill(history, _real_candidate_bid(reply_textbox), str(reply_text))
        forward_filled = bool(forward_to) and _history_has_fill(history, _real_candidate_bid(forward_textbox), str(forward_to))

        base_diag = {
            "target_sender": target_sender,
            "requested_email_action": requested_email_action,
            "reply_text": reply_text,
            "forward_to": forward_to,
            "opened_email_detected": opened_email_detected,
            "matched_row_bid": _real_candidate_bid(row_cand),
            "reply_button_bid": _real_candidate_bid(reply_button),
            "forward_button_bid": _real_candidate_bid(forward_button),
            "reply_textbox_bid": _real_candidate_bid(reply_textbox),
            "forward_textbox_bid": _real_candidate_bid(forward_textbox),
            "send_button_bid": _real_candidate_bid(send_button),
            "delete_button_bid": _real_candidate_bid(delete_button),
            "star_button_bid": _real_candidate_bid(star_button),
        }

        if requested_email_action == "reply":
            if reply_button and not reply_textbox:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(reply_button)), selected_candidate=reply_button, strategy="email_reply_click", diagnostics=base_diag)
            if reply_textbox and reply_text and not reply_filled:
                return ExtractionDecision(action=f'fill("{_real_candidate_bid(reply_textbox)}", "{reply_text}")', selected_candidate=reply_textbox, strategy="email_reply_fill", diagnostics=base_diag)
            if reply_filled and send_button:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(send_button)), selected_candidate=send_button, strategy="email_reply_send", diagnostics=base_diag)

        if requested_email_action == "forward":
            if forward_button and not forward_textbox:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(forward_button)), selected_candidate=forward_button, strategy="email_forward_click", diagnostics=base_diag)
            if forward_textbox and forward_to and not forward_filled:
                return ExtractionDecision(action=f'fill("{_real_candidate_bid(forward_textbox)}", "{forward_to}")', selected_candidate=forward_textbox, strategy="email_forward_fill", diagnostics=base_diag)
            if forward_filled and send_button:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(send_button)), selected_candidate=send_button, strategy="email_forward_send", diagnostics=base_diag)

        if requested_email_action == "trash" and delete_button:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(delete_button)), selected_candidate=delete_button, strategy="email_delete_click", diagnostics=base_diag)
        if requested_email_action == "star" and star_button:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(star_button)), selected_candidate=star_button, strategy="email_star_click", diagnostics=base_diag)
        if requested_email_action == "star":
            return ExtractionDecision(strategy="no_decision", diagnostics={**base_diag, "reason": "star_icon_not_found"})

        return ExtractionDecision(strategy="no_decision", diagnostics={**base_diag, "reason": "email_action_not_found"})

    if intent_name == "find_tree_node":
        trees = extraction_context.get("tree_like_items") or []
        targets = [str(t).strip().lower() for t in (constraints.get("quoted_targets") or [])]
        target = targets[0] if targets else ""
        if trees:
            selected = trees[0] if not targets else next((t for t in trees if str(t.get("node_text") or "").strip().lower() in targets), None)
            if selected is None:
                expand = _find_tree_expand_candidate(candidates, target, history)
                if expand and _real_candidate_bid(expand):
                    return ExtractionDecision(answer=target, action=_browsergym_click_action(_real_candidate_bid(expand)), selected_candidate=expand, strategy="tree_expand_toward_target", confidence=0.55, diagnostics={"target": target, "expand_bid": _real_candidate_bid(expand), "reason": "target_tree_node_not_visible_yet"})
                return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible", "target": target})
            cand = selected.get("candidate")
            if isinstance(cand, dict) and _real_candidate_bid(cand):
                return ExtractionDecision(answer=selected.get("node_text"), action=_browsergym_click_action(_real_candidate_bid(cand)), selected_candidate=cand, strategy="tree_node_click", confidence=0.6)
        direct = None
        if targets:
            direct = next((c for c in candidates if _candidate_text(c).strip().lower() in targets and _real_candidate_bid(c) and not _is_wrapper_candidate(c)), None)
        if direct:
            return ExtractionDecision(answer=_candidate_text(direct), action=_browsergym_click_action(_real_candidate_bid(direct)), selected_candidate=direct, strategy="tree_node_click", confidence=0.6)
        expand = _find_tree_expand_candidate(candidates, target, history)
        if expand and _real_candidate_bid(expand):
            return ExtractionDecision(answer=target, action=_browsergym_click_action(_real_candidate_bid(expand)), selected_candidate=expand, strategy="tree_expand_toward_target", confidence=0.55, diagnostics={"target": target, "expand_bid": _real_candidate_bid(expand), "reason": "target_tree_node_not_visible_yet"})
        return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible", "target": target})

    if intent_name in {"count_objects", "grid_lookup", "find_text", "find_calendar_event", "classify_object", "find_midpoint_or_middle_value"}:
        return ExtractionDecision(answer=None, extracted_data=None, action=None, selected_candidate=None, confidence=0.0, strategy="no_decision", diagnostics={"reason": "no generic extraction decision"})
    return None
