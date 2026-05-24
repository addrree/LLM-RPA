from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


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
    return str(c.get("text") or c.get("innerText") or c.get("textContent") or c.get("name") or c.get("ariaLabel") or c.get("title") or "").strip()


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


def _candidate_center_y(c: dict[str, Any]) -> float:
    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else {}
    y = float(bbox.get("y") or 0)
    h = float(bbox.get("height") or 0)
    return y + (h / 2)


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
            ctext = _candidate_text(c) or str(c.get("parent_text") or "").strip()
            low = ctext.lower()
            if not ctext or low == "submit" or any(x in low for x in ["find the", "type that", "press"]):
                continue
            if role in {"button", "textbox", "input"} or tag in {"input", "textarea", "button"}:
                continue
            words = re.findall(r"[A-Za-z0-9]+", ctext)
            if len(words) >= ord_idx and not _is_wrapper_candidate(c):
                paragraph_candidates.append((c, ctext, words))

        paragraph_candidate_bid = None
        paragraph_text = ""
        words: list[str] = []
        if paragraph_candidates:
            paragraph_candidates.sort(key=lambda x: (len(x[2]), len(x[1])), reverse=True)
            pc, paragraph_text, words = paragraph_candidates[0]
            paragraph_candidate_bid = _real_candidate_bid(pc)
        else:
            for c in candidates:
                txt = _candidate_text(c)
                if "submit" in txt.lower() and _is_wrapper_candidate(c):
                    split = re.split(r"\bsubmit\b", txt, flags=re.I)[0].strip()
                    split_words = re.findall(r"[A-Za-z0-9]+", split)
                    if len(split_words) >= ord_idx:
                        paragraph_text, words = split, split_words
                        break

        if len(words) < ord_idx:
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "ordinal_out_of_range", "ordinal": ord_idx})
        ans = words[ord_idx - 1]

        textbox = next((c for c in candidates if (str(c.get("tag") or "").lower() in {"input", "textarea"} or str(c.get("role") or "").lower() in {"textbox", "input"}) and "submit" not in _candidate_text(c).lower()), None)
        fill_already_done = any("fill(" in str(h.get("action") or "") and ans in str(h.get("action") or "") for h in history if isinstance(h, dict))
        if textbox and _real_candidate_bid(textbox) and not fill_already_done:
            return ExtractionDecision(answer=ans, action=f'fill("{_real_candidate_bid(textbox)}", "{ans}")', selected_candidate=textbox, confidence=0.9, strategy="ordinal_word_fill", diagnostics={"ordinal": ord_idx, "paragraph_text": paragraph_text, "words_count": len(words), "answer": ans, "textbox_bid": _real_candidate_bid(textbox), "paragraph_candidate_bid": paragraph_candidate_bid, "fill_already_done": False})

        submit = find_action_candidate(candidates, "submit")
        if submit and _real_candidate_bid(submit):
            return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="ordinal_word_submit", diagnostics={"ordinal": ord_idx, "paragraph_text": paragraph_text, "words_count": len(words), "answer": ans, "submit_bid": _real_candidate_bid(submit), "paragraph_candidate_bid": paragraph_candidate_bid, "fill_already_done": True})
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
        emails = extraction_context.get("email_like_items") or []
        if not emails:
            emails = []
            for c in candidates:
                txt = _candidate_text(c)
                if not (str(c.get("role") or "").lower() == "row" or "from:" in txt.lower()):
                    continue
                m = re.search(r"from:\\s*([a-zA-Z]+)", txt, flags=re.I)
                sender = m.group(1) if m else txt
                emails.append({"sender": sender, "candidate": c, "text": txt})
        target_sender = str(constraints.get("target_sender") or "").strip().lower()
        requested_email_action = str(constraints.get("requested_email_action") or "open").lower()
        reply_text = constraints.get("reply_text")
        opened_email_detected = any(any(k in (str(c.get("className") or "").lower() + " " + str(c.get("id") or "").lower()) for k in ["email-header", "email-body", "email-left"]) for c in candidates)

        row_match = next((e for e in emails if (not target_sender or target_sender in str(e.get("sender") or "").strip().lower())), None)
        row_cand = row_match.get("candidate") if isinstance(row_match, dict) else None
        row_opened = any(str(h.get("selected_candidate_bid") or "") == _real_candidate_bid(row_cand) for h in history if isinstance(h, dict)) if row_cand else False

        if requested_email_action == "star" and row_cand and not row_opened:
            star_inline = next((c for c in candidates if _real_candidate_bid(c) and any(k in (str(c.get("className") or "").lower() + " " + _candidate_text(c).lower() + " " + str(c.get("ariaLabel") or "").lower()) for k in ["star", "important"]) and str(c.get("parent_bid") or "") == _real_candidate_bid(row_cand)), None)
            if star_inline:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(star_inline)), selected_candidate=star_inline, strategy="email_star_click", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action})

        if row_cand and not row_opened:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(row_cand)), selected_candidate=row_cand, strategy="email_open_row", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "opened_email_detected": opened_email_detected})

        def _find_email_action(keys: list[str]) -> dict[str, Any] | None:
            for c in candidates:
                hay = " ".join([_candidate_text(c), str(c.get("ariaLabel") or ""), str(c.get("title") or ""), str(c.get("className") or ""), str(c.get("id") or "")]).lower()
                if any(k in hay for k in keys) and not _is_wrapper_candidate(c):
                    return c
            return None

        reply_button = _find_email_action(["reply"])
        send_button = _find_email_action(["send", "submit"])
        delete_button = _find_email_action(["trash", "delete"])
        star_button = _find_email_action(["star", "important"])
        reply_textbox = next((c for c in candidates if (str(c.get("tag") or "").lower() in {"input", "textarea"} or str(c.get("role") or "").lower() in {"textbox", "input"}) and not _is_wrapper_candidate(c)), None)
        reply_filled = bool(reply_text) and any("fill(" in str(h.get("action") or "") and str(reply_text) in str(h.get("action") or "") for h in history if isinstance(h, dict))

        if requested_email_action == "reply":
            if reply_button and not reply_textbox:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(reply_button)), selected_candidate=reply_button, strategy="email_reply_click", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "opened_email_detected": True})
            if reply_textbox and reply_text and not reply_filled:
                return ExtractionDecision(action=f'fill("{_real_candidate_bid(reply_textbox)}", "{reply_text}")', selected_candidate=reply_textbox, strategy="email_reply_fill", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "opened_email_detected": True})
            if reply_filled and send_button:
                return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(send_button)), selected_candidate=send_button, strategy="email_reply_send", diagnostics={"target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "opened_email_detected": True})

        if requested_email_action == "trash" and delete_button:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(delete_button)), selected_candidate=delete_button, strategy="email_delete_click")
        if requested_email_action == "star" and star_button:
            return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(star_button)), selected_candidate=star_button, strategy="email_star_click")
        if requested_email_action == "star":
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "star_icon_not_found", "target_sender": target_sender})

        return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "email_action_not_found", "target_sender": target_sender, "requested_email_action": requested_email_action, "reply_text": reply_text, "opened_email_detected": opened_email_detected, "reply_button_bid": _real_candidate_bid(reply_button), "reply_textbox_bid": _real_candidate_bid(reply_textbox), "send_button_bid": _real_candidate_bid(send_button), "delete_button_bid": _real_candidate_bid(delete_button), "star_button_bid": _real_candidate_bid(star_button)})

    if intent_name == "find_tree_node":
        trees = extraction_context.get("tree_like_items") or []
        targets = [str(t).strip().lower() for t in (constraints.get("quoted_targets") or [])]
        if trees:
            selected = trees[0] if not targets else next((t for t in trees if str(t.get("node_text") or "").strip().lower() in targets), None)
            if selected is None:
                return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible"})
            cand = selected.get("candidate")
            if isinstance(cand, dict) and _real_candidate_bid(cand):
                return ExtractionDecision(answer=selected.get("node_text"), action=_browsergym_click_action(_real_candidate_bid(cand)), selected_candidate=cand, strategy="tree_node_click", confidence=0.6)
        return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible"})

    if intent_name in {"count_objects", "grid_lookup", "find_text", "find_calendar_event", "classify_object", "find_midpoint_or_middle_value"}:
        return ExtractionDecision(answer=None, extracted_data=None, action=None, selected_candidate=None, confidence=0.0, strategy="no_decision", diagnostics={"reason": "no generic extraction decision"})
    return None
