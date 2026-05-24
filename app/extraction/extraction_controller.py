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
    return str(c.get("text") or c.get("innerText") or c.get("name") or c.get("ariaLabel") or c.get("title") or "").strip()


def find_action_candidate(candidates: list[dict[str, Any]], target_text: str, preferred_roles: set[str] | None = None) -> dict[str, Any] | None:
    t = str(target_text or "").strip().lower()
    preferred_roles = preferred_roles or {"button", "link", "menuitem", "option"}
    exact: list[dict[str, Any]] = []
    contains: list[dict[str, Any]] = []
    rejected = []
    for c in candidates:
        txt = _candidate_text(c)
        low = txt.lower()
        if not t or t not in low:
            continue
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if _is_wrapper_candidate(c) or len(lines) > 3:
            rejected.append(c)
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
            int(role in preferred_roles or any(k in cid_cls for k in ["button", "action", "submit"])),
            -len(txt),
        )
    pool = sorted(exact or contains, key=score, reverse=True)
    return pool[0] if pool else None


def _is_wrapper_candidate(c: dict[str, Any]) -> bool:
    cid = str(c.get("id") or "").lower()
    tag = str(c.get("tag") or "").lower()
    cls = str(c.get("className") or "").lower()
    text = str(c.get("text") or c.get("innerText") or "").lower()
    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else {}
    w = float(bbox.get("width") or 0)
    h = float(bbox.get("height") or 0)
    return cid in {"wrap", "area"} or tag in {"body", "html", "main"} or "wrapper" in cls or (w > 140 and h > 180) or ("submit" in text and len(text.splitlines()) >= 3)


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
        paragraph_cands = []
        for c in candidates:
            txt = _candidate_text(c)
            low = txt.lower()
            tag = str(c.get("tag") or "").lower()
            role = str(c.get("role") or "").lower()
            if not txt or low == "submit" or _is_wrapper_candidate(c):
                continue
            if tag in {"input", "textarea", "button"} or role in {"button", "textbox", "input"}:
                continue
            paragraph_cands.append(c)
        paragraph = _candidate_text(paragraph_cands[0]) if paragraph_cands else ""
        if not paragraph:
            for ln in extraction_context.get("text_lines") or []:
                low = ln.lower()
                if any(k in low for k in ["submit", "textbox", "press", "type that into", "find the"]):
                    continue
                paragraph = f"{paragraph} {ln}".strip()
        words = [w for w in re.findall(r"[A-Za-z0-9]+", paragraph)]
        if len(words) < int(ord_idx):
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "ordinal_out_of_range"})
        ans = words[int(ord_idx)-1].lower()
        textbox = next((c for c in candidates if str(c.get("role") or "").lower() in {"textbox", "input"} or str(c.get("tag") or "").lower() in {"input", "textarea"}), None)
        submitted = any("fill(" in str(h.get("action") or "") and ans in str(h.get("action") or "").lower() for h in history if isinstance(h, dict))
        if not submitted and textbox and _real_candidate_bid(textbox):
            return ExtractionDecision(answer=ans, action=f'fill("{_real_candidate_bid(textbox)}", "{ans}")', selected_candidate=textbox, confidence=0.9, strategy="ordinal_word_fill", diagnostics={"ordinal": ord_idx, "paragraph_text": paragraph, "words": words, "answer": ans, "textbox_bid": _real_candidate_bid(textbox)})
        submit = find_action_candidate(candidates, "submit")
        if submit and _real_candidate_bid(submit):
            return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="ordinal_word_submit", diagnostics={"ordinal": ord_idx, "paragraph_candidate_bid": _real_candidate_bid(paragraph_cands[0]) if paragraph_cands else None, "paragraph_text": paragraph, "words_count": len(words), "answer": ans, "submit_bid": _real_candidate_bid(submit), "fill_already_done": submitted})
        return ExtractionDecision(answer=ans, strategy="no_decision", diagnostics={"reason": "submit_not_found"})

    if intent_name == "find_max_numeric" and nums:
        mx = max(nums, key=lambda n: float(n.get("value", 0)))
        ans = str(int(mx["value"])) if float(mx["value"]).is_integer() else str(mx["value"])
        num_cands = [c for c in candidates if re.fullmatch(rf"\s*{re.escape(ans)}\s*", str(c.get("text") or c.get("innerText") or ""))]
        precise = [c for c in num_cands if not _is_wrapper_candidate(c)]
        cand = precise[0] if precise else None
        if cand is None:
            return ExtractionDecision(answer=ans, strategy="no_decision", diagnostics={"reason": "precise_numeric_candidate_not_found", "max_value": mx.get("value"), "numeric_candidates_count": len(num_cands), "wrapper_numeric_ignored_count": len([c for c in num_cands if _is_wrapper_candidate(c)])})
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        max_action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else ""
        max_clicked = any(str(h.get("selected_candidate_bid") or "") == _real_candidate_bid(cand) or str(h.get("action") or "").strip() == max_action for h in history if isinstance(h, dict))
        if max_clicked:
            submit = find_action_candidate(candidates, "submit")
            if submit and _real_candidate_bid(submit):
                return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="max_numeric_submit", diagnostics={"max_card_clicked": True, "submit_candidate_bid": _real_candidate_bid(submit)})
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "submit_button_not_found", "max_card_clicked": True, "submit_candidate_rejected_wrappers": True})
        return ExtractionDecision(answer=ans, extracted_data={"max_numeric": mx}, action=action, selected_candidate=cand, confidence=0.9, strategy="max_numeric_from_visible_text", diagnostics={"numeric_candidates_count": len(num_cands), "max_value": mx.get("value"), "selected_precise_candidate": _real_candidate_bid(cand), "submit_needed": True})

    if intent_name == "count_objects":
        count = len(extraction_context.get("list_like_items") or extraction_context.get("raw_candidates_summary") or [])
        ans = str(count)
        cand = find_action_candidate(candidates, ans)
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        return ExtractionDecision(answer=ans, extracted_data={"count": count}, action=action, selected_candidate=cand, confidence=0.7, strategy="count_visible_objects")

    if intent_name == "parity_check":
        clicked_bids = {str(h.get("selected_candidate_bid") or "") for h in history if isinstance(h, dict)}
        parity_buttons = {"odd": [c for c in candidates if _candidate_text(c).lower() == "odd"], "even": [c for c in candidates if _candidate_text(c).lower() == "even"]}
        row_map: dict[str, dict[str, Any]] = {}
        for num in extraction_context.get("numeric_values_candidates") or []:
            c = num.get("candidate") if isinstance(num, dict) else None
            if not isinstance(c, dict) or _is_wrapper_candidate(c):
                continue
            ptxt = str(c.get("parent_text") or "").strip()
            y = int(float(((c.get("bbox") or {}).get("y") or (c.get("browsergym_center_y") or 0))))
            row_id = ptxt or f"y:{y//10}:{int(float(num.get('value')))}"
            row_map.setdefault(row_id, {"row_id": row_id, "number": int(float(num.get("value"))), "odd": None, "even": None})
        pbids = [(_real_candidate_bid(c), c) for c in parity_buttons["odd"] + parity_buttons["even"] if _real_candidate_bid(c)]
        for row in row_map.values():
            for _, btn in pbids:
                by = int(float(((btn.get("bbox") or {}).get("y")) or (btn.get("browsergym_center_y") or 0)))
                match_row = True
                if str(row["row_id"]).startswith("y:"):
                    row_y_bucket = int(str(row["row_id"]).split(":")[1]) * 10
                    match_row = abs(by - row_y_bucket) <= 15
                if match_row:
                    if _candidate_text(btn).lower() == "odd" and row["odd"] is None:
                        row["odd"] = btn
                    if _candidate_text(btn).lower() == "even" and row["even"] is None:
                        row["even"] = btn
        rows = [r for r in row_map.values() if r.get("odd") and r.get("even")]
        answered_rows = [r["row_id"] for r in rows if _real_candidate_bid(r["odd"]) in clicked_bids or _real_candidate_bid(r["even"]) in clicked_bids]
        remaining = [r for r in rows if r["row_id"] not in answered_rows]
        if not remaining and rows:
            submit = find_action_candidate(candidates, "submit")
            if submit and _real_candidate_bid(submit):
                return ExtractionDecision(answer="submit", action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.8, strategy="parity_submit", diagnostics={"rows_total": len(rows), "row_numbers": [r["number"] for r in rows], "answered_row_ids": answered_rows, "submit_ready": True, "submit_bid": _real_candidate_bid(submit)})
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "submit_button_not_found", "rows_total": len(rows)})
        if not remaining:
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "parity_rows_not_found"})
        row = remaining[0]
        n = int(row["number"])
        ans = "even" if (n % 2 == 0) else "odd"
        cand = row[ans]
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        return ExtractionDecision(answer=ans, extracted_data={"number": n, "parity": ans}, action=action, selected_candidate=cand, confidence=0.8, strategy="parity_rowwise", diagnostics={"rows_total": len(rows), "row_numbers": [r["number"] for r in rows], "answered_row_ids": answered_rows, "current_row_id": row["row_id"], "current_row_number": n, "selected_parity": ans, "submit_ready": False})

    if intent_name in {"find_email", "find_important_email"}:
        emails = extraction_context.get("email_like_items") or []
        instruction = str(constraints.get("normalized_instruction") or "")
        m = re.search(r"(?:email by|from)\s+([a-zA-Z]+)", instruction, flags=re.I)
        target_sender = m.group(1).strip().lower() if m else ""
        requested = str(constraints.get("requested_email_action") or "open")
        reply_text = str(constraints.get("reply_text") or "")
        if keywords:
            filtered = []
            for e in emails:
                hay = " ".join([str(e.get("sender") or ""), str(e.get("subject") or ""), str(e.get("snippet") or ""), str(e.get("text") or "")]).lower()
                if any(k in hay for k in keywords):
                    filtered.append(e)
            emails = filtered or emails
        if emails:
            e = emails[0] if not target_sender else next((x for x in emails if str(x.get("sender") or "").strip().lower() == target_sender), None)
            if e is None:
                return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_sender_not_visible_or_not_found", "target_sender": target_sender, "email_rows_count": len(emails)})
            cand = e.get("candidate") if isinstance(e.get("candidate"), dict) else None
            if _is_wrapper_candidate(cand or {}):
                return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "email_row_candidate_invalid"})
            opened_email_detected = any("email-body" in str(c.get("className") or "").lower() or "email-header" in str(c.get("className") or "").lower() for c in candidates)
            if not opened_email_detected:
                action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
                return ExtractionDecision(answer=e.get("subject") or e.get("text"), extracted_data=e, action=action, selected_candidate=cand, confidence=0.75, strategy="email_open_row", diagnostics={"target_sender": target_sender, "requested_email_action": requested, "opened_email_detected": False, "matched_row_bid": _real_candidate_bid(cand)})
            if requested == "reply":
                reply_btn = find_action_candidate(candidates, "reply")
                if reply_btn and _real_candidate_bid(reply_btn) and not any("reply" in str(h.get("action") or "").lower() for h in history if isinstance(h, dict)):
                    return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(reply_btn)), selected_candidate=reply_btn, strategy="email_reply_click", diagnostics={"target_sender": target_sender, "requested_email_action": requested, "reply_text": reply_text, "opened_email_detected": True, "action_button_bid": _real_candidate_bid(reply_btn)})
                textbox = next((c for c in candidates if str(c.get("tag") or "").lower() in {"textarea", "input"} and "reply" in str(c.get("className") or "").lower()), None) or next((c for c in candidates if str(c.get("role") or "").lower() in {"textbox", "input"}), None)
                if textbox and _real_candidate_bid(textbox) and reply_text and not any(str(h.get("action") or "").strip() == f'fill("{_real_candidate_bid(textbox)}", "{reply_text}")' for h in history if isinstance(h, dict)):
                    return ExtractionDecision(action=f'fill("{_real_candidate_bid(textbox)}", "{reply_text}")', selected_candidate=textbox, strategy="email_reply_fill", diagnostics={"target_sender": target_sender, "requested_email_action": requested, "reply_text": reply_text, "opened_email_detected": True, "textbox_bid": _real_candidate_bid(textbox)})
                send_btn = find_action_candidate(candidates, "send") or find_action_candidate(candidates, "submit")
                if send_btn and _real_candidate_bid(send_btn):
                    return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(send_btn)), selected_candidate=send_btn, strategy="email_reply_send", diagnostics={"target_sender": target_sender, "requested_email_action": requested, "opened_email_detected": True, "send_bid": _real_candidate_bid(send_btn)})
            if requested in {"trash", "star", "forward"}:
                btn = find_action_candidate(candidates, "delete" if requested == "trash" else requested) or find_action_candidate(candidates, "trash" if requested == "trash" else requested)
                if btn and _real_candidate_bid(btn):
                    return ExtractionDecision(action=_browsergym_click_action(_real_candidate_bid(btn)), selected_candidate=btn, strategy=f"email_{requested}_action", diagnostics={"target_sender": target_sender, "requested_email_action": requested, "opened_email_detected": True, "action_button_bid": _real_candidate_bid(btn)})
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "requested_email_action_not_available", "target_sender": target_sender, "requested_email_action": requested, "opened_email_detected": opened_email_detected})

    if intent_name == "grid_lookup":
        grids = extraction_context.get("tables_or_grid_like_blocks") or []
        req_numbers = [int(n) for n in (constraints.get("numbers") or [])]
        if req_numbers:
            for g in grids:
                row = g.get("row")
                col = g.get("column")
                if (row in req_numbers or col in req_numbers) and isinstance(g.get("candidate"), dict):
                    cand = g["candidate"]
                    action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if _real_candidate_bid(cand) else None
                    return ExtractionDecision(answer=g.get("text"), extracted_data=g, action=action, selected_candidate=cand, confidence=0.75, strategy="grid_coordinate_match")
        return ExtractionDecision(answer=None, extracted_data={"reason": "no_grid_coordinate_mapping"}, action=None, selected_candidate=None, confidence=0.0, strategy="grid_no_decision", diagnostics={"reason": "no_grid_coordinate_mapping"})

    if intent_name == "find_text":
        lines = extraction_context.get("text_lines") or extraction_context.get("candidate_texts") or []
        quoted = [str(q).strip().lower() for q in (constraints.get("quoted_targets") or []) if str(q).strip()]
        if quoted:
            for q in quoted:
                cand = find_action_candidate(candidates, q, preferred_roles={"button", "link", "option", "menuitem", "treeitem"})
                if cand:
                    action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if _real_candidate_bid(cand) else None
                    return ExtractionDecision(answer=q, extracted_data={"text": q}, action=action, selected_candidate=cand, confidence=0.85, strategy="quoted_text_match")
                for ln in lines:
                    if q in str(ln).lower():
                        return ExtractionDecision(answer=str(ln), extracted_data={"text": str(ln)}, action=None, selected_candidate=None, confidence=0.7, strategy="quoted_text_in_lines")
        if lines:
            ans = str(lines[0])
            cand = find_action_candidate(candidates, ans)
            action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
            return ExtractionDecision(answer=ans, extracted_data={"text": ans}, action=action, selected_candidate=cand, confidence=0.5, strategy="first_text_line")

    if intent_name == "find_tree_node":
        trees = extraction_context.get("tree_like_items") or []
        targets = [str(t).strip().lower() for t in (constraints.get("quoted_targets") or [])]
        if trees:
            selected = trees[0] if not targets else next((t for t in trees if str(t.get("node_text") or "").strip().lower() == targets[0]), None)
            if selected is None:
                return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible"})
            cand = selected.get("candidate")
            if isinstance(cand, dict) and _real_candidate_bid(cand):
                return ExtractionDecision(answer=selected.get("node_text"), action=_browsergym_click_action(_real_candidate_bid(cand)), selected_candidate=cand, strategy="tree_node_click", confidence=0.6)
        return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "target_tree_node_not_visible"})
    if intent_name in {"find_calendar_event", "classify_object", "find_midpoint_or_middle_value"}:
        return ExtractionDecision(answer=None, extracted_data=None, action=None, selected_candidate=None, confidence=0.0, strategy="no_decision", diagnostics={"reason": "no generic extraction decision"})
    return None
