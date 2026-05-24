from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _find_click_candidate_by_text(candidates: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    t = str(target or "").strip().lower()
    for c in candidates:
        txt = str(c.get("text") or c.get("innerText") or c.get("name") or "").strip().lower()
        if txt == t or (t and t in txt):
            return c
    return None


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
        words = []
        for ln in extraction_context.get("text_lines") or []:
            if any(k in ln.lower() for k in ["submit", "textbox", "press"]):
                continue
            words.extend([w for w in __import__("re").findall(r"[A-Za-z0-9]+", ln)])
        if len(words) < int(ord_idx):
            return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "ordinal_out_of_range"})
        ans = words[int(ord_idx)-1].lower()
        textbox = next((c for c in candidates if str(c.get("role") or "").lower() in {"textbox", "input"} or str(c.get("tag") or "").lower() in {"input", "textarea"}), None)
        submitted = any("fill(" in str(h.get("action") or "") and ans in str(h.get("action") or "").lower() for h in history if isinstance(h, dict))
        if not submitted and textbox and _real_candidate_bid(textbox):
            return ExtractionDecision(answer=ans, action=f'fill("{_real_candidate_bid(textbox)}", "{ans}")', selected_candidate=textbox, confidence=0.9, strategy="ordinal_word_fill")
        submit = _find_click_candidate_by_text(candidates, "submit")
        if submit and _real_candidate_bid(submit):
            return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="ordinal_word_submit")
        return ExtractionDecision(answer=ans, strategy="no_decision", diagnostics={"reason": "submit_not_found"})

    if intent_name == "find_max_numeric" and nums:
        mx = max(nums, key=lambda n: float(n.get("value", 0)))
        ans = str(int(mx["value"])) if float(mx["value"]).is_integer() else str(mx["value"])
        cand = _find_click_candidate_by_text(candidates, ans)
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        if any("click(" in str(h.get("action") or "") and ans in str(h.get("selected_candidate_text") or "") for h in history if isinstance(h, dict)):
            submit = _find_click_candidate_by_text(candidates, "submit")
            if submit and _real_candidate_bid(submit):
                return ExtractionDecision(answer=ans, action=_browsergym_click_action(_real_candidate_bid(submit)), selected_candidate=submit, confidence=0.85, strategy="max_numeric_submit")
        return ExtractionDecision(answer=ans, extracted_data={"max_numeric": mx}, action=action, selected_candidate=cand, confidence=0.9, strategy="max_numeric_from_visible_text", diagnostics={"numeric_count": len(nums), "max_value": mx.get("value")})

    if intent_name == "count_objects":
        count = len(extraction_context.get("list_like_items") or extraction_context.get("raw_candidates_summary") or [])
        ans = str(count)
        cand = _find_click_candidate_by_text(candidates, ans)
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        return ExtractionDecision(answer=ans, extracted_data={"count": count}, action=action, selected_candidate=cand, confidence=0.7, strategy="count_visible_objects")

    if intent_name == "parity_check" and nums:
        answered = {str(h.get("selected_candidate_text") or "").strip().lower() for h in (constraints.get("history") or []) if isinstance(h, dict)}
        target_num = next((n for n in nums if int(float(n["value"])) in set(constraints.get("numbers") or [])), None)
        n = int(float((target_num or nums[0])["value"]))
        ans = "even" if (n % 2 == 0) else "odd"
        cand = next((c for c in candidates if str(c.get("text") or c.get("name") or "").strip().lower() == ans and ans not in answered), None) or _find_click_candidate_by_text(candidates, ans)
        action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
        return ExtractionDecision(answer=ans, extracted_data={"number": n, "parity": ans}, action=action, selected_candidate=cand, confidence=0.8, strategy="parity_rowwise")

    if intent_name in {"find_email", "find_important_email"}:
        emails = extraction_context.get("email_like_items") or []
        if intent_name == "find_important_email":
            emails = [e for e in emails if e.get("important")]
        if keywords:
            filtered = []
            for e in emails:
                hay = " ".join([str(e.get("sender") or ""), str(e.get("subject") or ""), str(e.get("snippet") or ""), str(e.get("text") or "")]).lower()
                if any(k in hay for k in keywords):
                    filtered.append(e)
            emails = filtered or emails
        if emails:
            e = emails[0]
            cand = e.get("candidate") if isinstance(e.get("candidate"), dict) else None
            if intent_name == "find_important_email":
                sender = " ".join(constraints.get("keywords") or [])
                row_text = str(e.get("text") or "").lower()
                if sender and sender not in row_text:
                    return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "sender_not_found"})
            action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
            return ExtractionDecision(answer=e.get("subject") or e.get("text"), extracted_data=e, action=action, selected_candidate=cand, confidence=0.75, strategy="email_item_match")

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
                cand = _find_click_candidate_by_text(candidates, q)
                if cand:
                    action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if _real_candidate_bid(cand) else None
                    return ExtractionDecision(answer=q, extracted_data={"text": q}, action=action, selected_candidate=cand, confidence=0.85, strategy="quoted_text_match")
                for ln in lines:
                    if q in str(ln).lower():
                        return ExtractionDecision(answer=str(ln), extracted_data={"text": str(ln)}, action=None, selected_candidate=None, confidence=0.7, strategy="quoted_text_in_lines")
        if lines:
            ans = str(lines[0])
            cand = _find_click_candidate_by_text(candidates, ans)
            action = _browsergym_click_action(_real_candidate_bid(cand), action_syntax=action_syntax or []) if cand and _real_candidate_bid(cand) else None
            return ExtractionDecision(answer=ans, extracted_data={"text": ans}, action=action, selected_candidate=cand, confidence=0.5, strategy="first_text_line")

    if intent_name == "find_tree_node":
        trees = extraction_context.get("tree_like_items") or []
        if trees:
            cand = trees[0].get("candidate")
            if isinstance(cand, dict) and _real_candidate_bid(cand):
                return ExtractionDecision(answer=trees[0].get("node_text"), action=_browsergym_click_action(_real_candidate_bid(cand)), selected_candidate=cand, strategy="tree_node_click", confidence=0.6)
        return ExtractionDecision(strategy="no_decision", diagnostics={"reason": "tree_structure_not_found"})
    if intent_name in {"find_calendar_event", "classify_object", "find_midpoint_or_middle_value"}:
        return ExtractionDecision(answer=None, extracted_data=None, action=None, selected_candidate=None, confidence=0.0, strategy="no_decision", diagnostics={"reason": "no generic extraction decision"})
    return None
