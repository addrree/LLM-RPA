from __future__ import annotations

import re
from typing import Any

from app.browsergym_integration.miniwob_grounding import MiniWoBGroundingResult, browsergym_click_action, normalize_text, real_candidate_bid


def unwrap_ax_value(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        for key in ("value", "name", "text", "role", "attributeValue"):
            if key in x:
                return unwrap_ax_value(x.get(key))
        return ""
    if isinstance(x, list):
        return " ".join(str(unwrap_ax_value(item) or "") for item in x).strip()
    return ""


class MiniWoBDeterministicPolicy:
    NON_RECOVERABLE_ERRORS = {
        "menu_requires_hover_no_supported_action",
        "autocomplete_suggestions_not_found",
        "datepicker_header_not_found",
    }
    def _norm(self, s: Any) -> str:
        return normalize_text(str(unwrap_ax_value(s) or ""))

    def _candidate_texts(self, c: dict[str, Any]) -> list[str]:
        keys = ("text", "innerText", "textContent", "name", "title", "aria-label", "aria_label", "href", "label", "value", "className", "role", "tag", "type", "id", "placeholder")
        return [str(unwrap_ax_value(c.get(k)) or "") for k in keys if str(unwrap_ax_value(c.get(k)) or "").strip()]

    def candidate_role(self, c: dict[str, Any]) -> str:
        return self._norm(c.get("role"))

    def candidate_type(self, c: dict[str, Any]) -> str:
        return self._norm(c.get("type"))

    def candidate_tag(self, c: dict[str, Any]) -> str:
        return self._norm(c.get("tag"))

    def candidate_text(self, c: dict[str, Any]) -> str:
        for key in ("text", "innerText", "textContent", "name", "label", "title", "placeholder", "value"):
            v = str(unwrap_ax_value(c.get(key)) or "").strip()
            if v:
                return v
        vals = self._candidate_texts(c)
        return vals[0] if vals else ""

    def _find_by_text(self, candidates: list[dict], target: str, roles: set[str] | None = None) -> dict | None:
        tn = self._norm(target)
        matches = []
        for c in candidates:
            role = self.candidate_role(c)
            if roles and role not in roles:
                continue
            if any(self._norm(v) == tn for v in self._candidate_texts(c)) and real_candidate_bid(c):
                matches.append(c)
        return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

    def _task(self, env_id: str, task_name: str) -> str:
        return (task_name or env_id or "").lower()

    def _has_mapping(self, history: list[dict], mapping: str) -> bool:
        return any(str(h.get("mapping_strategy") or "") == mapping for h in history if isinstance(h, dict))

    def _action_supported(self, action_syntax: list[str], action_name: str) -> bool:
        return any(str(a or "").strip().lower().startswith(f"{action_name}(") for a in (action_syntax or []))
    def _extract_autocomplete_suggestions(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blocked_role_tokens = {"generic", "listitem", "option", "menuitem"}
        out: list[dict[str, Any]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            role = self.candidate_role(c)
            cls = self._norm(c.get("className"))
            if role not in {"option", "listitem", "menuitem", "generic"} and "ui-autocomplete" not in cls and "ui-menu-item" not in cls and "ui-menu" not in cls:
                continue
            txt = str(unwrap_ax_value(c.get("innerText")) or unwrap_ax_value(c.get("textContent")) or unwrap_ax_value(c.get("text")) or unwrap_ax_value(c.get("name")) or "").strip()
            txt_norm = self._norm(txt)
            if txt_norm in blocked_role_tokens:
                continue
            out.append(c)
        return out

    def try_act(self, *, env_id: str, task_name: str, instruction: str, candidates: list[dict], history: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        t = self._task(env_id, task_name)
        instr = str(instruction or "")
        if "click-button-sequence" in t:
            labels = re.findall(r"click button\s+([A-Za-z0-9_-]+)", instr, flags=re.I)
            if len(labels) >= 2:
                clicked = {self._norm(h.get("selected_candidate_text") or "") for h in history if isinstance(h, dict)}
                for lbl in labels:
                    if self._norm(lbl) in clicked:
                        continue
                    c = self._find_by_text(candidates, lbl, {"button"})
                    if c and real_candidate_bid(c):
                        return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_click_button_sequence", mapping_diagnostics={"policy_name": "click-button-sequence"})
        if "click-dialog" in t:
            close_words = {"close", "x", "×"}
            button = None
            fallback = None
            for c in candidates:
                bid = real_candidate_bid(c)
                if not bid:
                    continue
                vals = {self._norm(v) for v in self._candidate_texts(c)}
                if vals & close_words:
                    if self._norm(c.get("role")) == "button":
                        button = c
                        break
                    fallback = fallback or c
            chosen = button or fallback
            if chosen:
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(chosen), action_syntax=action_syntax), selected_candidate=chosen, mapping_strategy="policy_click_dialog", mapping_diagnostics={"policy_name": "click-dialog"})
        if "click-link" in t:
            m = re.search(r"['\"]([^'\"]+)['\"]", instr)
            if m:
                raw_target = m.group(1).strip()
                target = self._norm(raw_target)
                loose = self._norm(raw_target.rstrip(".,!?;:"))
                links = [c for c in candidates if (self.candidate_role(c) == "link" or self.candidate_tag(c) == "a" or c.get("href")) and real_candidate_bid(c) and any(self._norm(v) for v in self._candidate_texts(c))]
                exact = [c for c in links if any(self._norm(v) == target for v in self._candidate_texts(c))]
                if len(exact) == 1:
                    c = exact[0]
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_click_link", mapping_diagnostics={"policy_name": "click-link"})
                fallback = [c for c in links if any(self._norm(v) == loose for v in self._candidate_texts(c))]
                if len(fallback) == 1:
                    c = fallback[0]
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_click_link_fallback", mapping_diagnostics={"policy_name": "click-link"})
                return None
        if "use-autocomplete" in t:
            p = re.search(r"starts with\s+['\"]([^'\"]+)['\"]", instr, flags=re.I)
            s = re.search(r"ends with\s+['\"]([^'\"]+)['\"]", instr, flags=re.I)
            prefix = (p.group(1) if p else "").strip()
            suffix = (s.group(1) if s else "").strip()
            submit = self._find_by_text(candidates, "submit")
            suggestions = [c for c in self._extract_autocomplete_suggestions(candidates) if real_candidate_bid(c)]
            chosen_recently = self._has_mapping(history, "policy_use_autocomplete_pick") or any(self._norm(h.get("selected_candidate_role")) in {"option", "listitem", "menuitem"} for h in history[-3:] if isinstance(h, dict))
            if chosen_recently and submit and real_candidate_bid(submit):
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_use_autocomplete_submit", mapping_diagnostics={"policy_name": "use-autocomplete"})
            suggestion_texts = [self.candidate_text(c) for c in suggestions if self.candidate_text(c).strip()]
            for c in suggestions:
                vals = [self._norm(v) for v in (self.candidate_text(c), c.get("innerText"), c.get("textContent"), c.get("name")) if str(unwrap_ax_value(v) or "").strip()]
                if any(v.startswith(self._norm(prefix)) and (not suffix or v.endswith(self._norm(suffix))) for v in vals):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_use_autocomplete_pick", mapping_diagnostics={"policy_name": "use-autocomplete", "autocomplete_prefix": prefix, "autocomplete_suffix": suffix, "suggestions_count": len(suggestions), "suggestion_texts": suggestion_texts, "selected_suggestion_bid": real_candidate_bid(c), "repeated_fill_blocked": True})
            textbox = next((c for c in candidates if self.candidate_role(c) in {"textbox", "combobox", "input"} and ("tags" in " ".join(self._candidate_texts(c)).lower()) and real_candidate_bid(c)), None) or next((c for c in candidates if self.candidate_role(c) in {"textbox", "combobox", "input"} and real_candidate_bid(c)), None)
            if textbox and prefix:
                textbox_value = self._norm(textbox.get("value") or textbox.get("text") or textbox.get("innerText") or textbox.get("textContent"))
                prefix_norm = self._norm(prefix)
                base_diag = {"policy_name": "use-autocomplete", "autocomplete_prefix": prefix, "autocomplete_suffix": suffix, "textbox_value": textbox_value, "suggestions_count": len(suggestions), "suggestion_texts": suggestion_texts}
                if not textbox_value.startswith(prefix_norm):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(textbox)}", "{prefix}")', selected_candidate=textbox, mapping_strategy="policy_use_autocomplete_fill_prefix", mapping_diagnostics={**base_diag, "repeated_fill_blocked": False})
                if not self._has_mapping(history, "policy_use_autocomplete_wait_suggestions"):
                    return MiniWoBGroundingResult(action="noop()", selected_candidate=textbox, mapping_strategy="policy_use_autocomplete_wait_suggestions", mapping_diagnostics={**base_diag, "repeated_fill_blocked": True})
                return MiniWoBGroundingResult(action="noop()", selected_candidate=textbox, mapping_strategy="policy_use_autocomplete_not_found", mapping_error="autocomplete_suggestions_not_found", mapping_diagnostics={**base_diag, "repeated_fill_blocked": True})
        if "choose-date" in t:
            submit = self._find_by_text(candidates, "submit")
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", instr)
            if (self._has_mapping(history, "policy_choose_date_day") or any(str((h.get("selected_candidate_text") or "")).isdigit() for h in history[-2:] if isinstance(h, dict))) and submit and real_candidate_bid(submit) and m:
                target_date = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{int(m.group(3)):04d}"
                tbs = [c for c in candidates if self.candidate_role(c) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
                date_tb = next((c for c in tbs if any("date" in self._norm(v) for v in self._candidate_texts(c))), None) or (tbs[0] if len(tbs) == 1 else None)
                tb_value = self._norm((date_tb or {}).get("value") or (date_tb or {}).get("text") or "")
                if target_date.lower() in tb_value or self._has_mapping(history, "policy_choose_date_fill"):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_choose_date_submit", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date, "textbox_value": tb_value})
                return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_invalid_state", mapping_error="datepicker_header_not_found", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date, "textbox_value": tb_value})
            if m:
                day = str(int(m.group(2)))
                month = int(m.group(1))
                year = int(m.group(3))
                header = " ".join(" ".join(self._candidate_texts(c)) for c in candidates).lower()
                if str(year) in header:
                    month_words = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june", 7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}
                    prev_btn = next((c for c in candidates if ("prev" in " ".join(self._candidate_texts(c)).lower() or "ui-datepicker-prev" in self._norm(c.get("className"))) and real_candidate_bid(c)), None)
                    next_btn = next((c for c in candidates if ("next" in " ".join(self._candidate_texts(c)).lower() or "ui-datepicker-next" in self._norm(c.get("className"))) and real_candidate_bid(c)), None)
                    current_month = next((mno for mno, mname in month_words.items() if mname in header), None)
                    if current_month and current_month != month:
                        btn = next_btn if current_month < month else prev_btn
                        if btn:
                            strategy = "policy_choose_date_next_month" if current_month < month else "policy_choose_date_prev_month"
                            return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(btn), action_syntax=action_syntax), selected_candidate=btn, mapping_strategy=strategy, mapping_diagnostics={"policy_name": "choose-date", "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header, "current_month": current_month, "current_year": year})
                header_candidates = [c for c in candidates if "ui-datepicker-title" in self._norm(c.get("className")) or "ui-datepicker-month" in self._norm(c.get("className")) or "ui-datepicker-year" in self._norm(c.get("className"))]
                if header_candidates and str(year) not in header:
                    return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_header_not_found", mapping_error="datepicker_header_not_found", mapping_diagnostics={"policy_name": "choose-date", "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header, "datepicker_header_not_found": True})
                day_candidates = [c for c in candidates if real_candidate_bid(c) and self.candidate_text(c) == day and ("ui-state-default" in self._norm(c.get("className")) or self.candidate_tag(c) in {"a", "button"}) and "ui-priority-secondary" not in self._norm(c.get("className")) and "other-month" not in self._norm(c.get("className"))]
                day_c = day_candidates[0] if day_candidates else None
                if day_c and real_candidate_bid(day_c):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(day_c), action_syntax=action_syntax), selected_candidate=day_c, mapping_strategy="policy_choose_date_day", mapping_diagnostics={"policy_name": "choose-date", "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header})
            tbs = [c for c in candidates if self.candidate_role(c) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
            date_tb = next((c for c in tbs if any("date" in self._norm(v) for v in self._candidate_texts(c))), None) or (tbs[0] if len(tbs) == 1 else None)
            if date_tb:
                if m and self._action_supported(action_syntax, "fill"):
                    target_date = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{int(m.group(3)):04d}"
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(date_tb)}", "{target_date}")', selected_candidate=date_tb, mapping_strategy="policy_choose_date_fill", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date})
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(date_tb), action_syntax=action_syntax), selected_candidate=date_tb, mapping_strategy="policy_choose_date_open", mapping_diagnostics={"policy_name": "choose-date"})
        if "book-flight" in t:
            fm = re.search(r"from:\s*(.*?)\s*to:\s*(.*?)\s*on\s*(\d{1,2}/\d{1,2}/\d{4})", instr, flags=re.I)
            if fm:
                from_v, to_v, date_v = fm.group(1).strip(), fm.group(2).strip(), fm.group(3).strip()
                tbs = [c for c in candidates if self.candidate_role(c) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
                from_tb = next((c for c in tbs if "from" in " ".join(self._candidate_texts(c)).lower()), tbs[0] if tbs else None)
                to_tb = next((c for c in tbs if "to" in " ".join(self._candidate_texts(c)).lower() and c is not from_tb), tbs[1] if len(tbs) > 1 else None)
                date_tb = next((c for c in tbs if any(k in " ".join(self._candidate_texts(c)).lower() for k in ["date", "depart"] ) and c is not from_tb and c is not to_tb), tbs[2] if len(tbs) > 2 else None)
                if from_tb and not any(from_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(from_tb)}", "{from_v}")', selected_candidate=from_tb, mapping_strategy="policy_book_flight_from", mapping_diagnostics={"policy_name": "book-flight"})
                if to_tb and not any(to_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(to_tb)}", "{to_v}")', selected_candidate=to_tb, mapping_strategy="policy_book_flight_to", mapping_diagnostics={"policy_name": "book-flight"})
                if date_tb and not any(date_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(date_tb)}", "{date_v}")', selected_candidate=date_tb, mapping_strategy="policy_book_flight_date", mapping_diagnostics={"policy_name": "book-flight"})
                if not (from_tb and to_tb and date_tb):
                    return None
                submit = self._find_by_text(candidates, "search") or self._find_by_text(candidates, "submit")
                if submit and real_candidate_bid(submit) and all(any(v.lower() in str(h.get("action") or "").lower() for h in history) for v in (from_v, to_v, date_v)):
                    search_bid = real_candidate_bid(submit)
                    recent_search = [h for h in history if isinstance(h, dict) and str(h.get("mapping_strategy") or "") == "policy_book_flight_search" and f'"{search_bid}"' in str(h.get("action") or "")]
                    if len(recent_search) >= 2:
                        return MiniWoBGroundingResult(action="noop()", selected_candidate=submit, mapping_strategy="policy_book_flight_search_no_progress", mapping_error="search_no_progress", mapping_diagnostics={"policy_name": "book-flight", "search_no_progress": True})
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_book_flight_search", mapping_diagnostics={"policy_name": "book-flight"})
        if "click-menu" in t:
            m = re.search(r"select\s+(.+)$", instr, flags=re.I)
            if m:
                parts = [p.strip() for p in m.group(1).split(">") if p.strip()]
                if len(parts) >= 2:
                    final = self._find_by_text(candidates, parts[-1])
                    if final and real_candidate_bid(final):
                        return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(final), action_syntax=action_syntax), selected_candidate=final, mapping_strategy="policy_click_menu_leaf", mapping_diagnostics={"policy_name": "click-menu"})
                    parent = self._find_by_text(candidates, parts[0])
                    if parent and real_candidate_bid(parent) and self._action_supported(action_syntax, "hover"):
                        return MiniWoBGroundingResult(action=f'hover("{real_candidate_bid(parent)}")', selected_candidate=parent, mapping_strategy="policy_click_menu_hover_parent", mapping_diagnostics={"policy_name": "click-menu"})
                    if parent and (self._action_supported(action_syntax, "mouse_move") or self._action_supported(action_syntax, "move_to")):
                        x = parent.get("browsergym_center_x") or parent.get("center_x")
                        y = parent.get("browsergym_center_y") or parent.get("center_y")
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            return MiniWoBGroundingResult(action=f"mouse_move({int(x)}, {int(y)})", selected_candidate=parent, mapping_strategy="policy_click_menu_hover_parent", mapping_diagnostics={"policy_name": "click-menu"})
                    return MiniWoBGroundingResult(action="noop()", selected_candidate=parent, mapping_strategy="policy_click_menu_hover_required", mapping_error="menu_requires_hover_no_supported_action", mapping_diagnostics={"policy_name": "click-menu", "menu_requires_hover_no_supported_action": True})
        return None
