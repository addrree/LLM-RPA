from __future__ import annotations

import re
from typing import Any

from app.browsergym_integration.miniwob_grounding import MiniWoBGroundingResult, browsergym_click_action, normalize_text, real_candidate_bid


class MiniWoBDeterministicPolicy:
    def _norm(self, s: Any) -> str:
        return normalize_text(str(s or ""))

    def _candidate_texts(self, c: dict[str, Any]) -> list[str]:
        return [str(c.get(k)) for k in ("text", "innerText", "textContent", "name", "title", "aria-label", "aria_label", "href", "label", "value", "className") if c.get(k)]

    def _find_by_text(self, candidates: list[dict], target: str, roles: set[str] | None = None) -> dict | None:
        tn = self._norm(target)
        matches = []
        for c in candidates:
            role = self._norm(c.get("role"))
            if roles and role not in roles:
                continue
            if any(self._norm(v) == tn for v in self._candidate_texts(c)) and real_candidate_bid(c):
                matches.append(c)
        return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

    def _task(self, env_id: str, task_name: str) -> str:
        return (task_name or env_id or "").lower()

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
                target = self._norm(m.group(1))
                links = [c for c in candidates if (self._norm(c.get("role")) == "link" or self._norm(c.get("tag")) == "a" or c.get("href")) and real_candidate_bid(c)]
                exact = [c for c in links if any(self._norm(v) == target for v in self._candidate_texts(c))]
                if len(exact) == 1:
                    c = exact[0]
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_click_link", mapping_diagnostics={"policy_name": "click-link"})
                fallback = [c for c in candidates if real_candidate_bid(c) and any(self._norm(v) == target for v in self._candidate_texts(c))]
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
            suggestions = [c for c in candidates if self._norm(c.get("role")) in {"option", "listitem", "menuitem"} and real_candidate_bid(c)]
            chosen_recently = any(self._norm(h.get("selected_candidate_role")) in {"option", "listitem", "menuitem"} for h in history[-2:] if isinstance(h, dict))
            if chosen_recently and submit and real_candidate_bid(submit):
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_use_autocomplete_submit", mapping_diagnostics={"policy_name": "use-autocomplete"})
            for c in suggestions:
                vals = [self._norm(v) for v in self._candidate_texts(c)]
                if any(v.startswith(self._norm(prefix)) and (not suffix or v.endswith(self._norm(suffix))) for v in vals):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_use_autocomplete_pick", mapping_diagnostics={"policy_name": "use-autocomplete"})
            textbox = next((c for c in candidates if self._norm(c.get("role")) in {"textbox", "combobox", "input"} and real_candidate_bid(c)), None)
            if textbox and prefix:
                return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(textbox)}", "{prefix}")', selected_candidate=textbox, mapping_strategy="policy_use_autocomplete_fill", mapping_diagnostics={"policy_name": "use-autocomplete"})
        if "choose-date" in t:
            submit = self._find_by_text(candidates, "submit")
            if any(str((h.get("selected_candidate_text") or "")).isdigit() for h in history[-2:] if isinstance(h, dict)) and submit and real_candidate_bid(submit):
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_choose_date_submit", mapping_diagnostics={"policy_name": "choose-date"})
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", instr)
            if m:
                day = str(int(m.group(2)))
                day_c = self._find_by_text(candidates, day)
                if day_c and real_candidate_bid(day_c):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(day_c), action_syntax=action_syntax), selected_candidate=day_c, mapping_strategy="policy_choose_date_day", mapping_diagnostics={"policy_name": "choose-date"})
            tbs = [c for c in candidates if self._norm(c.get("role")) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
            date_tb = next((c for c in tbs if any("date" in self._norm(v) for v in self._candidate_texts(c))), None) or (tbs[0] if len(tbs) == 1 else None)
            if date_tb:
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(date_tb), action_syntax=action_syntax), selected_candidate=date_tb, mapping_strategy="policy_choose_date_open", mapping_diagnostics={"policy_name": "choose-date"})
        if "book-flight" in t:
            fm = re.search(r"from:\s*(.*?)\s*to:\s*(.*?)\s*on\s*(\d{1,2}/\d{1,2}/\d{4})", instr, flags=re.I)
            if fm:
                from_v, to_v, date_v = fm.group(1).strip(), fm.group(2).strip(), fm.group(3).strip()
                tbs = [c for c in candidates if self._norm(c.get("role")) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
                from_tb = next((c for c in tbs if "from" in " ".join(self._candidate_texts(c)).lower()), tbs[0] if tbs else None)
                to_tb = next((c for c in tbs if "to" in " ".join(self._candidate_texts(c)).lower() and c is not from_tb), tbs[1] if len(tbs) > 1 else None)
                date_tb = next((c for c in tbs if any(k in " ".join(self._candidate_texts(c)).lower() for k in ["date", "depart"] ) and c is not from_tb and c is not to_tb), tbs[2] if len(tbs) > 2 else None)
                if from_tb and not any(from_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(from_tb)}", "{from_v}")', selected_candidate=from_tb, mapping_strategy="policy_book_flight_from", mapping_diagnostics={"policy_name": "book-flight"})
                if to_tb and not any(to_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(to_tb)}", "{to_v}")', selected_candidate=to_tb, mapping_strategy="policy_book_flight_to", mapping_diagnostics={"policy_name": "book-flight"})
                if date_tb and not any(date_v.lower() in str(h.get("action") or "").lower() for h in history):
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(date_tb)}", "{date_v}")', selected_candidate=date_tb, mapping_strategy="policy_book_flight_date", mapping_diagnostics={"policy_name": "book-flight"})
                submit = self._find_by_text(candidates, "search") or self._find_by_text(candidates, "submit")
                if submit and real_candidate_bid(submit):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_book_flight_search", mapping_diagnostics={"policy_name": "book-flight"})
        if "click-menu" in t:
            m = re.search(r"select\s+(.+)$", instr, flags=re.I)
            if m:
                parts = [p.strip() for p in m.group(1).split(">") if p.strip()]
                if len(parts) >= 2:
                    final = self._find_by_text(candidates, parts[-1])
                    if final and real_candidate_bid(final):
                        return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(final), action_syntax=action_syntax), selected_candidate=final, mapping_strategy="policy_click_menu_leaf", mapping_diagnostics={"policy_name": "click-menu"})
        return None
