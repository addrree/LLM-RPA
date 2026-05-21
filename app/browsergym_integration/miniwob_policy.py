from __future__ import annotations

import re
from typing import Any

from app.browsergym_integration.miniwob_grounding import MiniWoBGroundingResult, browsergym_click_action, normalize_text, real_candidate_bid


class MiniWoBDeterministicPolicy:
    def _norm(self, s: Any) -> str:
        return normalize_text(str(s or ""))

    def _candidate_texts(self, c: dict[str, Any]) -> list[str]:
        out = []
        for k in ("text", "innerText", "textContent", "name", "title", "aria-label", "aria_label", "href", "label", "value"):
            v = c.get(k)
            if v:
                out.append(str(v))
        return out

    def _find_by_text(self, candidates: list[dict], target: str, roles: set[str] | None = None) -> dict | None:
        tn = self._norm(target)
        matches = []
        for c in candidates:
            role = self._norm(c.get("role"))
            if roles and role not in roles:
                continue
            if any(self._norm(v) == tn for v in self._candidate_texts(c)):
                if real_candidate_bid(c):
                    matches.append(c)
        return matches[0] if len(matches) == 1 else (matches[0] if matches else None)

    def try_act(self, *, env_id: str, task_name: str, instruction: str, candidates: list[dict], history: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        t = (task_name or env_id or "").lower()
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
                links = [c for c in candidates if isinstance(c, dict) and (self._norm(c.get("role")) == "link" or self._norm(c.get("tag")) == "a" or c.get("href")) and real_candidate_bid(c)]
                exact = [c for c in links if any(self._norm(v) == target for v in self._candidate_texts(c))]
                if len(exact) == 1:
                    c = exact[0]
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_click_link", mapping_diagnostics={"policy_name": "click-link"})
        if "choose-date" in t:
            submit = self._find_by_text(candidates, "submit")
            day_clicked = any((h.get("selected_candidate_text") or "").strip().isdigit() for h in history if isinstance(h, dict))
            if day_clicked and submit and real_candidate_bid(submit):
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_choose_date_submit", mapping_diagnostics={"policy_name": "choose-date"})
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", instr)
            if m:
                day = str(int(m.group(2)))
                day_c = self._find_by_text(candidates, day)
                if day_c and real_candidate_bid(day_c):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(day_c), action_syntax=action_syntax), selected_candidate=day_c, mapping_strategy="policy_choose_date_day", mapping_diagnostics={"policy_name": "choose-date"})
            for c in candidates:
                role = self._norm(c.get("role"))
                if role in {"textbox", "input", "combobox"} and any("date" in self._norm(v) for v in self._candidate_texts(c)) and real_candidate_bid(c):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_choose_date_open", mapping_diagnostics={"policy_name": "choose-date"})
        return None
