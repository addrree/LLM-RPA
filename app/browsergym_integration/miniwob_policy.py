from __future__ import annotations

import re
from typing import Any

from app.browsergym_integration.miniwob_grounding import MiniWoBGroundingResult, browsergym_click_action, browsergym_fill_action, browsergym_select_option_action, find_submit_button, normalize_text, real_candidate_bid


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


def make_click_action(candidate: dict[str, Any], action_syntax: list[str] | None = None) -> tuple[str, str] | tuple[None, None]:
    bid = real_candidate_bid(candidate)
    if bid:
        return browsergym_click_action(bid, action_syntax=action_syntax or []), "bid_click"
    visible = candidate.get("visible") is not False
    x = candidate.get("browsergym_center_x")
    y = candidate.get("browsergym_center_y")
    bbox = candidate.get("bbox") or candidate.get("browsergym_bbox")
    if visible and isinstance(bbox, dict) and isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return f'mouse_click({int(x)}, {int(y)}, "left")', "dom_center_mouse_click"
    return None, None


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
    def _link_texts(self, c: dict[str, Any]) -> list[str]:
        keys = ("text", "innerText", "textContent", "name", "title", "ariaLabel", "href")
        blocked = {"generic", "listitem", "option", "menuitem"}
        out = []
        for k in keys:
            v = str(unwrap_ax_value(c.get(k)) or "").strip()
            if v and self._norm(v) not in blocked:
                out.append(v)
        return out

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

    def _is_buttonish(self, c: dict[str, Any]) -> bool:
        tag = self.candidate_tag(c)
        role = self.candidate_role(c)
        typ = self.candidate_type(c)
        return role == "button" or tag == "button" or (tag == "input" and typ in {"button", "submit"})

    def _is_textbox(self, c: dict[str, Any]) -> bool:
        tag = self.candidate_tag(c)
        role = self.candidate_role(c)
        typ = self.candidate_type(c)
        return role in {"textbox", "input", "combobox"} or tag == "textarea" or (tag == "input" and typ not in {"button", "submit", "checkbox", "radio"})

    def _is_select_control(self, c: dict[str, Any]) -> bool:
        tag = self.candidate_tag(c)
        role = self.candidate_role(c)
        return tag == "select" or role in {"combobox", "listbox"}

    def _is_link_like(self, c: dict[str, Any]) -> bool:
        return self.candidate_tag(c) == "a" or self.candidate_role(c) == "link" or bool(str(c.get("href") or "").strip()) or "alink" in self._norm(c.get("className"))

    def _is_wrapper(self, c: dict[str, Any]) -> bool:
        cid = self._norm(c.get("id"))
        tag = self.candidate_tag(c)
        parent_tag = self._norm(c.get("parent_tag"))
        text = str(unwrap_ax_value(c.get("text")) or unwrap_ax_value(c.get("innerText")) or "").strip()
        return cid in {"wrap", "area"} or parent_tag == "body" and tag in {"div", "span"} and ("\n" in text or len(text) > 80)

    def _visible_enabled(self, c: dict[str, Any]) -> bool:
        return isinstance(c, dict) and c.get("visible") is not False and c.get("disabled") is not True

    def _history_clicked_bid(self, history: list[dict], bid: str) -> bool:
        if not bid:
            return False
        needle = f'"{bid}"'
        return any(
            str(h.get("selected_candidate_bid") or "") == bid or needle in str(h.get("action") or "")
            for h in history
            if isinstance(h, dict)
        )

    def _history_clicked_texts(self, history: list[dict]) -> set[str]:
        return {self._norm(h.get("selected_candidate_text") or "") for h in history if isinstance(h, dict) and self._norm(h.get("selected_candidate_text") or "")}

    def _history_has_fill(self, history: list[dict], bid: str, text: str) -> bool:
        bid_n = str(bid or "").strip()
        text_n = self._norm(text)
        for h in history:
            action = str((h or {}).get("action") or "")
            if bid_n and f'fill("{bid_n}"' in action and text_n in self._norm(action):
                return True
        return False

    def _history_has_select_option(self, history: list[dict], bid: str, text: str) -> bool:
        bid_n = str(bid or "").strip()
        text_n = self._norm(text)
        for h in history:
            action = str((h or {}).get("action") or "")
            if bid_n and f'select_option("{bid_n}"' in action and text_n in self._norm(action):
                return True
        return False

    def _textbox_value_matches(self, c: dict[str, Any] | None, text: str) -> bool:
        if not isinstance(c, dict):
            return False
        current = self._norm(c.get("value") or c.get("text") or c.get("innerText") or c.get("textContent"))
        return bool(text and self._norm(text) == current)

    def _text_match_score(self, c: dict[str, Any], target: str, *, include_parent: bool = False) -> int:
        target_n = self._norm(target)
        if not target_n:
            return 0
        text_keys = ("text", "innerText", "textContent", "name", "label", "title", "ariaLabel", "aria_label", "aria-label", "value", "href", "placeholder")
        fields = [str(unwrap_ax_value(c.get(k)) or "") for k in text_keys if str(unwrap_ax_value(c.get(k)) or "").strip()]
        score = 0
        for raw in fields:
            text = self._norm(raw)
            if not text:
                continue
            if text == target_n:
                score = max(score, 120)
            elif text.rstrip(".,!?;:") == target_n.rstrip(".,!?;:"):
                score = max(score, 110)
            elif len(target_n) > 2 and (target_n in text or text in target_n):
                score = max(score, 60)
        if include_parent:
            own_text = self._norm(self.candidate_text(c))
            parent_text = str(unwrap_ax_value(c.get("parent_text")) or "")
            parent_lines = [line.strip() for line in parent_text.splitlines() if line.strip()]
            parent_match_allowed = (
                not own_text
                or own_text == target_n
                or self.candidate_tag(c) in {"input", "option"}
                or self.candidate_role(c) in {"checkbox", "radio", "option"}
            )
            if parent_match_allowed:
                for raw in parent_lines:
                    text = self._norm(raw)
                    if text == target_n:
                        score = max(score, 80)
                    elif not own_text and (target_n in text or text in target_n):
                        score = max(score, 40)
        return score

    def _best_by_text(
        self,
        candidates: list[dict[str, Any]],
        target: str,
        *,
        prefer: str = "",
        include_parent: bool = False,
        allow_dom_mouse: bool = True,
    ) -> dict[str, Any] | None:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, c in enumerate(candidates or []):
            if not self._visible_enabled(c):
                continue
            score = self._text_match_score(c, target, include_parent=include_parent)
            if score <= 0:
                continue
            if prefer == "button" and self._is_buttonish(c):
                score += 45
            elif prefer == "link" and (self.candidate_tag(c) == "a" or self.candidate_role(c) == "link" or "alink" in self._norm(c.get("className")) or c.get("href")):
                score += 45
            elif prefer == "checkbox" and (self.candidate_role(c) == "checkbox" or self.candidate_type(c) == "checkbox" or self.candidate_tag(c) == "label"):
                score += 35
            elif prefer == "radio" and (self.candidate_role(c) == "radio" or self.candidate_type(c) == "radio" or self.candidate_tag(c) == "label"):
                score += 35
            elif prefer == "tab" and ("tab" in self.candidate_role(c) or self.candidate_tag(c) == "a"):
                score += 35
            if self._is_wrapper(c):
                score -= 80
            if not real_candidate_bid(c) and not allow_dom_mouse:
                score -= 60
            scored.append((score, -idx, c))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return scored[0][2] if scored and scored[0][0] >= 50 else None

    def _click_result(self, c: dict[str, Any], action_syntax: list[str], strategy: str, diagnostics: dict[str, Any] | None = None) -> MiniWoBGroundingResult | None:
        action, click_strategy = make_click_action(c, action_syntax)
        if not action:
            return None
        md = dict(diagnostics or {})
        md.setdefault("click_strategy", click_strategy)
        return MiniWoBGroundingResult(action=action, selected_candidate=c, mapping_strategy=strategy, mapping_diagnostics=md)

    def _submit_result(self, candidates: list[dict[str, Any]], action_syntax: list[str], strategy: str, diagnostics: dict[str, Any] | None = None) -> MiniWoBGroundingResult | None:
        submit = find_submit_button(candidates) or self._best_by_text(candidates, "submit", prefer="button", allow_dom_mouse=False)
        if not submit:
            return None
        return self._click_result(submit, action_syntax, strategy, diagnostics)

    def _bbox_contains(self, bbox: dict[str, Any] | None, x: float, y: float) -> bool:
        if not isinstance(bbox, dict):
            return False
        left = float(bbox.get("left", bbox.get("x", 0)) or 0)
        top = float(bbox.get("top", bbox.get("y", 0)) or 0)
        right = bbox.get("right")
        bottom = bbox.get("bottom")
        if right is None:
            right = left + float(bbox.get("width") or 0)
        if bottom is None:
            bottom = top + float(bbox.get("height") or 0)
        return left <= x <= float(right) and top <= y <= float(bottom)

    def _safe_button_click_result(
        self,
        c: dict[str, Any],
        candidates: list[dict[str, Any]],
        action_syntax: list[str],
        strategy: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> MiniWoBGroundingResult | None:
        bbox = self._bbox(c)
        if not (self._action_supported(action_syntax, "mouse_click") and isinstance(bbox, dict)):
            return self._click_result(c, action_syntax, strategy, diagnostics)
        left = float(bbox.get("left", bbox.get("x", 0)) or 0)
        top = float(bbox.get("top", bbox.get("y", 0)) or 0)
        width = float(bbox.get("width") or 0)
        height = float(bbox.get("height") or 0)
        if width <= 0 or height <= 0:
            return self._click_result(c, action_syntax, strategy, diagnostics)
        target_bid = real_candidate_bid(c)
        target_text = self._norm(self.candidate_text(c))
        blockers = []
        for other in candidates or []:
            if other is c or not self._visible_enabled(other) or not self._is_buttonish(other):
                continue
            other_bid = real_candidate_bid(other)
            if target_bid and other_bid == target_bid:
                continue
            other_bbox = self._bbox(other)
            if not other_bbox:
                continue
            if target_text and self._norm(self.candidate_text(other)) == target_text and other_bbox == bbox:
                continue
            blockers.append(other_bbox)
        xs = [left + width * frac for frac in (0.5, 0.18, 0.82, 0.3, 0.7)]
        ys = [top + height * frac for frac in (0.5, 0.18, 0.82, 0.3, 0.7)]
        points = [(x, y) for y in ys for x in xs]
        chosen = next(((x, y) for x, y in points if not any(self._bbox_contains(blocker, x, y) for blocker in blockers)), None)
        if chosen is None:
            chosen = (left + width / 2.0, top + height / 2.0)
        scale = self._infer_browsergym_scale(candidates)
        md = dict(diagnostics or {})
        md.update({"click_strategy": "safe_mouse_click", "page_x": chosen[0], "page_y": chosen[1], "scale": scale})
        return MiniWoBGroundingResult(action=self._browsergym_mouse_click(chosen[0], chosen[1], scale), selected_candidate=c, mapping_strategy=strategy, mapping_diagnostics=md)

    def _parse_select_targets(self, instruction: str) -> tuple[list[str], bool]:
        text = str(instruction or "").strip()
        m = re.search(r"select\s+(.+?)\s+from\s+the\s+scroll\s+list\s+and\s+click\s+submit\.?$", text, flags=re.I)
        if m:
            return [part.strip() for part in re.split(r",\s*", m.group(1)) if part.strip()], True
        m = re.search(r"select\s+(.+?)\s+and\s+click\s+submit\.?$", text, flags=re.I)
        if not m:
            return [], False
        target = m.group(1).strip()
        if self._norm(target) == "nothing":
            return [], True
        return [part.strip() for part in re.split(r",\s*", target) if part.strip()], True

    def _try_basic_action_intent(self, *, instruction: str, candidates: list[dict], history: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        instr = str(instruction or "").strip()
        instr_n = self._norm(instr)
        if not instr_n:
            return None

        if "dialog" in instr_n and ("\"x\"" in instr.lower() or " close" in f" {instr_n} "):
            close_candidates = [
                c
                for c in candidates
                if self._visible_enabled(c)
                and ("ui-dialog-titlebar-close" in self._norm(c.get("className")) or self._norm(c.get("title")) in {"close", "x"} or self._norm(self.candidate_text(c)) in {"close", "x"})
                and not self._is_wrapper(c)
            ]
            close_candidates.sort(key=lambda c: (0 if self._is_buttonish(c) else 1, len(str(self.candidate_text(c) or ""))))
            if close_candidates:
                return self._click_result(close_candidates[0], action_syntax, "policy_basic_dialog_close", {"policy_name": "basic_action", "intent": "dialog_close"})

        labels = re.findall(r"click button\s+([A-Za-z0-9#_-]+)", instr, flags=re.I)
        if len(labels) >= 2:
            clicked = self._history_clicked_texts(history)
            for label in labels:
                if self._norm(label) in clicked:
                    continue
                c = self._best_by_text(candidates, label, prefer="button", allow_dom_mouse=False)
                if c:
                    return self._safe_button_click_result(c, candidates, action_syntax, "policy_basic_button_sequence", {"policy_name": "basic_action", "target_text": label})

        login = re.search(r'username\s+"([^"]+)"\s+and\s+the\s+password\s+"([^"]+)"', instr, flags=re.I)
        if login:
            username, password = login.group(1), login.group(2)
            boxes = [c for c in candidates if self._visible_enabled(c) and self._is_textbox(c) and real_candidate_bid(c)]
            password_box = next((c for c in boxes if self.candidate_type(c) == "password"), None)
            username_box = next((c for c in boxes if c is not password_box), boxes[0] if boxes else None)
            if password_box is None and len(boxes) >= 2:
                password_box = boxes[1]
            if username_box and not (self._textbox_value_matches(username_box, username) or self._history_has_fill(history, real_candidate_bid(username_box), username)):
                return MiniWoBGroundingResult(action=browsergym_fill_action(real_candidate_bid(username_box), username), selected_candidate=username_box, mapping_strategy="policy_basic_login_username", mapping_diagnostics={"policy_name": "basic_action", "target_field": "username"})
            if password_box and not (self._textbox_value_matches(password_box, password) or self._history_has_fill(history, real_candidate_bid(password_box), password)):
                return MiniWoBGroundingResult(action=browsergym_fill_action(real_candidate_bid(password_box), password), selected_candidate=password_box, mapping_strategy="policy_basic_login_password", mapping_diagnostics={"policy_name": "basic_action", "target_field": "password"})
            submit = self._best_by_text(candidates, "login", prefer="button", allow_dom_mouse=False) or find_submit_button(candidates)
            if submit:
                return self._click_result(submit, action_syntax, "policy_basic_login_submit", {"policy_name": "basic_action"})

        quoted = re.findall(r'"([^"]+)"', instr)
        if quoted and any(word in instr_n for word in ("enter", "type")) and "text" in instr_n:
            target_text = quoted[0]
            boxes = [c for c in candidates if self._visible_enabled(c) and self._is_textbox(c) and real_candidate_bid(c)]
            textbox = boxes[0] if boxes else None
            if textbox and not (self._textbox_value_matches(textbox, target_text) or self._history_has_fill(history, real_candidate_bid(textbox), target_text)):
                return MiniWoBGroundingResult(action=browsergym_fill_action(real_candidate_bid(textbox), target_text), selected_candidate=textbox, mapping_strategy="policy_basic_text_fill", mapping_diagnostics={"policy_name": "basic_action", "target_text": target_text})
            submit = self._submit_result(candidates, action_syntax, "policy_basic_text_submit", {"policy_name": "basic_action"})
            if submit:
                return submit

        if "focus" in instr_n and "textbox" in instr_n:
            boxes = [c for c in candidates if self._visible_enabled(c) and self._is_textbox(c) and real_candidate_bid(c)]
            if boxes:
                bid = real_candidate_bid(boxes[0])
                if self._action_supported(action_syntax, "focus"):
                    return MiniWoBGroundingResult(action=f'focus("{bid}")', selected_candidate=boxes[0], mapping_strategy="policy_basic_focus_textbox", mapping_diagnostics={"policy_name": "basic_action"})
                return self._click_result(boxes[0], action_syntax, "policy_basic_focus_textbox", {"policy_name": "basic_action"})

        list_match = re.search(r"select\s+(.+?)\s+from\s+the\s+list\s+and\s+click\s+submit\.?$", instr, flags=re.I)
        if list_match:
            target = list_match.group(1).strip()
            controls = [c for c in candidates if self._visible_enabled(c) and self._is_select_control(c) and real_candidate_bid(c)]
            control = controls[0] if controls else None
            current = self._norm((control or {}).get("value") or (control or {}).get("selected_value") or (control or {}).get("current_value"))
            if control and self._norm(target) not in current:
                return MiniWoBGroundingResult(action=browsergym_select_option_action(real_candidate_bid(control), target, action_syntax), selected_candidate=control, mapping_strategy="policy_basic_select_option", mapping_diagnostics={"policy_name": "basic_action", "target_text": target})
            if control:
                submit = self._submit_result(candidates, action_syntax, "policy_basic_select_submit", {"policy_name": "basic_action", "target_text": target})
                if submit:
                    return submit

        select_targets, select_like = self._parse_select_targets(instr)
        if select_like:
            controls = [c for c in candidates if self._visible_enabled(c) and self._is_select_control(c) and real_candidate_bid(c)]
            if controls and select_targets:
                target = select_targets[0]
                control = next(
                    (
                        c
                        for c in controls
                        if self._norm(target) in self._norm(c.get("text") or c.get("innerText") or c.get("textContent") or "")
                        or not str(c.get("text") or c.get("innerText") or c.get("textContent") or "").strip()
                    ),
                    controls[0],
                )
                control_bid = real_candidate_bid(control)
                current = self._norm(control.get("value") or control.get("selected_value") or control.get("current_value"))
                selected_all = all(self._norm(target) in current for target in select_targets)
                history_all = all(self._history_has_select_option(history, control_bid, target) for target in select_targets)
                if not selected_all and not history_all:
                    option_arg = select_targets if len(select_targets) > 1 else target
                    return MiniWoBGroundingResult(action=browsergym_select_option_action(control_bid, option_arg, action_syntax), selected_candidate=control, mapping_strategy="policy_basic_select_option", mapping_diagnostics={"policy_name": "basic_action", "target_text": option_arg, "select_like": "scroll_or_select"})
                submit = self._submit_result(candidates, action_syntax, "policy_basic_select_submit", {"policy_name": "basic_action", "target_text": select_targets if len(select_targets) > 1 else target, "select_like": "scroll_or_select"})
                if submit:
                    return submit
            clicked = self._history_clicked_texts(history)
            if not select_targets:
                submit = self._submit_result(candidates, action_syntax, "policy_basic_select_submit", {"policy_name": "basic_action", "targets": []})
                if submit:
                    return submit
            for target in select_targets:
                if self._norm(target) in clicked:
                    continue
                prefer = "checkbox" if len(select_targets) > 1 else "radio"
                c = self._best_by_text(candidates, target, prefer=prefer, include_parent=True)
                if c:
                    return self._click_result(c, action_syntax, "policy_basic_select_item", {"policy_name": "basic_action", "target_text": target, "targets": select_targets})
            submit = self._submit_result(candidates, action_syntax, "policy_basic_select_submit", {"policy_name": "basic_action", "targets": select_targets})
            if submit:
                return submit

        link_match = re.search(r'link\s+"([^"]+)"', instr, flags=re.I)
        if link_match:
            target = link_match.group(1).strip()
            link_like = [c for c in candidates if self._visible_enabled(c) and self._is_link_like(c) and not self._is_wrapper(c)]
            c = self._best_by_text(link_like, target, prefer="link") if link_like else None
            if c:
                return self._click_result(c, action_syntax, "policy_basic_link_click", {"policy_name": "basic_action", "target_text": target})
            if "switch between the tabs" not in instr_n and "expand the sections" not in instr_n:
                return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_basic_link_target_not_found", mapping_error="link_target_not_found", mapping_diagnostics={"policy_name": "basic_action", "target_text": target})

        tab_match = re.search(r"click\s+on\s+tab\s+#?(\d+)", instr, flags=re.I)
        if tab_match:
            target = f"Tab #{tab_match.group(1)}"
            c = self._best_by_text(candidates, target, prefer="tab", allow_dom_mouse=False)
            if c:
                return self._click_result(c, action_syntax, "policy_basic_tab_click", {"policy_name": "basic_action", "target_text": target})

        if "switch between the tabs" in instr_n:
            clicked = self._history_clicked_texts(history)
            tabs = [c for c in candidates if self._visible_enabled(c) and "tab #" in self._norm(self.candidate_text(c)) and not self._is_wrapper(c)]
            for tab in tabs:
                if self._norm(self.candidate_text(tab)) not in clicked:
                    return self._click_result(tab, action_syntax, "policy_basic_tab_probe", {"policy_name": "basic_action"})

        if "expand the section below" in instr_n and "click submit" in instr_n:
            if not any(str(h.get("mapping_strategy") or "") == "policy_basic_collapsible_expand" for h in history if isinstance(h, dict)):
                header = next((c for c in candidates if self._visible_enabled(c) and ("ui-accordion-header" in self._norm(c.get("className")) or "section #" in self._norm(self.candidate_text(c))) and not self._is_wrapper(c)), None)
                if header:
                    return self._click_result(header, action_syntax, "policy_basic_collapsible_expand", {"policy_name": "basic_action"})
            submit = self._submit_result(candidates, action_syntax, "policy_basic_collapsible_submit", {"policy_name": "basic_action"})
            if submit:
                return submit

        if "expand the sections" in instr_n:
            clicked = self._history_clicked_texts(history)
            headers = [c for c in candidates if self._visible_enabled(c) and ("ui-accordion-header" in self._norm(c.get("className")) or "section #" in self._norm(self.candidate_text(c))) and not self._is_wrapper(c)]
            for header in headers:
                if self._norm(self.candidate_text(header)) not in clicked:
                    return self._click_result(header, action_syntax, "policy_basic_collapsible_probe", {"policy_name": "basic_action"})

        button_target = ""
        m = re.search(r'click\s+on\s+the\s+"([^"]+)"\s+button', instr, flags=re.I)
        if m:
            button_target = m.group(1).strip()
        elif len(labels) == 1:
            button_target = labels[0].strip()
        if button_target:
            c = self._best_by_text(candidates, button_target, prefer="button", allow_dom_mouse=False)
            if c:
                return self._click_result(c, action_syntax, "policy_basic_button_click", {"policy_name": "basic_action", "target_text": button_target})
        if re.search(r"\bclick\s+the\s+button\.?$", instr, flags=re.I):
            buttons = [c for c in candidates if self._visible_enabled(c) and self._is_buttonish(c) and not self._is_wrapper(c)]
            buttons.sort(key=lambda c: (0 if real_candidate_bid(c) else 1, len(str(self.candidate_text(c) or ""))))
            if buttons:
                return self._click_result(buttons[0], action_syntax, "policy_basic_button_click", {"policy_name": "basic_action", "target_text": self.candidate_text(buttons[0])})
        return None
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

    def _is_empty_generic(self, c: dict[str, Any]) -> bool:
        role = self.candidate_role(c)
        if role != "generic":
            return False
        return not any(self._norm(v) for v in (c.get("text"), c.get("innerText"), c.get("textContent"), c.get("name"), c.get("title"), c.get("href")))

    def _is_date_input(self, c: dict[str, Any]) -> bool:
        cls = self._norm(c.get("className"))
        cid = self._norm(c.get("id"))
        tag = self.candidate_tag(c)
        role = self.candidate_role(c)
        typ = self.candidate_type(c)
        parent_text = self._norm(c.get("parent_text") or c.get("parentText") or c.get("container_text"))
        texts = " ".join(self._candidate_texts(c)).lower()
        return (
            "hasdatepicker" in cls
            or "datepicker" in cid
            or "date" in cid
            or ("date" in parent_text and tag == "input")
            or (tag == "input" and typ in {"text", "date"} and "date" in texts)
            or ("date" in texts and role in {"textbox", "input", "combobox"})
        )

    def _bbox(self, c: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(c, dict):
            return None
        bbox = c.get("bbox") or c.get("browsergym_bbox") or c.get("bounding_box")
        return bbox if isinstance(bbox, dict) else None

    def _infer_browsergym_scale(self, candidates: list[dict[str, Any]]) -> float:
        ratios: list[float] = []
        for c in candidates or []:
            if not isinstance(c, dict):
                continue
            for page_key, bgym_key in (("page_center_x", "browsergym_center_x"), ("page_center_y", "browsergym_center_y")):
                page_v = c.get(page_key)
                bgym_v = c.get(bgym_key)
                if isinstance(page_v, (int, float)) and isinstance(bgym_v, (int, float)) and abs(page_v) > 1e-6:
                    ratio = float(bgym_v) / float(page_v)
                    if 0.25 <= ratio <= 5.0:
                        ratios.append(ratio)
        if ratios:
            ratios.sort()
            return ratios[len(ratios) // 2]
        return 1.5

    def _browsergym_mouse_click(self, page_x: float, page_y: float, scale: float) -> str:
        return f'mouse_click({int(round(page_x * scale))}, {int(round(page_y * scale))}, "left")'

    def _try_grid_coordinate(self, *, instruction: str, candidates: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        match = re.search(r"grid coordinate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(instruction or ""), flags=re.I)
        if not match:
            return None
        target_x = int(match.group(1))
        target_y = int(match.group(2))
        target_id = f"({target_x},{target_y})"
        scale = self._infer_browsergym_scale(candidates)
        point_candidates = [
            c
            for c in candidates or []
            if self._visible_enabled(c)
            and (
                self.candidate_tag(c) == "circle"
                or "plot-point" in self._norm(c.get("className"))
                or re.fullmatch(r"\(-?\d+\s*,\s*-?\d+\)", str(c.get("id") or "").strip())
            )
        ]
        exact_point = next((c for c in point_candidates if str(c.get("id") or "").replace(" ", "") == target_id), None)
        if exact_point:
            bgx = exact_point.get("browsergym_center_x")
            bgy = exact_point.get("browsergym_center_y")
            if not (isinstance(bgx, (int, float)) and isinstance(bgy, (int, float))):
                page_x = exact_point.get("page_center_x") or exact_point.get("center_x")
                page_y = exact_point.get("page_center_y") or exact_point.get("center_y")
                if isinstance(page_x, (int, float)) and isinstance(page_y, (int, float)):
                    bgx = page_x * scale
                    bgy = page_y * scale
            if isinstance(bgx, (int, float)) and isinstance(bgy, (int, float)):
                return MiniWoBGroundingResult(
                    action=f'mouse_click({int(round(bgx))}, {int(round(bgy))}, "left")',
                    selected_candidate=exact_point,
                    mapping_strategy="policy_grid_coordinate_point",
                    mapping_diagnostics={
                        "policy_name": "grid-coordinate",
                        "target_x": target_x,
                        "target_y": target_y,
                        "target_id": target_id,
                        "point_candidates": len(point_candidates),
                        "scale": scale,
                        "click_source": "svg_circle_candidate",
                    },
                )

        svg = next((c for c in candidates or [] if self._visible_enabled(c) and self.candidate_tag(c) == "svg" and self._bbox(c)), None)
        if svg:
            bbox = self._bbox(svg) or {}
            left = float(bbox.get("x", bbox.get("left", 0)) or 0)
            top = float(bbox.get("y", bbox.get("top", 0)) or 0)
            width = float(bbox.get("width") or 0)
            height = float(bbox.get("height") or 0)
            if width > 0 and height > 0 and -2 <= target_x <= 2 and -2 <= target_y <= 2:
                step_x = width / 5.0
                step_y = height / 5.0
                page_x = left + ((target_x + 2) * step_x) + (step_x / 2.0)
                page_y = top + ((2 - target_y) * step_y) + (step_y / 2.0)
                return MiniWoBGroundingResult(
                    action=self._browsergym_mouse_click(page_x, page_y, scale),
                    selected_candidate=svg,
                    mapping_strategy="policy_grid_coordinate_geometry",
                    mapping_diagnostics={
                        "policy_name": "grid-coordinate",
                        "target_x": target_x,
                        "target_y": target_y,
                        "target_id": target_id,
                        "page_x": page_x,
                        "page_y": page_y,
                        "scale": scale,
                        "svg_bbox": bbox,
                        "point_candidates": len(point_candidates),
                        "click_source": "svg_bbox_geometry",
                    },
                )
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_strategy="policy_grid_coordinate_no_geometry",
            mapping_error="grid_coordinate_geometry_not_found",
            mapping_diagnostics={"policy_name": "grid-coordinate", "target_x": target_x, "target_y": target_y, "target_id": target_id, "point_candidates": len(point_candidates)},
        )

    def _float_attr(self, value: Any) -> float | None:
        text = str(unwrap_ax_value(value) or "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _singular_count_target(self, word: str) -> str:
        word = self._norm(word)
        aliases = {
            "circles": "circle",
            "rectangles": "rectangle",
            "triangles": "triangle",
            "letters": "letter",
            "digits": "digit",
            "numbers": "digit",
            "number": "digit",
            "shapes": "shape",
            "items": "item",
        }
        if word in aliases:
            return aliases[word]
        if len(word) == 2 and word.endswith("s") and word[0].isalnum():
            return word[0]
        if len(word) > 3 and word.endswith("s"):
            return word[:-1]
        return word

    def _parse_count_shape_query(self, instruction: str) -> dict[str, str] | None:
        match = re.search(r"how many\s+(.+?)\s+are there\??\s*$", str(instruction or ""), flags=re.I)
        if not match:
            return None
        desc = match.group(1).strip()
        words = re.findall(r"[A-Za-z0-9]+", desc.casefold())
        if not words:
            return None
        colors = {"red", "green", "blue", "aqua", "black", "magenta", "yellow"}
        sizes = {"large", "small"}
        content_words = [w for w in words if w not in colors and w not in sizes]
        target_word = content_words[-1] if content_words else "item"
        return {
            "description": desc,
            "size": next((w for w in words if w in sizes), ""),
            "color": next((w for w in words if w in colors), ""),
            "target": self._singular_count_target(target_word),
        }

    def _svg_shape_size(self, c: dict[str, Any]) -> str:
        tag = self.candidate_tag(c)
        values: list[float] = []
        if tag == "text":
            font_size = self._float_attr(c.get("fontSize"))
            if font_size is not None:
                values.append(font_size)
        for key in ("width_attr", "height_attr"):
            value = self._float_attr(c.get(key))
            if value is not None:
                values.append(value)
        radius = self._float_attr(c.get("r"))
        if radius is not None:
            values.append(radius * 2.0)
        bbox = self._bbox(c) or {}
        for key in ("width", "height"):
            value = self._float_attr(bbox.get(key))
            if value is not None:
                values.append(value)
        if not values:
            return ""
        return "large" if max(values) >= 15.0 else "small"

    def _svg_shape_properties(self, c: dict[str, Any]) -> dict[str, str] | None:
        tag = self.candidate_tag(c)
        if tag not in {"circle", "rect", "polygon", "text"}:
            return None
        if self._norm(c.get("parent_tag")) != "svg":
            return None
        text = str(unwrap_ax_value(c.get("text")) or unwrap_ax_value(c.get("textContent")) or "").strip()
        fill = self._norm(c.get("fill"))
        if fill.startswith("rgb("):
            fill = ""
        if tag == "circle":
            item_type, item_text = "shape", "circle"
        elif tag == "rect":
            item_type, item_text = "shape", "rectangle"
        elif tag == "polygon":
            item_type, item_text = "shape", "triangle"
        else:
            normalized_text = self._norm(text)
            if not normalized_text:
                return None
            item_type = "digit" if normalized_text.isdigit() else "letter"
            item_text = normalized_text
        return {"type": item_type, "text": item_text, "color": fill, "size": self._svg_shape_size(c)}

    def _count_shape_matches(self, props: dict[str, str], query: dict[str, str]) -> bool:
        if query.get("size") and props.get("size") != query["size"]:
            return False
        if query.get("color") and props.get("color") != query["color"]:
            return False
        target = query.get("target") or "item"
        if target == "item":
            return True
        if target in {"shape", "letter", "digit"}:
            return props.get("type") == target
        return props.get("text") == target

    def _try_count_shape(self, *, instruction: str, candidates: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        query = self._parse_count_shape_query(instruction)
        if not query:
            return None
        svg_items: list[tuple[dict[str, Any], dict[str, str]]] = []
        seen: set[tuple[str, str, str, int, int, int, int]] = set()
        for c in candidates or []:
            if not self._visible_enabled(c):
                continue
            props = self._svg_shape_properties(c)
            if not props:
                continue
            bbox = self._bbox(c) or {}
            key = (
                self.candidate_tag(c),
                props.get("text", ""),
                props.get("color", ""),
                int(round(float(bbox.get("left", bbox.get("x", 0)) or 0))),
                int(round(float(bbox.get("top", bbox.get("y", 0)) or 0))),
                int(round(float(bbox.get("width", 0) or 0))),
                int(round(float(bbox.get("height", 0) or 0))),
            )
            if key in seen:
                continue
            seen.add(key)
            svg_items.append((c, props))
        if not svg_items:
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_strategy="policy_count_shape_no_svg_items",
                mapping_error="count_shape_svg_items_not_found",
                mapping_diagnostics={"policy_name": "count-shape", **query},
            )
        count = sum(1 for _, props in svg_items if self._count_shape_matches(props, query))
        answer_text = str(count)
        buttons = [
            c
            for c in candidates or []
            if self._visible_enabled(c)
            and self._is_buttonish(c)
            and self._norm(self.candidate_text(c)) == answer_text
            and not self._is_wrapper(c)
        ]
        buttons.sort(key=lambda c: (0 if real_candidate_bid(c) else 1, len(str(self.candidate_text(c) or ""))))
        answer = buttons[0] if buttons else self._best_by_text(candidates, answer_text, prefer="button", allow_dom_mouse=True)
        diagnostics = {
            "policy_name": "count-shape",
            **query,
            "matched_count": count,
            "svg_items_total": len(svg_items),
            "button_texts": [self.candidate_text(c) for c in candidates or [] if self._visible_enabled(c) and self._is_buttonish(c)],
            "svg_item_props": [props for _, props in svg_items],
        }
        if answer:
            return self._click_result(answer, action_syntax, "policy_count_shape_answer", diagnostics)
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_strategy="policy_count_shape_answer_button_not_found",
            mapping_error="count_shape_answer_button_not_found",
            mapping_diagnostics=diagnostics,
        )

    def try_act(self, *, env_id: str, task_name: str, instruction: str, candidates: list[dict], history: list[dict], action_syntax: list[str]) -> MiniWoBGroundingResult | None:
        t = self._task(env_id, task_name)
        instr = str(instruction or "")
        generic = self._try_basic_action_intent(instruction=instr, candidates=candidates, history=history, action_syntax=action_syntax)
        if generic is not None:
            return generic
        if "count-shape" in t:
            count_result = self._try_count_shape(instruction=instr, candidates=candidates, action_syntax=action_syntax)
            if count_result is not None:
                return count_result
        if "grid-coordinate" in t:
            grid_result = self._try_grid_coordinate(instruction=instr, candidates=candidates, action_syntax=action_syntax)
            if grid_result is not None:
                return grid_result
        if "click-button-sequence" in t:
            labels = re.findall(r"click button\s+([A-Za-z0-9_-]+)", instr, flags=re.I)
            if len(labels) >= 2:
                clicked = {self._norm(h.get("selected_candidate_text") or "") for h in history if isinstance(h, dict)}
                for lbl in labels:
                    if self._norm(lbl) in clicked:
                        continue
                    c = self._find_by_text(candidates, lbl, {"button"})
                    if c and real_candidate_bid(c):
                        return self._safe_button_click_result(c, candidates, action_syntax, "policy_click_button_sequence", {"policy_name": "click-button-sequence"})
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
                links = [c for c in candidates if (self.candidate_tag(c) == "a" or bool(str(c.get("href") or "").strip()) or self.candidate_role(c) == "link" or (c.get("source") == "dom" and self.candidate_tag(c) == "a")) and not self._is_empty_generic(c) and self._link_texts(c)]
                exact = [c for c in links if any(self._norm(v) == target for v in self._link_texts(c))]
                pools = [exact, [c for c in links if any(self._norm(v) == loose for v in self._link_texts(c))], [c for c in links if any(target in self._norm(v) or self._norm(v) in target for v in self._link_texts(c))]]
                for pool in pools:
                    if len(pool) == 1:
                        c = pool[0]
                        action, click_strategy = make_click_action(c, action_syntax)
                        if action:
                            return MiniWoBGroundingResult(action=action, selected_candidate=c, mapping_strategy="policy_click_link", mapping_diagnostics={"policy_name": "click-link", "click_strategy": click_strategy})
                doms = [c for c in candidates if c.get("source") == "dom"]
                return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_click_link_target_not_found", mapping_error="link_target_not_found", mapping_diagnostics={"policy_name": "click-link", "target": raw_target, "candidates_total": len(candidates), "link_candidates_count": len(links), "page_clickable_candidates_count": len(doms), "first_20_candidate_texts": [self.candidate_text(c) for c in candidates[:20]], "first_20_dom_candidate_texts": [self.candidate_text(c) for c in doms[:20]], "dom_candidates_count": len(doms), "ax_candidates_count": len([c for c in candidates if c.get("source") != "dom"])})
        if "use-autocomplete" in t:
            p = re.search(r"starts with\s+['\"]([^'\"]+)['\"]", instr, flags=re.I)
            s = re.search(r"ends with\s+['\"]([^'\"]+)['\"]", instr, flags=re.I)
            prefix = (p.group(1) if p else "").strip()
            suffix = (s.group(1) if s else "").strip()
            submit = next((c for c in candidates if self._norm(self.candidate_text(c)) == "submit"), None)
            suggestions = [c for c in self._extract_autocomplete_suggestions(candidates) if real_candidate_bid(c)]
            chosen_recently = self._has_mapping(history, "policy_use_autocomplete_pick") or any(self._norm(h.get("selected_candidate_role")) in {"option", "listitem", "menuitem"} for h in history[-3:] if isinstance(h, dict))
            if chosen_recently and submit and real_candidate_bid(submit):
                return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_use_autocomplete_submit", mapping_diagnostics={"policy_name": "use-autocomplete"})
            suggestion_texts = [self.candidate_text(c) for c in suggestions if self.candidate_text(c).strip()]
            for c in suggestions:
                vals = [self._norm(v) for v in (self.candidate_text(c), c.get("innerText"), c.get("textContent"), c.get("name")) if str(unwrap_ax_value(v) or "").strip()]
                if any(v.startswith(self._norm(prefix)) and (not suffix or v.endswith(self._norm(suffix))) for v in vals):
                    return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(c), action_syntax=action_syntax), selected_candidate=c, mapping_strategy="policy_use_autocomplete_pick", mapping_diagnostics={"policy_name": "use-autocomplete", "autocomplete_prefix": prefix, "autocomplete_suffix": suffix, "suggestions_count": len(suggestions), "suggestion_texts": suggestion_texts, "selected_suggestion_bid": real_candidate_bid(c), "repeated_fill_blocked": True})
            textbox = next((c for c in candidates if self._is_textbox(c) and ("tags" in " ".join(self._candidate_texts(c)).lower()) and real_candidate_bid(c)), None) or next((c for c in candidates if self._is_textbox(c) and real_candidate_bid(c)), None)
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
            submit = next((c for c in candidates if self._norm(self.candidate_text(c)) == "submit"), None)
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", instr)
            open_attempts = sum(1 for h in history if isinstance(h, dict) and str(h.get("mapping_strategy") or "") == "policy_choose_date_open")
            if open_attempts >= 2 and not any("ui-datepicker" in self._norm(" ".join(self._candidate_texts(c)) + " " + str(c.get("className") or "")) for c in candidates):
                return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_no_datepicker", mapping_error="datepicker_not_opened", mapping_diagnostics={"policy_name": "choose-date", "datepicker_visible": False, "strategy": "policy_choose_date_no_datepicker"})
            if (self._has_mapping(history, "policy_choose_date_day") or any(str((h.get("selected_candidate_text") or "")).isdigit() for h in history[-2:] if isinstance(h, dict))) and submit and m:
                target_date = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{int(m.group(3)):04d}"
                date_inputs = [c for c in candidates if real_candidate_bid(c) and self._is_date_input(c)]
                tbs = [c for c in candidates if self.candidate_role(c) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
                date_tb = (date_inputs[0] if date_inputs else None) or (tbs[0] if len(tbs) == 1 else None)
                tb_value = self._norm((date_tb or {}).get("value") or (date_tb or {}).get("text") or "")
                if target_date.lower() in tb_value or self._has_mapping(history, "policy_choose_date_fill"):
                    action, click_strategy = make_click_action(submit, action_syntax)
                    if action:
                        return MiniWoBGroundingResult(action=action, selected_candidate=submit, mapping_strategy="policy_choose_date_submit", mapping_diagnostics={"policy_name": "choose-date", "chosen_stage": "submit", "target_date": target_date, "textbox_value": tb_value, "click_strategy": click_strategy})
                return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_invalid_state", mapping_error="datepicker_header_not_found", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date, "textbox_value": tb_value})
            if m:
                day = str(int(m.group(2)))
                month = int(m.group(1))
                year = int(m.group(3))
                header = " ".join(" ".join(self._candidate_texts(c)) for c in candidates).lower()
                if str(year) in header:
                    month_words = {1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june", 7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december"}
                    prev_btn = next((c for c in candidates if ("prev" in " ".join(self._candidate_texts(c)).lower() or "ui-datepicker-prev" in self._norm(c.get("className")))), None)
                    next_btn = next((c for c in candidates if ("next" in " ".join(self._candidate_texts(c)).lower() or "ui-datepicker-next" in self._norm(c.get("className")))), None)
                    current_month = next((mno for mno, mname in month_words.items() if mname in header), None)
                    if current_month and current_month != month:
                        btn = next_btn if current_month < month else prev_btn
                        if btn:
                            strategy = "policy_choose_date_next_month" if current_month < month else "policy_choose_date_prev_month"
                            action, click_strategy = make_click_action(btn, action_syntax)
                            if action:
                                return MiniWoBGroundingResult(action=action, selected_candidate=btn, mapping_strategy=strategy, mapping_diagnostics={"policy_name": "choose-date", "chosen_stage": ("next" if current_month < month else "prev"), "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header, "current_month": current_month, "current_year": year, "click_strategy": click_strategy})
                header_candidates = [c for c in candidates if "ui-datepicker-title" in self._norm(c.get("className")) or "ui-datepicker-month" in self._norm(c.get("className")) or "ui-datepicker-year" in self._norm(c.get("className"))]
                day_candidates = [
                    c
                    for c in candidates
                    if day in {self._norm(t).lstrip("0") for t in self._candidate_texts(c) if self._norm(t).isdigit()}
                    and ("ui-state-default" in self._norm(c.get("className")) or self.candidate_tag(c) in {"a", "button"})
                    and "ui-priority-secondary" not in self._norm(c.get("className"))
                    and "other-month" not in self._norm(c.get("className"))
                    and c.get("visible") is not False
                ]
                if not day_candidates and header_candidates and str(year) not in header:
                    return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_header_not_found", mapping_error="datepicker_header_not_found", mapping_diagnostics={"policy_name": "choose-date", "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header, "datepicker_header_not_found": True})
                day_c = day_candidates[0] if day_candidates else None
                if day_c:
                    action, click_strategy = make_click_action(day_c, action_syntax)
                    if action:
                        return MiniWoBGroundingResult(action=action, selected_candidate=day_c, mapping_strategy="policy_choose_date_day", mapping_diagnostics={"policy_name": "choose-date", "chosen_stage": "day", "target_date": f"{month:02d}/{int(day):02d}/{year}", "datepicker_header_text": header, "click_strategy": click_strategy})
                if m and any("ui-datepicker" in self._norm(" ".join(self._candidate_texts(c)) + " " + str(c.get("className") or "")) for c in candidates):
                    return MiniWoBGroundingResult(action="noop()", mapping_strategy="policy_choose_date_day_not_found", mapping_error="datepicker_day_not_found", mapping_diagnostics={"policy_name": "choose-date", "target_date": f"{month:02d}/{int(day):02d}/{year}", "day_candidates": [self.candidate_text(c) for c in day_candidates], "chosen_day": day, "datepicker_visible": True})
            date_inputs = [c for c in candidates if real_candidate_bid(c) and self._is_date_input(c)]
            tbs = [c for c in candidates if self.candidate_role(c) in {"textbox", "input", "combobox"} and real_candidate_bid(c)]
            date_tb = (date_inputs[0] if date_inputs else None) or (tbs[0] if len(tbs) == 1 else None)
            if date_tb:
                fill_attempts = sum(1 for h in history if isinstance(h, dict) and str(h.get("mapping_strategy") or "") == "policy_choose_date_fill")
                if False and m and self._action_supported(action_syntax, "fill"):
                    target_date = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{int(m.group(3)):04d}"
                    current_value = self._norm(date_tb.get("value") or date_tb.get("text") or "")
                    if target_date.lower() in current_value:
                        if submit and real_candidate_bid(submit):
                            return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(submit), action_syntax=action_syntax), selected_candidate=submit, mapping_strategy="policy_choose_date_submit", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date, "textbox_value_current": current_value, "fill_attempt_count": fill_attempts})
                    if fill_attempts >= 1:
                        return MiniWoBGroundingResult(action="noop()", selected_candidate=date_tb, mapping_strategy="policy_choose_date_fill_no_progress", mapping_error="choose_date_fill_no_progress", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date, "textbox_value_current": current_value, "fill_attempt_count": fill_attempts, "choose_date_fill_no_progress": True})
                    return MiniWoBGroundingResult(action=f'fill("{real_candidate_bid(date_tb)}", "{target_date}")', selected_candidate=date_tb, mapping_strategy="policy_choose_date_fill", mapping_diagnostics={"policy_name": "choose-date", "target_date": target_date})
                action, click_strategy = make_click_action(date_tb, action_syntax)
                if action:
                    return MiniWoBGroundingResult(action=action, selected_candidate=date_tb, mapping_strategy="policy_choose_date_open", mapping_diagnostics={"policy_name": "choose-date", "chosen_stage": "open", "date_input_bid": real_candidate_bid(date_tb), "datepicker_visible": any("ui-datepicker" in self._norm(" ".join(self._candidate_texts(c)) + " " + str(c.get("className") or "")) for c in candidates), "strategy": "policy_choose_date_open", "click_strategy": click_strategy})
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
                if any(h.get("mapping_strategy") == "policy_book_flight_search" for h in history if isinstance(h, dict)):
                    picks = [c for c in candidates if any(k in self._norm(self.candidate_text(c)) for k in ["select", "book"]) ]
                    if len(picks) == 1:
                        action, click_strategy = make_click_action(picks[0], action_syntax)
                        if action:
                            return MiniWoBGroundingResult(action=action, selected_candidate=picks[0], mapping_strategy="policy_book_flight_result_pick", mapping_diagnostics={"policy_name": "book-flight", "click_strategy": click_strategy})
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
                    def menu_item(label: str) -> dict[str, Any] | None:
                        label_n = self._norm(label)
                        exact = [
                            c
                            for c in candidates
                            if self._visible_enabled(c)
                            and real_candidate_bid(c)
                            and self._norm(self.candidate_text(c)) == label_n
                            and (self.candidate_role(c) == "menuitem" or "ui-menu-item-wrapper" in self._norm(c.get("className")))
                        ]
                        return exact[0] if exact else self._find_by_text(candidates, label)

                    final = menu_item(parts[-1])
                    if final and real_candidate_bid(final):
                        return MiniWoBGroundingResult(action=browsergym_click_action(real_candidate_bid(final), action_syntax=action_syntax), selected_candidate=final, mapping_strategy="policy_click_menu_leaf", mapping_diagnostics={"policy_name": "click-menu"})
                    hovered_labels = {
                        self._norm(h.get("selected_candidate_text") or "")
                        for h in history
                        if isinstance(h, dict) and str(h.get("mapping_strategy") or "") == "policy_click_menu_hover_parent"
                    }
                    parent_label = next((part for part in parts[:-1] if self._norm(part) not in hovered_labels), parts[0])
                    parent = menu_item(parent_label)
                    if parent and self._has_mapping(history, "policy_click_menu_hover_parent") and self._norm(parent_label) == self._norm(parts[0]):
                        return MiniWoBGroundingResult(action="noop()", selected_candidate=parent, mapping_strategy="policy_click_menu_hover_required", mapping_error="menu_requires_hover_no_supported_action", mapping_diagnostics={"policy_name": "click-menu", "menu_requires_hover_no_supported_action": True, "hover_attempted": True, "next_label": parent_label})
                    if parent and real_candidate_bid(parent) and self._action_supported(action_syntax, "hover"):
                        return MiniWoBGroundingResult(action=f'hover("{real_candidate_bid(parent)}")', selected_candidate=parent, mapping_strategy="policy_click_menu_hover_parent", mapping_diagnostics={"policy_name": "click-menu", "next_label": parent_label})
                    if parent and (self._action_supported(action_syntax, "mouse_move") or self._action_supported(action_syntax, "move_to")):
                        x = parent.get("browsergym_center_x") or parent.get("center_x")
                        y = parent.get("browsergym_center_y") or parent.get("center_y")
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            return MiniWoBGroundingResult(action=f"mouse_move({int(x)}, {int(y)})", selected_candidate=parent, mapping_strategy="policy_click_menu_hover_parent", mapping_diagnostics={"policy_name": "click-menu", "next_label": parent_label})
                    return MiniWoBGroundingResult(action="noop()", selected_candidate=parent, mapping_strategy="policy_click_menu_hover_required", mapping_error="menu_requires_hover_no_supported_action", mapping_diagnostics={"policy_name": "click-menu", "menu_requires_hover_no_supported_action": True, "next_label": parent_label})
        return None
