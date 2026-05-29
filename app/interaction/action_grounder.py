from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re


_INPUT_KINDS = {"textbox", "date"}
_CLICK_KINDS = {"button", "link", "clickable", "menuitem", "option", "radio", "checkbox"}
_SUBMIT_WORDS = {"submit", "send", "go", "ok", "continue", "login", "log in", "sign in"}


@dataclass(slots=True)
class GroundedAction:
    action: str
    args: dict[str, Any]


@dataclass(slots=True)
class GroundingResult:
    actions: list[GroundedAction]
    selected_candidate: dict[str, Any] | None = None
    grounding_strategy: str = "unresolved"
    confidence: float = 0.0
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def first_action_dict(self) -> dict[str, Any]:
        if not self.actions:
            raise ValueError("No grounded action available")
        action = self.actions[0]
        return {"action": action.action, "args": dict(action.args)}


class ActionGrounder:
    """Map high-level or LLM-proposed interaction intents to executable Playwright actions."""

    def ground(
        self,
        requested_action: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        user_goal: str = "",
        page_snapshot: dict[str, Any] | str | None = None,
    ) -> GroundingResult:
        payload = dict(requested_action or {})
        intent = self._normalize_intent(str(payload.get("intent") or payload.get("action") or payload.get("type") or "click"))
        if intent == "type":
            intent = "fill"
        if self._looks_like_login(payload, user_goal=user_goal):
            login = self._ground_login(payload, candidates)
            if login.actions:
                return login
        if intent in {"fill", "select_autocomplete", "choose_date"}:
            return self._ground_fill_like(intent, payload, candidates)
        if intent in {"check", "uncheck"}:
            return self._ground_check(intent, payload, candidates)
        if intent == "select_option":
            return self._ground_select_option(payload, candidates)
        if intent in {"click", "hover", "focus"}:
            return self._ground_target_action(intent, payload, candidates)
        if intent in {"press", "clear"}:
            return self._ground_press_or_clear(intent, payload, candidates)
        if intent == "finish":
            return GroundingResult(actions=[GroundedAction("finish", {})], grounding_strategy="finish", confidence=1.0)
        raise ValueError(f"Unsupported grounding intent: {intent}")

    @staticmethod
    def _normalize_intent(value: str) -> str:
        normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
        aliases = {"enter_text": "fill", "input_text": "fill", "write": "fill", "select": "select_option", "choose_list": "select_option", "autocomplete": "select_autocomplete"}
        return aliases.get(normalized, normalized)

    def _ground_fill_like(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        value = payload.get("value", payload.get("text", payload.get("query", payload.get("date", ""))))
        if intent == "choose_date":
            value = payload.get("date", value)
        if value is None or str(value) == "":
            raise ValueError(f"{intent} requires text/value/query/date")
        wanted = self._target_tokens(payload)
        preferred = ["date"] if intent == "choose_date" else ["textbox", "date"]
        selected, rejected, confidence = self._select_candidate(candidates, wanted, preferred_kinds=preferred)
        if not selected and self._looks_like_search_target(wanted):
            selected, search_rejected, confidence = self._select_search_field(candidates, preferred_kinds=preferred)
            rejected.extend(search_rejected)
        if not selected:
            raise ValueError(f"Unable to ground {intent}: no textbox/date candidate matched target={wanted!r}")
        action = "fill"
        args = {"selector": selected["selector"], "text": str(value), "candidate_id": selected.get("candidate_id")}
        actions = [GroundedAction(action, args)]
        if intent == "select_autocomplete":
            suggestion = payload.get("suggestion_target") or payload.get("suggestion") or payload.get("target_text") or payload.get("option_text") or payload.get("value")
            if suggestion:
                suggestion_candidate, suggestion_rejected, suggestion_conf = self._select_candidate(candidates, [str(suggestion)], preferred_kinds=["option", "button", "clickable"])
                rejected.extend(suggestion_rejected)
                if suggestion_candidate and suggestion_candidate.get("candidate_id") != selected.get("candidate_id"):
                    actions.append(GroundedAction("click", {"selector": suggestion_candidate["selector"], "candidate_id": suggestion_candidate.get("candidate_id")}))
                    confidence = min(confidence, max(suggestion_conf, 0.5))
        return GroundingResult(actions=actions, selected_candidate=selected, grounding_strategy=f"{intent}_candidate", confidence=confidence, rejected_candidates=rejected)

    def _ground_login(self, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        textboxes = [c for c in candidates if c.get("kind") in _INPUT_KINDS and c.get("enabled", True)]
        if len(textboxes) < 2:
            return GroundingResult(actions=[], grounding_strategy="login_not_enough_inputs")
        username = payload.get("username") or payload.get("user") or payload.get("login") or payload.get("text") or payload.get("value") or ""
        password = payload.get("password") or ""
        if not username or not password:
            return GroundingResult(actions=[], grounding_strategy="login_missing_credentials")
        submit, rejected, conf = self._select_candidate(candidates, ["login", "log in", "sign in", "submit"], preferred_kinds=["button", "clickable"])
        if not submit:
            raise ValueError("Login grounding refused: no explicit Login/Sign in/Submit candidate found")
        actions = [
            GroundedAction("fill", {"selector": textboxes[0]["selector"], "text": str(username), "candidate_id": textboxes[0].get("candidate_id")}),
            GroundedAction("fill", {"selector": textboxes[1]["selector"], "text": str(password), "candidate_id": textboxes[1].get("candidate_id")}),
            GroundedAction("click", {"selector": submit["selector"], "candidate_id": submit.get("candidate_id")}),
        ]
        return GroundingResult(actions=actions, selected_candidate=submit, grounding_strategy="login_form_sequence", confidence=min(conf, 0.9), rejected_candidates=rejected)

    def _ground_check(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        selected, rejected, confidence = self._select_candidate(candidates, self._target_tokens(payload), preferred_kinds=["checkbox", "radio"])
        if not selected:
            raise ValueError(f"Unable to ground {intent}: no checkbox/radio matched")
        action = "check" if intent == "check" else "uncheck"
        return GroundingResult(actions=[GroundedAction(action, {"selector": selected["selector"], "candidate_id": selected.get("candidate_id")})], selected_candidate=selected, grounding_strategy=f"{action}_candidate", confidence=confidence, rejected_candidates=rejected)

    def _ground_select_option(self, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        option_value = payload.get("option_text") or payload.get("option_value") or payload.get("value") or payload.get("target_text")
        if not option_value:
            raise ValueError("select_option requires option_text/value")
        selected, rejected, confidence = self._select_candidate(candidates, self._target_tokens(payload), preferred_kinds=["select"])
        if selected:
            return GroundingResult(actions=[GroundedAction("select_option", {"selector": selected["selector"], "option_text": str(option_value), "candidate_id": selected.get("candidate_id")})], selected_candidate=selected, grounding_strategy="select_element_option", confidence=confidence, rejected_candidates=rejected)
        select_candidates = [c for c in candidates if c.get("kind") == "select" and c.get("enabled", True)]
        if not self._target_tokens(payload) and len(select_candidates) == 1:
            selected = select_candidates[0]
            return GroundingResult(actions=[GroundedAction("select_option", {"selector": selected["selector"], "option_text": str(option_value), "candidate_id": selected.get("candidate_id")})], selected_candidate=selected, grounding_strategy="single_select_element_option", confidence=0.72, rejected_candidates=rejected)
        option, more_rejected, option_conf = self._select_candidate(candidates, [str(option_value)], preferred_kinds=["option", "radio"])
        rejected.extend(more_rejected)
        if option:
            return GroundingResult(actions=[GroundedAction("click", {"selector": option["selector"], "candidate_id": option.get("candidate_id")})], selected_candidate=option, grounding_strategy="click_option_candidate", confidence=option_conf, rejected_candidates=rejected)
        raise ValueError("Unable to ground select_option: no select or option candidate matched")

    def _ground_target_action(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        target_tokens = self._target_tokens(payload)
        selected, rejected, confidence = self._select_candidate(candidates, target_tokens, preferred_kinds=list(_CLICK_KINDS) if intent in {"click", "hover"} else None)
        if not selected and intent == "click" and self._looks_like_search_target(target_tokens):
            selected, search_rejected, confidence = self._select_search_submit(candidates)
            rejected.extend(search_rejected)
        if not selected:
            raise ValueError(f"Unable to ground {intent}: unknown target; refusing Submit fallback")
        if intent == "click" and selected.get("kind") in _INPUT_KINDS and (payload.get("value") or payload.get("text")):
            raise ValueError("Refusing to click textbox for text-input intent; use fill")
        return GroundingResult(actions=[GroundedAction(intent, {"selector": selected["selector"], "candidate_id": selected.get("candidate_id")})], selected_candidate=selected, grounding_strategy=f"{intent}_candidate", confidence=confidence, rejected_candidates=rejected)

    def _ground_press_or_clear(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        selected = None
        rejected: list[dict[str, Any]] = []
        confidence = 1.0
        if self._target_tokens(payload):
            selected, rejected, confidence = self._select_candidate(candidates, self._target_tokens(payload), preferred_kinds=["textbox", "date", "select"])
            if not selected:
                raise ValueError(f"Unable to ground {intent}: target not found")
        args = {"key": payload.get("key", "Enter")} if intent == "press" else {}
        if selected:
            args["selector"] = selected["selector"]
            args["candidate_id"] = selected.get("candidate_id")
        return GroundingResult(actions=[GroundedAction(intent, args)], selected_candidate=selected, grounding_strategy=f"{intent}_candidate" if selected else f"page_{intent}", confidence=confidence, rejected_candidates=rejected)

    @staticmethod
    def _target_tokens(payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("candidate_id", "selector", "target", "target_text", "label", "name", "placeholder", "href", "text"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                values.append(str(value).strip())
                simplified = ActionGrounder._strip_generic_target_words(str(value))
                if simplified and simplified not in values:
                    values.append(simplified)
        return values

    @staticmethod
    def _strip_generic_target_words(value: str) -> str:
        words = re.findall(r"[\wА-Яа-яЁё]+", value, flags=re.UNICODE)
        generic = {
            "field",
            "input",
            "textbox",
            "text",
            "box",
            "button",
            "link",
            "control",
            "поле",
            "ввода",
            "кнопка",
            "кнопку",
            "ссылка",
            "ссылку",
        }
        kept = [word for word in words if word.casefold() not in generic]
        return " ".join(kept).strip()

    @staticmethod
    def _looks_like_login(payload: dict[str, Any], *, user_goal: str) -> bool:
        text = " ".join(str(payload.get(k, "")) for k in ("intent", "action", "target", "target_text")) + " " + user_goal
        return bool(re.search(r"\b(log\s*in|sign\s*in|username|password)\b", text, flags=re.I)) and bool(payload.get("password"))

    def _select_candidate(self, candidates: list[dict[str, Any]], tokens: list[str], *, preferred_kinds: list[str] | None = None) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
        rejected: list[dict[str, Any]] = []
        normalized_tokens = [self._norm(t) for t in tokens if str(t).strip()]
        scored: list[tuple[int, int, dict[str, Any]]] = []
        input_candidates = [c for c in candidates if (not preferred_kinds or c.get("kind") in preferred_kinds)]
        for index, candidate in enumerate(candidates):
            if preferred_kinds and candidate.get("kind") not in preferred_kinds:
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "kind_mismatch", "kind": candidate.get("kind")})
                continue
            if self._is_skip_link_candidate(candidate):
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "skip_link_candidate", "kind": candidate.get("kind")})
                continue
            if not candidate.get("enabled", True):
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "disabled"})
                continue
            if normalized_tokens:
                parts = self._candidate_parts(candidate)
                hay = " ".join(parts)
                raw_token_hit = any(str(candidate.get(key, "")).strip() in tokens for key in ("candidate_id", "selector"))
                if raw_token_hit:
                    scored.append((120, -index, candidate))
                    continue
                exact = any(tok and tok == part for tok in normalized_tokens for part in parts)
                contains = any(tok and tok in hay for tok in normalized_tokens)
                if exact:
                    scored.append((100, -index, candidate))
                elif contains:
                    scored.append((70, -index, candidate))
                else:
                    rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "text_mismatch", "kind": candidate.get("kind")})
                continue
            if preferred_kinds and set(preferred_kinds).issubset(_INPUT_KINDS) and len(input_candidates) == 1:
                return candidate, rejected, 0.55
            label = self._norm(candidate.get("text") or candidate.get("aria_label") or candidate.get("value") or "")
            if label in _SUBMIT_WORDS:
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "refused_submit_fallback"})
                continue
        if scored:
            score, _neg_index, candidate = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
            return candidate, rejected, 0.98 if score >= 120 else 0.94 if score >= 100 else 0.78
        return None, rejected, 0.0

    def _candidate_parts(self, candidate: dict[str, Any]) -> list[str]:
        return [
            self._norm(candidate.get(key, ""))
            for key in (
                "candidate_id",
                "selector",
                "text",
                "inner_text",
                "aria_label",
                "title",
                "name",
                "placeholder",
                "value",
                "href",
                "id",
                "className",
                "input_type",
                "role",
            )
            if candidate.get(key)
        ]

    def _candidate_haystack(self, candidate: dict[str, Any]) -> str:
        return " ".join(self._candidate_parts(candidate))

    def _is_skip_link_candidate(self, candidate: dict[str, Any]) -> bool:
        text = self._norm(candidate.get("text") or candidate.get("inner_text") or candidate.get("aria_label") or "")
        href = self._norm(candidate.get("href") or "")
        return text.startswith("skip to ") or bool(re.fullmatch(r"#(?:main|content|search|navigation|nav)", href))

    @classmethod
    def _looks_like_search_target(cls, tokens: list[str]) -> bool:
        hay = " ".join(cls._norm(token) for token in tokens)
        return any(token in hay for token in ("search", "query", "find", "поиск", "найди", "искать"))

    def _select_search_field(self, candidates: list[dict[str, Any]], *, preferred_kinds: list[str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") not in preferred_kinds:
                continue
            if not candidate.get("enabled", True):
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "disabled"})
                continue
            hay = self._candidate_haystack(candidate)
            score = 0
            if "search" in hay or "query" in hay or "поиск" in hay:
                score += 70
            if candidate.get("input_type") == "search":
                score += 60
            if self._norm(candidate.get("name")) in {"q", "query", "search"}:
                score += 30
            if score:
                scored.append((score, -index, candidate))
            else:
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "search_field_mismatch", "kind": candidate.get("kind")})
        if scored:
            score, _neg_index, candidate = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
            return candidate, rejected, 0.9 if score >= 100 else 0.72
        return None, rejected, 0.0

    def _select_search_submit(self, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
        scored: list[tuple[int, int, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") not in {"button", "clickable"}:
                continue
            if not candidate.get("enabled", True):
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "disabled"})
                continue
            hay = self._candidate_haystack(candidate)
            score = 0
            if any(token in hay for token in ("search", "find", "поиск")):
                score += 90
            if any(token == self._norm(candidate.get("value")) for token in ("go", "search", "find")):
                score += 45
            if self._norm(candidate.get("text") or candidate.get("name") or candidate.get("aria_label")) in _SUBMIT_WORDS:
                score += 35
            if candidate.get("input_type") == "submit":
                score += 25
            if score:
                scored.append((score, -index, candidate))
            else:
                rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "search_submit_mismatch", "kind": candidate.get("kind")})
        if scored:
            score, _neg_index, candidate = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
            return candidate, rejected, 0.88 if score >= 90 else 0.68
        return None, rejected, 0.0

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())
