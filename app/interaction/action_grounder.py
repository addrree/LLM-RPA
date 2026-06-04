from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


_INPUT_KINDS = {"textbox", "date"}
_CLICK_KINDS = {"button", "link", "clickable", "menuitem", "option", "radio", "checkbox"}


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
    """Map structural interaction intents to executable Playwright actions."""

    def ground(
        self,
        requested_action: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        user_goal: str = "",
        page_snapshot: dict[str, Any] | str | None = None,
    ) -> GroundingResult:
        _ = user_goal, page_snapshot
        payload = dict(requested_action or {})
        intent = self._normalize_intent(str(payload.get("intent") or payload.get("action") or payload.get("type") or "click"))
        if intent == "type":
            intent = "fill"
        if self._field_values(payload):
            return self._ground_multi_field(payload, candidates)
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
        aliases = {
            "enter_text": "fill",
            "input_text": "fill",
            "write": "fill",
            "select": "select_option",
            "choose_list": "select_option",
            "autocomplete": "select_autocomplete",
        }
        return aliases.get(normalized, normalized)

    def _ground_multi_field(self, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        actions: list[GroundedAction] = []
        rejected: list[dict[str, Any]] = []
        confidences: list[float] = []
        selected_ids: set[str] = set()
        for target, value in self._field_values(payload):
            available = [
                candidate
                for candidate in candidates
                if str(candidate.get("candidate_id") or "") not in selected_ids
            ]
            selected, field_rejected, confidence = self._select_candidate(
                available,
                [target],
                preferred_kinds=["textbox", "date"],
            )
            rejected.extend(field_rejected)
            if not selected:
                raise ValueError(f"Unable to ground multi-field value: no input matched target={target!r}")
            selected_ids.add(str(selected.get("candidate_id") or ""))
            confidences.append(confidence)
            actions.append(
                GroundedAction(
                    "fill",
                    {
                        "selector": selected["selector"],
                        "text": str(value),
                        "candidate_id": selected.get("candidate_id"),
                    },
                )
            )

        completion_target = str(payload.get("completion_target") or "").strip()
        selected_candidate = None
        if completion_target:
            selected_candidate, completion_rejected, confidence = self._select_candidate(
                candidates,
                [completion_target],
                preferred_kinds=["button", "clickable", "link"],
            )
            rejected.extend(completion_rejected)
            if not selected_candidate:
                raise ValueError(f"Unable to ground multi-field completion target={completion_target!r}")
            confidences.append(confidence)
            actions.append(
                GroundedAction(
                    "click",
                    {
                        "selector": selected_candidate["selector"],
                        "candidate_id": selected_candidate.get("candidate_id"),
                    },
                )
            )
        return GroundingResult(
            actions=actions,
            selected_candidate=selected_candidate,
            grounding_strategy="multi_field_sequence",
            confidence=min(confidences) if confidences else 0.0,
            rejected_candidates=rejected,
        )

    def _ground_fill_like(
        self,
        intent: str,
        payload: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> GroundingResult:
        value = payload.get("value", payload.get("text", payload.get("query", payload.get("date", ""))))
        if intent == "choose_date":
            value = payload.get("date", value)
        if value is None or str(value) == "":
            raise ValueError(f"{intent} requires text/value/query/date")
        preferred = ["date"] if intent == "choose_date" else ["textbox", "date"]
        selected, rejected, confidence = self._select_candidate(
            candidates,
            self._target_tokens(payload),
            preferred_kinds=preferred,
        )
        if not selected:
            raise ValueError(f"Unable to ground {intent}: no textbox/date candidate matched")
        actions = [
            GroundedAction(
                "fill",
                {
                    "selector": selected["selector"],
                    "text": str(value),
                    "candidate_id": selected.get("candidate_id"),
                },
            )
        ]
        if intent == "select_autocomplete":
            suggestion = (
                payload.get("suggestion_target")
                or payload.get("suggestion")
                or payload.get("target_text")
                or payload.get("option_text")
                or payload.get("value")
            )
            if suggestion:
                suggestion_candidate, suggestion_rejected, suggestion_confidence = self._select_candidate(
                    candidates,
                    [str(suggestion)],
                    preferred_kinds=["option", "button", "clickable"],
                )
                rejected.extend(suggestion_rejected)
                if suggestion_candidate and suggestion_candidate.get("candidate_id") != selected.get("candidate_id"):
                    actions.append(
                        GroundedAction(
                            "click",
                            {
                                "selector": suggestion_candidate["selector"],
                                "candidate_id": suggestion_candidate.get("candidate_id"),
                            },
                        )
                    )
                    confidence = min(confidence, max(suggestion_confidence, 0.5))
        return GroundingResult(
            actions=actions,
            selected_candidate=selected,
            grounding_strategy=f"{intent}_candidate",
            confidence=confidence,
            rejected_candidates=rejected,
        )

    def _ground_check(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        selected, rejected, confidence = self._select_candidate(
            candidates,
            self._target_tokens(payload),
            preferred_kinds=["checkbox", "radio"],
        )
        if not selected:
            raise ValueError(f"Unable to ground {intent}: no checkbox/radio matched")
        action = "check" if intent == "check" else "uncheck"
        return GroundingResult(
            actions=[GroundedAction(action, {"selector": selected["selector"], "candidate_id": selected.get("candidate_id")})],
            selected_candidate=selected,
            grounding_strategy=f"{action}_candidate",
            confidence=confidence,
            rejected_candidates=rejected,
        )

    def _ground_select_option(self, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        option_value = payload.get("option_text") or payload.get("option_value") or payload.get("value") or payload.get("target_text")
        if not option_value:
            raise ValueError("select_option requires option_text/value")
        tokens = self._target_tokens(payload)
        selected, rejected, confidence = self._select_candidate(candidates, tokens, preferred_kinds=["select"])
        if selected:
            return GroundingResult(
                actions=[
                    GroundedAction(
                        "select_option",
                        {
                            "selector": selected["selector"],
                            "option_text": str(option_value),
                            "candidate_id": selected.get("candidate_id"),
                        },
                    )
                ],
                selected_candidate=selected,
                grounding_strategy="select_element_option",
                confidence=confidence,
                rejected_candidates=rejected,
            )
        select_candidates = [candidate for candidate in candidates if candidate.get("kind") == "select" and candidate.get("enabled", True)]
        if not tokens and len(select_candidates) == 1:
            selected = select_candidates[0]
            return GroundingResult(
                actions=[
                    GroundedAction(
                        "select_option",
                        {
                            "selector": selected["selector"],
                            "option_text": str(option_value),
                            "candidate_id": selected.get("candidate_id"),
                        },
                    )
                ],
                selected_candidate=selected,
                grounding_strategy="single_select_element_option",
                confidence=0.72,
                rejected_candidates=rejected,
            )
        option, more_rejected, option_confidence = self._select_candidate(
            candidates,
            [str(option_value)],
            preferred_kinds=["option", "radio"],
        )
        rejected.extend(more_rejected)
        if option:
            return GroundingResult(
                actions=[GroundedAction("click", {"selector": option["selector"], "candidate_id": option.get("candidate_id")})],
                selected_candidate=option,
                grounding_strategy="click_option_candidate",
                confidence=option_confidence,
                rejected_candidates=rejected,
            )
        raise ValueError("Unable to ground select_option: no select or option candidate matched")

    def _ground_target_action(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        selected, rejected, confidence = self._select_candidate(
            candidates,
            self._target_tokens(payload),
            preferred_kinds=list(_CLICK_KINDS) if intent in {"click", "hover"} else None,
        )
        if not selected:
            raise ValueError(f"Unable to ground {intent}: unknown target; refusing implicit fallback")
        if intent == "click" and selected.get("kind") in _INPUT_KINDS and (payload.get("value") or payload.get("text")):
            raise ValueError("Refusing to click textbox for text-input intent; use fill")
        return GroundingResult(
            actions=[GroundedAction(intent, {"selector": selected["selector"], "candidate_id": selected.get("candidate_id")})],
            selected_candidate=selected,
            grounding_strategy=f"{intent}_candidate",
            confidence=confidence,
            rejected_candidates=rejected,
        )

    def _ground_press_or_clear(self, intent: str, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> GroundingResult:
        selected = None
        rejected: list[dict[str, Any]] = []
        confidence = 1.0
        if self._target_tokens(payload):
            selected, rejected, confidence = self._select_candidate(
                candidates,
                self._target_tokens(payload),
                preferred_kinds=["textbox", "date", "select"],
            )
            if not selected:
                raise ValueError(f"Unable to ground {intent}: target not found")
        args = {"key": payload.get("key", "Enter")} if intent == "press" else {}
        if selected:
            args["selector"] = selected["selector"]
            args["candidate_id"] = selected.get("candidate_id")
        return GroundingResult(
            actions=[GroundedAction(intent, args)],
            selected_candidate=selected,
            grounding_strategy=f"{intent}_candidate" if selected else f"page_{intent}",
            confidence=confidence,
            rejected_candidates=rejected,
        )

    @staticmethod
    def _field_values(payload: dict[str, Any]) -> list[tuple[str, Any]]:
        raw = payload.get("fields") or payload.get("field_values")
        if isinstance(raw, dict):
            return [
                (str(target), value)
                for target, value in raw.items()
                if str(target).strip() and value is not None
            ]
        if isinstance(raw, list):
            return [
                (str(item.get("target") or item.get("name") or item.get("label")), item.get("value"))
                for item in raw
                if isinstance(item, dict)
                and str(item.get("target") or item.get("name") or item.get("label") or "").strip()
                and item.get("value") is not None
            ]
        return []

    @staticmethod
    def _target_tokens(payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("candidate_id", "selector", "target", "target_text", "label", "name", "placeholder", "href", "text"):
            value = payload.get(key)
            if value is None or not str(value).strip():
                continue
            values.append(str(value).strip())
            simplified = ActionGrounder._strip_generic_target_words(str(value))
            if simplified and simplified not in values:
                values.append(simplified)
        return values

    @staticmethod
    def _strip_generic_target_words(value: str) -> str:
        generic = {"field", "input", "textbox", "text", "box", "button", "link", "control"}
        words = re.findall(r"\w+", value, flags=re.UNICODE)
        return " ".join(word for word in words if word.casefold() not in generic).strip()

    def _select_candidate(
        self,
        candidates: list[dict[str, Any]],
        tokens: list[str],
        *,
        preferred_kinds: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
        rejected: list[dict[str, Any]] = []
        normalized_tokens = [self._norm(token) for token in tokens if str(token).strip()]
        scored: list[tuple[int, int, dict[str, Any]]] = []
        eligible = [candidate for candidate in candidates if not preferred_kinds or candidate.get("kind") in preferred_kinds]
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
                haystack = " ".join(parts)
                raw_token_hit = any(str(candidate.get(key, "")).strip() in tokens for key in ("candidate_id", "selector"))
                if raw_token_hit:
                    scored.append((120, -index, candidate))
                    continue
                exact = any(token and token == part for token in normalized_tokens for part in parts)
                contains = any(token and token in haystack for token in normalized_tokens)
                if exact:
                    scored.append((100, -index, candidate))
                elif contains:
                    scored.append((70, -index, candidate))
                else:
                    rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "text_mismatch", "kind": candidate.get("kind")})
                continue
            if preferred_kinds and set(preferred_kinds).issubset(_INPUT_KINDS) and len(eligible) == 1:
                return candidate, rejected, 0.55
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

    def _is_skip_link_candidate(self, candidate: dict[str, Any]) -> bool:
        text = self._norm(candidate.get("text") or candidate.get("inner_text") or candidate.get("aria_label") or "")
        href = self._norm(candidate.get("href") or "")
        return text.startswith("skip to ") or bool(re.fullmatch(r"#[a-z0-9_-]+", href) and "skip" in text)

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())
