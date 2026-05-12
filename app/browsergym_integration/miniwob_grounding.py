from __future__ import annotations

from dataclasses import dataclass
import re
from difflib import SequenceMatcher
from typing import Any


@dataclass
class MiniWoBGroundingResult:
    action: str
    mapping_error: str | None = None
    selected_candidate: dict[str, Any] | None = None
    repeated_warning: str | None = None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _candidate_id(candidate: dict[str, Any]) -> str:
    for key in ("bid", "element_id", "backend_node_id", "node_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _candidate_text(candidate: dict[str, Any]) -> str:
    for key in ("name", "text", "label", "aria_label", "title", "value"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_click_target(action: str) -> str:
    match = re.match(r"^\s*click\s*\(\s*(.*?)\s*\)\s*$", str(action or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    raw = match.group(1).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    return raw.strip()


def _score_candidate(candidate: dict[str, Any], target: str) -> tuple[int, float, int]:
    text = _norm(_candidate_text(candidate))
    target_n = _norm(target)
    if not text or not target_n:
        return (0, 0.0, 0)
    role = _norm(candidate.get("role"))
    button_bonus = 20 if role == "button" else 0
    enabled_bonus = 4 if candidate.get("enabled", True) is not False else -10
    visible_bonus = 4 if candidate.get("visible", True) is not False else -10
    clickable_bonus = 2 if candidate.get("clickable", True) is not False else -8
    base = 0
    ratio = SequenceMatcher(None, text, target_n).ratio()
    if text == target_n:
        base = 100
    elif text.casefold() == target_n.casefold():
        base = 95
    elif target_n in text or text in target_n:
        base = 70
    elif ratio >= 0.72:
        base = 45
    return (base + button_bonus + enabled_bonus + visible_bonus + clickable_bonus, ratio, -len(text))


def find_click_candidate(candidates: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    scored: list[tuple[tuple[int, float, int], int, dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict) or not _candidate_id(candidate):
            continue
        score = _score_candidate(candidate, target)
        if score[0] > 0:
            scored.append((score, -idx, candidate))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    best = scored[0][2]
    return best if scored[0][0][0] >= 45 else None


def browsergym_click_action(candidate_id: str, action_syntax: list[str] | None = None) -> str:
    # BrowserGym high-level actions accept string browser ids as click("bid") in current releases.
    escaped = str(candidate_id).replace("\\", "\\\\").replace("\"", "\\\"")
    return f'click("{escaped}")'


def ground_miniwob_action(
    *,
    action: str,
    parsed_response: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    history: list[dict] | None = None,
    action_syntax: list[str] | None = None,
) -> MiniWoBGroundingResult:
    parsed_response = parsed_response or {}
    before = " ".join(str(action or "").strip().split())
    target = str(parsed_response.get("target_text") or "").strip() or _extract_click_target(before)
    target_bid = str(parsed_response.get("target_bid") or "").strip()
    history = history or []

    repeats = 0
    for item in reversed(history):
        if str(item.get("action") or "").strip() == before and float(item.get("reward") or 0) <= 0:
            repeats += 1
        else:
            break
    if repeats >= 2:
        return MiniWoBGroundingResult(
            action="noop()",
            mapping_error=f"action_mapping_failure: repeated ineffective action {before!r} without progress",
            repeated_warning="previous action had no effect; exact repeat blocked",
        )

    if before.lower().startswith("click"):
        selected = None
        if target_bid:
            selected = next((c for c in candidates if _candidate_id(c) == target_bid), None)
        if selected is None and target:
            selected = find_click_candidate(candidates, target)
        if selected is not None:
            grounded = browsergym_click_action(_candidate_id(selected), action_syntax=action_syntax)
            return MiniWoBGroundingResult(action=grounded, selected_candidate=selected)
        if target and not re.search(r"[\d_.:-]", target):
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error=f"action_mapping_failure: no clickable candidate matched target_text={target!r}",
            )
    return MiniWoBGroundingResult(action=before or "noop()")
