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
    mapping_strategy: str | None = None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


REAL_BID_KEYS = ("bid", "browsergym_id", "data-bid", "data_bid", "data-testid", "data_testid", "ref")


def _candidate_id(candidate: dict[str, Any]) -> str:
    for key in REAL_BID_KEYS:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _candidate_text_values(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("name", "text", "value", "label", "ariaLabel", "aria_label", "title"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    return values


def _candidate_text(candidate: dict[str, Any]) -> str:
    values = _candidate_text_values(candidate)
    return values[0] if values else ""


def _extract_click_target(action: str) -> str:
    match = re.match(r"^\s*click\s*\(\s*(.*?)\s*\)\s*$", str(action or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    raw = match.group(1).strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    return raw.strip()


def _visible_enabled_bonus(candidate: dict[str, Any]) -> int:
    enabled_bonus = 12 if candidate.get("enabled", True) is not False and candidate.get("disabled") is not True else -30
    visible_bonus = 12 if candidate.get("visible", True) is not False else -30
    clickable_bonus = 4 if candidate.get("clickable", True) is not False else -8
    return enabled_bonus + visible_bonus + clickable_bonus


def _score_candidate(candidate: dict[str, Any], target: str) -> tuple[int, float, int]:
    target_n = _norm(target)
    if not target_n:
        return (0, 0.0, 0)
    role = _norm(candidate.get("role"))
    tag = _norm(candidate.get("tag"))
    buttonish_bonus = 25 if role == "button" or tag in {"button", "input"} else 0
    best_base = 0
    best_ratio = 0.0
    best_len = 9999
    for text_value in _candidate_text_values(candidate):
        text = _norm(text_value)
        if not text:
            continue
        ratio = SequenceMatcher(None, text, target_n).ratio()
        if text == target_n:
            base = 120
        elif target_n in text or text in target_n:
            base = 80
        elif ratio >= 0.72:
            base = 45
        else:
            base = 0
        if base > best_base or (base == best_base and ratio > best_ratio):
            best_base = base
            best_ratio = ratio
            best_len = len(text)
    return (best_base + buttonish_bonus + _visible_enabled_bonus(candidate), best_ratio, -best_len)


def find_click_candidate(candidates: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    scored: list[tuple[tuple[int, float, int], int, dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict):
            continue
        score = _score_candidate(candidate, target)
        if score[0] > 0:
            scored.append((score, -idx, candidate))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    best_score, _, best = scored[0]
    return best if best_score[0] >= 45 else None


def browsergym_click_action(candidate_id: str, action_syntax: list[str] | None = None) -> str:
    escaped = str(candidate_id).replace("\\", "\\\\").replace('"', '\\"')
    return f'click("{escaped}")'


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if isinstance(bbox, dict):
        x = _numeric(bbox.get("x")) or 0.0
        y = _numeric(bbox.get("y")) or 0.0
        width = _numeric(bbox.get("width"))
        height = _numeric(bbox.get("height"))
        if width is not None and height is not None:
            return x + width / 2.0, y + height / 2.0
        left = _numeric(bbox.get("left"))
        right = _numeric(bbox.get("right"))
        top = _numeric(bbox.get("top"))
        bottom = _numeric(bbox.get("bottom"))
        if None not in (left, right, top, bottom):
            return (left + right) / 2.0, (top + bottom) / 2.0
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        nums = [_numeric(v) for v in bbox[:4]]
        if all(v is not None for v in nums):
            x, y, w, h = nums  # type: ignore[misc]
            return float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
    return None


def candidate_center_with_strategy(candidate: dict[str, Any]) -> tuple[float, float, str] | None:
    coordinate_pairs = (
        ("action_x", "action_y", "coordinate_scaled"),
        ("action_center_x", "action_center_y", "coordinate_scaled"),
        ("browsergym_center_x", "browsergym_center_y", "coordinate_scaled"),
        ("click_x", "click_y", "coordinate_scaled" if str(candidate.get("action_coordinate_space") or "").lower() == "browsergym_scaled" else "coordinate"),
        ("center_x", "center_y", "coordinate_raw"),
    )
    for x_key, y_key, strategy in coordinate_pairs:
        cx = _numeric(candidate.get(x_key))
        cy = _numeric(candidate.get(y_key))
        if cx is not None and cy is not None:
            return cx, cy, strategy
    for bbox_key, strategy in (
        ("action_bbox", "coordinate_scaled"),
        ("browsergym_bbox", "coordinate_scaled"),
        ("bbox", "coordinate_raw"),
        ("bounding_box", "coordinate_raw"),
    ):
        center = _bbox_center(candidate.get(bbox_key))
        if center is not None:
            return center[0], center[1], strategy
    return None


def candidate_center(candidate: dict[str, Any]) -> tuple[float, float] | None:
    center = candidate_center_with_strategy(candidate)
    if center is None:
        return None
    return center[0], center[1]


def browsergym_mouse_click_action(x: float, y: float) -> str:
    def fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")

    return f'mouse_click({fmt(x)}, {fmt(y)}, "left")'


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
            mapping_strategy="none",
        )

    if before.lower().startswith("click") or (before.lower().startswith("mouse_click") and (target or target_bid)):
        selected = None
        if target_bid:
            selected = next((c for c in candidates if _candidate_id(c) == target_bid), None)
        if selected is None and target:
            selected = find_click_candidate(candidates, target)
        if selected is not None:
            candidate_id = _candidate_id(selected)
            if candidate_id:
                return MiniWoBGroundingResult(action=browsergym_click_action(candidate_id, action_syntax=action_syntax), selected_candidate=selected, mapping_strategy="bid")
            center = candidate_center_with_strategy(selected)
            if center is not None:
                return MiniWoBGroundingResult(action=browsergym_mouse_click_action(center[0], center[1]), selected_candidate=selected, mapping_strategy=center[2])
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error=f"action_mapping_failure: no grounded bid or bbox for target_text={target!r}",
                selected_candidate=selected,
                mapping_strategy="none",
            )
        if target and not re.search(r"[\d_.:-]", target):
            return MiniWoBGroundingResult(
                action="noop()",
                mapping_error=f"action_mapping_failure: no clickable candidate matched target_text={target!r}",
                mapping_strategy="none",
            )
    return MiniWoBGroundingResult(action=before or "noop()")
