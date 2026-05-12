#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_grounding import browsergym_click_action, browsergym_mouse_click_action
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner

REAL_BID_KEYS = ("bid", "browsergym_id", "data-bid", "data_bid", "data-testid", "data_testid", "ref")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _candidate_text(candidate: dict[str, Any]) -> str:
    for key in ("text", "name", "value", "label", "ariaLabel", "aria_label", "title"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _choose_candidate(candidates: list[dict[str, Any]], target_text: str | None, instruction: str | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    target = _norm(target_text)
    if target:
        for candidate in candidates:
            text = _norm(_candidate_text(candidate))
            if text and (target in text or text in target):
                return candidate
    instruction_n = _norm(instruction)
    if instruction_n:
        for candidate in candidates:
            text = _norm(_candidate_text(candidate))
            if text and text in instruction_n:
                return candidate
    return candidates[0]


def _real_bid(candidate: dict[str, Any] | None) -> str | None:
    if not isinstance(candidate, dict):
        return None
    for key in REAL_BID_KEYS:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _reset_and_extract(env, seed: int | None):
    try:
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        obs, info = env.reset()
    page = BrowserGymRunner._find_page(env)
    scale_factor = getattr(page, "_bgym_scale_factor", 1.0) if page is not None else 1.0
    candidates, failed = BrowserGymRunner._extract_page_clickable_candidates(env)
    if not candidates and isinstance(obs, dict) and isinstance(obs.get("page_clickable_candidates"), list):
        candidates = list(obs.get("page_clickable_candidates") or [])
        failed = False
    ctx = browsergym_obs_to_page_context({**obs, "page_clickable_candidates": candidates}, info) if isinstance(obs, dict) else {}
    return obs, info, ctx, candidates, failed, scale_factor


def _candidate_points(candidate: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    page_x = float(candidate.get("page_center_x", candidate.get("center_x")))
    page_y = float(candidate.get("page_center_y", candidate.get("center_y")))
    browsergym_x = float(candidate.get("browsergym_center_x", page_x))
    browsergym_y = float(candidate.get("browsergym_center_y", page_y))
    return (page_x, page_y), (browsergym_x, browsergym_y)


def _step_action(env, action: str) -> dict[str, Any]:
    obs, reward, terminated, truncated, info = env.step(action)
    return {"action": action, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}


def _run_method(env, *, method: str, seed: int | None, target_text: str | None) -> dict[str, Any]:
    _, _, ctx, candidates, failed, scale_factor = _reset_and_extract(env, seed)
    instruction = ctx.get("goal_instruction")
    candidate = _choose_candidate(candidates, target_text, instruction)
    result: dict[str, Any] = {
        "method": method,
        "instruction": instruction,
        "candidate_extraction_failed": failed,
        "scale_factor": scale_factor,
        "selected_candidate": candidate,
    }
    if candidate is None:
        result.update({"action": None, "reward": None, "terminated": False, "truncated": False, "error": "no selected candidate"})
        return result
    page_center, browsergym_center = _candidate_points(candidate)
    result["page_center"] = page_center
    result["browsergym_center"] = browsergym_center

    if method == "raw_mouse_click":
        action = browsergym_mouse_click_action(page_center[0], page_center[1])
        result.update(_step_action(env, action))
        return result
    if method == "scaled_mouse_click":
        action = browsergym_mouse_click_action(browsergym_center[0], browsergym_center[1])
        result.update(_step_action(env, action))
        return result
    if method == "playwright_direct":
        page = BrowserGymRunner._find_page(env)
        mouse = getattr(page, "mouse", None) if page is not None else None
        if mouse is None or not hasattr(mouse, "click"):
            result.update({"action": None, "reward": None, "terminated": False, "truncated": False, "error": "playwright page.mouse.click unavailable"})
            return result
        mouse.click(page_center[0], page_center[1])
        action = "noop()"
        result.update(_step_action(env, action))
        result["action"] = f"page.mouse.click({page_center[0]}, {page_center[1]}) + {action}"
        return result
    if method == "bid_click":
        bid = _real_bid(candidate)
        if not bid:
            result.update({"action": None, "reward": None, "terminated": False, "truncated": False, "error": "no real bid/data-testid/browsergym_id/data-bid/ref on selected candidate"})
            return result
        action = browsergym_click_action(bid)
        result.update(_step_action(env, action))
        return result
    result.update({"action": None, "reward": None, "terminated": False, "truncated": False, "error": f"unknown method {method}"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MiniWoB click methods; each method resets and reselects its own candidate")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--target-text", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--methods", default="raw_mouse_click,scaled_mouse_click,playwright_direct,bid_click")
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    importlib.import_module("browsergym.miniwob")

    env = gym.make(args.env_id)
    try:
        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            result = _run_method(env, method=method, seed=args.seed, target_text=args.target_text)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
