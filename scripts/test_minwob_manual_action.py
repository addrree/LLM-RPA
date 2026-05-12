#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_grounding import browsergym_mouse_click_action
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner


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
            if target in _norm(_candidate_text(candidate)) or _norm(_candidate_text(candidate)) in target:
                return candidate
    instruction_n = _norm(instruction)
    if instruction_n:
        for candidate in candidates:
            text = _norm(_candidate_text(candidate))
            if text and text in instruction_n:
                return candidate
    return candidates[0]


def _reset_and_extract(env):
    obs, info = env.reset()
    page = getattr(getattr(env, "unwrapped", None), "page", None)
    if page is None:
        page = BrowserGymRunner._find_page(env)
    scale_factor = getattr(page, "_bgym_scale_factor", 1.0) if page is not None else 1.0
    candidates, failed = BrowserGymRunner._extract_page_clickable_candidates(env)
    ctx = browsergym_obs_to_page_context({**obs, "page_clickable_candidates": candidates}, info) if isinstance(obs, dict) else {}
    return obs, info, ctx, candidates, failed, scale_factor


def _try_action(env, candidate: dict[str, Any], *, scaled: bool):
    x_key = "browsergym_center_x" if scaled else "page_center_x"
    y_key = "browsergym_center_y" if scaled else "page_center_y"
    x = candidate.get(x_key, candidate.get("center_x"))
    y = candidate.get(y_key, candidate.get("center_y"))
    action = browsergym_mouse_click_action(float(x), float(y))
    obs, reward, terminated, truncated, info = env.step(action)
    return {"action": action, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare unscaled vs BrowserGym-scaled MiniWoB mouse clicks without an LLM")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--target-text", default=None)
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    importlib.import_module("browsergym.miniwob")

    env = gym.make(args.env_id)
    try:
        _, _, ctx, candidates, failed, scale_factor = _reset_and_extract(env)
        instruction = ctx.get("goal_instruction")
        print("instruction:", instruction)
        print("scale_factor:", scale_factor)
        print("candidate_extraction_failed:", failed)
        print("candidates:", json.dumps(candidates, ensure_ascii=False, indent=2, default=str))
        candidate = _choose_candidate(candidates, args.target_text, instruction)
        if candidate is None:
            print("selected_candidate: null")
            return 2
        print("selected_candidate:", json.dumps(candidate, ensure_ascii=False, indent=2, default=str))

        unscaled_result = _try_action(env, candidate, scaled=False)
        print("unscaled_result:", json.dumps(unscaled_result, ensure_ascii=False, indent=2, default=str))

        _, _, ctx, candidates, failed, scale_factor = _reset_and_extract(env)
        candidate = _choose_candidate(candidates, args.target_text, ctx.get("goal_instruction"))
        if candidate is None:
            print("scaled_selected_candidate: null")
            return 2
        print("scaled_scale_factor:", scale_factor)
        scaled_result = _try_action(env, candidate, scaled=True)
        print("scaled_result:", json.dumps(scaled_result, ensure_ascii=False, indent=2, default=str))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
