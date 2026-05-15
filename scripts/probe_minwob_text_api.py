#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_grounding import (
    extract_textbox_candidates_from_observation,
    find_submit_button,
    map_login_textboxes,
    merge_textbox_candidates,
    parse_quoted_strings,
    parse_username_password_instruction,
    real_candidate_bid,
    textbox_candidates,
)
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner

TASK_INFO_KEYS = ("REWARD_GLOBAL", "RAW_REWARD_GLOBAL", "DONE_GLOBAL")


def _reset(env: Any, seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        obs, info = env.reset()
    return obs if isinstance(obs, dict) else {"raw_obs": repr(obs)}, info if isinstance(info, dict) else {}


def _merge_candidates(clickables: list[dict[str, Any]], textboxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(clickables or [])
    for textbox in textboxes or []:
        bid = real_candidate_bid(textbox)
        backend_id = str(textbox.get("backendDOMNodeId") or "").strip()
        if not any((bid and real_candidate_bid(c) == bid) or (backend_id and str(c.get("backendDOMNodeId") or "").strip() == backend_id) for c in merged):
            merged.append(textbox)
    return merged


def _reset_and_extract(env: Any, seed: int | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    obs, info = _reset(env, seed)
    candidates, failed = BrowserGymRunner._extract_page_clickable_candidates(env)
    merged_obs = dict(obs)
    if candidates:
        merged_obs["page_clickable_candidates"] = candidates
    elif isinstance(obs.get("page_clickable_candidates"), list):
        candidates = list(obs.get("page_clickable_candidates") or [])
    if failed and not candidates:
        merged_obs["page_candidate_extraction_failed"] = True
    ctx = browsergym_obs_to_page_context(merged_obs, info)
    button_candidates = list(ctx.get("clickable_candidates") or candidates or [])
    textboxes = merge_textbox_candidates(textbox_candidates(button_candidates) + extract_textbox_candidates_from_observation(obs, info))
    all_candidates = _merge_candidates(button_candidates, textboxes)
    return obs, info, ctx, all_candidates, textboxes, button_candidates


def _task_info(info: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = [info]
    for key in ("task_info", "info", "env_info"):
        nested = info.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    return {key: next((source.get(key) for source in sources if key in source), None) for key in TASK_INFO_KEYS}


def _json_action_arg(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _fill(bid: str, text: str) -> str:
    return f"fill({_json_action_arg(bid)}, {_json_action_arg(text)})"


def _click(bid: str) -> str:
    return f"click({_json_action_arg(bid)}, \"left\")"


def _focus(bid: str) -> str:
    return f"focus({_json_action_arg(bid)})"


def _clear(bid: str) -> str:
    return f"clear({_json_action_arg(bid)})"


def _press(bid: str, key: str) -> str:
    return f"press({_json_action_arg(bid)}, {_json_action_arg(key)})"


def _keyboard_type(text: str) -> str:
    return f"keyboard_type({_json_action_arg(text)})"


def _keyboard_insert_text(text: str) -> str:
    return f"keyboard_insert_text({_json_action_arg(text)})"


def _instruction_targets(instruction: str) -> tuple[list[str], dict[str, str]]:
    login_values = parse_username_password_instruction(instruction)
    if login_values:
        return [login_values[k] for k in ("username", "password") if k in login_values], login_values
    quoted = parse_quoted_strings(instruction)
    return quoted[:1], {}


def _sequence_for(method: str, textbox_bids: list[str], targets: list[str], submit_bid: str | None) -> list[str]:
    actions: list[str] = []
    pairs = list(zip(textbox_bids, targets))
    if method == "fill_then_submit":
        actions.extend(_fill(bid, text) for bid, text in pairs)
    elif method == "click_keyboard_type_then_submit":
        for bid, text in pairs:
            actions.extend([_click(bid), _keyboard_type(text)])
    elif method == "focus_keyboard_type_then_submit":
        for bid, text in pairs:
            actions.extend([_focus(bid), _keyboard_type(text)])
    elif method == "clear_fill_then_submit":
        for bid, text in pairs:
            actions.extend([_clear(bid), _fill(bid, text)])
    elif method == "fill_press_enter":
        actions.extend(_fill(bid, text) for bid, text in pairs)
        if textbox_bids:
            actions.append(_press(textbox_bids[-1], "Enter"))
        return actions
    elif method == "click_keyboard_insert_text_then_submit":
        for bid, text in pairs:
            actions.extend([_click(bid), _keyboard_insert_text(text)])
    else:
        raise ValueError(method)
    if submit_bid:
        actions.append(_click(submit_bid))
    return actions


def _select_textboxes(instruction: str, textboxes: list[dict[str, Any]], targets: list[str]) -> tuple[list[dict[str, Any]], str]:
    if not textboxes:
        return [], "no textbox candidates with BrowserGym bid"
    login_map = map_login_textboxes(instruction, textboxes)
    if login_map:
        selected = [login_map[key] for key in ("username", "password") if key in login_map]
        return selected, "login instruction: first textbox=username, second textbox=password"
    return textboxes[: max(1, len(targets))], "first textbox candidate with real BrowserGym bid"


def _run_method(env: Any, *, env_id: str, seed: int | None, method: str) -> dict[str, Any]:
    reward = None
    terminated = False
    truncated = False
    error = None
    actions: list[str] = []
    info: dict[str, Any] = {}
    instruction = ""
    targets: list[str] = []
    textbox_bids: list[str] = []
    submit_bid = None
    try:
        _obs, info, ctx, candidates, textboxes, button_candidates = _reset_and_extract(env, seed)
        instruction = str(ctx.get("goal_instruction") or "")
        targets, login_values = _instruction_targets(instruction)
        selected_textboxes, selection_reason = _select_textboxes(instruction, [c for c in textboxes if real_candidate_bid(c)], targets)
        textbox_bids = [real_candidate_bid(c) for c in selected_textboxes if real_candidate_bid(c)]
        submit = find_submit_button(candidates)
        submit_bid = real_candidate_bid(submit) if submit else None
        if not textbox_bids:
            actions = []
            error = "missing textbox bid"
        elif not targets:
            actions = []
            error = "missing target text"
        else:
            actions = _sequence_for(method, textbox_bids, targets, submit_bid)
        for action in actions:
            if terminated or truncated:
                break
            try:
                _obs2, reward, terminated, truncated, info2 = env.step(action)
                if isinstance(info2, dict):
                    info = info2
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "method": method,
        "env_id": env_id,
        "seed": seed,
        "instruction": instruction,
        "target_text": targets[0] if len(targets) == 1 else targets,
        "selected_textbox_bid": textbox_bids[0] if len(textbox_bids) == 1 else textbox_bids,
        "selected_password_bid": textbox_bids[1] if len(textbox_bids) > 1 else None,
        "submit_bid": submit_bid,
        "textbox_candidates": textboxes if 'textboxes' in locals() else [],
        "button_candidates": [c for c in button_candidates if str(c.get("role") or c.get("tag") or c.get("type") or "").lower() in {"button", "submit"}] if 'button_candidates' in locals() else [],
        "textbox_selection_reason": selection_reason if 'selection_reason' in locals() else None,
        "why_textbox_candidate_selected": selection_reason if 'selection_reason' in locals() else None,
        "actions": actions,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "error": error,
        "task_info": _task_info(info),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe BrowserGym MiniWoB text-entry high-level actions without LLM or Playwright fallback.")
    parser.add_argument("--env-id", default="browsergym/miniwob.enter-text")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    if "miniwob" in args.env_id.lower():
        importlib.import_module("browsergym.miniwob")

    env = gym.make(args.env_id)
    if "login" in args.env_id.lower():
        methods = ["fill_then_submit", "click_keyboard_type_then_submit"]
    else:
        methods = [
            "fill_then_submit",
            "click_keyboard_type_then_submit",
            "focus_keyboard_type_then_submit",
            "clear_fill_then_submit",
            "fill_press_enter",
            "click_keyboard_insert_text_then_submit",
        ]
    try:
        for method in methods:
            print(json.dumps(_run_method(env, env_id=args.env_id, seed=args.seed, method=method), ensure_ascii=False), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
