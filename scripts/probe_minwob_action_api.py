#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_grounding import browsergym_click_action
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner

REAL_BID_KEYS = ("bid", "browsergym_id", "data-bid", "data_bid", "data-testid", "data_testid", "ref")
TASK_INFO_KEYS = ("REWARD_GLOBAL", "RAW_REWARD_GLOBAL", "DONE_GLOBAL")


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


def _bid_click_action_variant(bid: str, variant: str) -> str:
    escaped_double = str(bid).replace("\\", "\\\\").replace('"', '\\"')
    escaped_single = str(bid).replace("\\", "\\\\").replace("'", "\\'")
    if variant == "bid_click_double_with_left":
        return f'click("{escaped_double}", "left")'
    if variant == "bid_click_single_with_left":
        return f"click('{escaped_single}', 'left')"
    if variant == "bid_click_double_no_button":
        return f'click("{escaped_double}")'
    if variant == "bid_click_single_no_button":
        return f"click('{escaped_single}')"
    if variant == "bid_click":
        return browsergym_click_action(bid)
    raise ValueError(f"unknown bid click variant {variant}")


def _candidate_verbose(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    return {
        "bid": _real_bid(candidate),
        "bid_source": candidate.get("bid_source"),
        "text": _candidate_text(candidate),
        "bbox": candidate.get("bbox"),
        "browsergym_bbox": candidate.get("browsergym_bbox"),
        "centers": {
            "center_x": candidate.get("center_x"),
            "center_y": candidate.get("center_y"),
            "page_center_x": candidate.get("page_center_x"),
            "page_center_y": candidate.get("page_center_y"),
            "browsergym_center_x": candidate.get("browsergym_center_x"),
            "browsergym_center_y": candidate.get("browsergym_center_y"),
        },
    }


def _safe_repr(value: Any, *, limit: int = 1000) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive diagnostics path
        rendered = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(rendered) > limit:
        return rendered[:limit] + f"...<truncated {len(rendered) - limit} chars>"
    return rendered


def _safe_attr_dump(obj: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = {}
    if obj is None:
        return dumped
    try:
        names = dir(obj)
    except Exception as exc:  # pragma: no cover - defensive diagnostics path
        return {"<dir_error>": f"{type(exc).__name__}: {exc}"}
    needles = ("action", "action_set", "action_mapping")
    for name in sorted(n for n in names if any(needle in n for needle in needles)):
        try:
            value = getattr(obj, name)
        except Exception as exc:  # pragma: no cover - defensive diagnostics path
            dumped[name] = {"type": "<getattr_error>", "repr": f"{type(exc).__name__}: {exc}"}
            continue
        dumped[name] = {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": _safe_repr(value)}
    return dumped


def _describe_action_set(env: Any) -> dict[str, Any]:
    unwrapped = getattr(env, "unwrapped", None)
    action_set = getattr(unwrapped, "action_set", None) if unwrapped is not None else None
    description: dict[str, Any] = {
        "type": f"{type(action_set).__module__}.{type(action_set).__qualname__}" if action_set is not None else None,
        "repr": _safe_repr(action_set) if action_set is not None else None,
    }
    if action_set is None:
        return description
    for attr in ("get_action_description", "describe", "action_description", "to_python_code", "docs", "description"):
        member = getattr(action_set, attr, None)
        if callable(member):
            try:
                description[attr] = str(member())
            except Exception as exc:  # pragma: no cover - depends on installed action set
                description[attr] = f"<call failed: {type(exc).__name__}: {exc}>"
        elif member is not None:
            description[attr] = str(member)
    return description


def _safe_dump_text(value: Any, *, limit: int = 5000) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = _safe_repr(value, limit=limit * 2)
    if len(rendered) > limit:
        return rendered[:limit] + f"...<truncated {len(rendered) - limit} chars>"
    return rendered


def _obs_text_dump(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {"obs_repr": _safe_repr(obs, limit=5000)}
    dump: dict[str, Any] = {"obs.keys": sorted(str(key) for key in obs.keys())}
    for key in ("axtree_txt", "dom_object"):
        if key in obs:
            dump[f"{key}_first_5000"] = _safe_dump_text(obs.get(key), limit=5000)
    return dump


def _reset_and_extract(env: Any, seed: int | None) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]], bool, float]:
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
    return obs, info if isinstance(info, dict) else {}, ctx, candidates, failed, float(scale_factor or 1.0)


def _candidate_points(candidate: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    page_x = float(candidate.get("page_center_x", candidate.get("center_x")))
    page_y = float(candidate.get("page_center_y", candidate.get("center_y")))
    browsergym_x = float(candidate.get("browsergym_center_x", page_x))
    browsergym_y = float(candidate.get("browsergym_center_y", page_y))
    return (page_x, page_y), (browsergym_x, browsergym_y)


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _mouse_click_action(x: float, y: float, *, with_button: bool) -> str:
    if with_button:
        return f'mouse_click({_fmt_number(x)}, {_fmt_number(y)}, "left")'
    return f"mouse_click({_fmt_number(x)}, {_fmt_number(y)})"


def _mouse_move_action(x: float, y: float) -> str:
    return f"mouse_move({_fmt_number(x)}, {_fmt_number(y)})"


def _mouse_down_action(x: float | None = None, y: float | None = None, *, with_coords: bool) -> str:
    if with_coords:
        return f'mouse_down({_fmt_number(float(x))}, {_fmt_number(float(y))}, "left")'
    return 'mouse_down("left")'


def _mouse_up_action(x: float | None = None, y: float | None = None, *, with_coords: bool) -> str:
    if with_coords:
        return f'mouse_up({_fmt_number(float(x))}, {_fmt_number(float(y))}, "left")'
    return 'mouse_up("left")'


def _task_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {key: None for key in TASK_INFO_KEYS}
    sources = [info]
    for nested_key in ("task_info", "info", "env_info"):
        nested = info.get(nested_key)
        if isinstance(nested, dict):
            sources.append(nested)
    return {key: next((source.get(key) for source in sources if key in source), None) for key in TASK_INFO_KEYS}


def _step_action(env: Any, action: str) -> tuple[Any, float | None, bool, bool, dict[str, Any], str | None]:
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except Exception as exc:
        return None, None, False, False, {}, f"{type(exc).__name__}: {exc}"
    return obs, reward, bool(terminated), bool(truncated), info if isinstance(info, dict) else {}, None


def _step_actions(env: Any, actions: list[str]) -> dict[str, Any]:
    last_obs: Any = None
    reward: float | None = None
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    for action in actions:
        last_obs, reward, terminated, truncated, info, error = _step_action(env, action)
        if error is not None:
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "error": error, "task_info": _task_info(info)}
        if terminated or truncated:
            break
    return {"reward": reward, "terminated": terminated, "truncated": truncated, "error": None, "task_info": _task_info(info), "last_obs_keys": sorted(last_obs.keys()) if isinstance(last_obs, dict) else None}


def _method_actions(method: str, page_center: tuple[float, float], browsergym_center: tuple[float, float], candidate: dict[str, Any]) -> tuple[list[str] | None, str | None]:
    raw_x, raw_y = page_center
    scaled_x, scaled_y = browsergym_center
    if method == "raw_mouse_click_with_button":
        return [_mouse_click_action(raw_x, raw_y, with_button=True)], None
    if method == "scaled_mouse_click_with_button":
        return [_mouse_click_action(scaled_x, scaled_y, with_button=True)], None
    if method == "raw_mouse_click_no_button":
        return [_mouse_click_action(raw_x, raw_y, with_button=False)], None
    if method == "scaled_mouse_click_no_button":
        return [_mouse_click_action(scaled_x, scaled_y, with_button=False)], None
    if method == "raw_mouse_move_down_up":
        return [_mouse_move_action(raw_x, raw_y), _mouse_down_action(with_coords=False), _mouse_up_action(with_coords=False)], None
    if method == "scaled_mouse_move_down_up":
        return [_mouse_move_action(scaled_x, scaled_y), _mouse_down_action(with_coords=False), _mouse_up_action(with_coords=False)], None
    if method == "raw_mouse_down_up_with_coords":
        return [_mouse_down_action(raw_x, raw_y, with_coords=True), _mouse_up_action(raw_x, raw_y, with_coords=True)], None
    if method == "scaled_mouse_down_up_with_coords":
        return [_mouse_down_action(scaled_x, scaled_y, with_coords=True), _mouse_up_action(scaled_x, scaled_y, with_coords=True)], None
    if method in {"bid_click", "bid_click_double_with_left", "bid_click_single_with_left", "bid_click_double_no_button", "bid_click_single_no_button"}:
        bid = _real_bid(candidate)
        if not bid:
            return None, "no real bid/data-testid/browsergym_id/data-bid/ref on selected candidate"
        return [_bid_click_action_variant(bid, method)], None
    return None, f"unknown method {method}"


def _run_method(env: Any, *, method: str, seed: int | None, target_text: str | None) -> dict[str, Any]:
    _, _, ctx, candidates, failed, scale_factor = _reset_and_extract(env, seed)
    instruction = ctx.get("goal_instruction")
    candidate = _choose_candidate(candidates, target_text, instruction)
    result: dict[str, Any] = {
        "method": method,
        "action": None,
        "actions": None,
        "instruction": instruction,
        "candidate_extraction_failed": failed,
        "scale_factor": scale_factor,
        "selected_candidate": candidate,
        "selected_candidate_verbose": _candidate_verbose(candidate),
        "page_center": None,
        "browsergym_center": None,
        "reward": None,
        "terminated": False,
        "truncated": False,
        "error": None,
        "task_info": {key: None for key in TASK_INFO_KEYS},
    }
    if candidate is None:
        result["error"] = "no selected candidate"
        return result
    try:
        page_center, browsergym_center = _candidate_points(candidate)
    except Exception as exc:
        result["error"] = f"candidate center unavailable: {type(exc).__name__}: {exc}"
        return result
    result["page_center"] = page_center
    result["browsergym_center"] = browsergym_center

    if method == "playwright_direct_control":
        page = BrowserGymRunner._find_page(env)
        mouse = getattr(page, "mouse", None) if page is not None else None
        click = getattr(mouse, "click", None)
        if not callable(click):
            result["error"] = "playwright page.mouse.click unavailable"
            return result
        try:
            click(page_center[0], page_center[1])
        except Exception as exc:
            result["error"] = f"playwright click failed: {type(exc).__name__}: {exc}"
            return result
        actions = [f"page.mouse.click({page_center[0]}, {page_center[1]})", "noop()"]
        result["action"] = " + ".join(actions)
        result["actions"] = actions
        result.update(_step_actions(env, ["noop()"]))
        return result

    actions, error = _method_actions(method, page_center, browsergym_center, candidate)
    if error is not None or actions is None:
        result["error"] = error
        return result
    result["action"] = actions[0] if len(actions) == 1 else " ; ".join(actions)
    result["actions"] = actions
    result.update(_step_actions(env, actions))
    return result


def _emit(row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe MiniWoB BrowserGym action API without any LLM calls")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-text", default=None)
    parser.add_argument(
        "--methods",
        default=",".join(
            [
                "raw_mouse_click_with_button",
                "scaled_mouse_click_with_button",
                "raw_mouse_click_no_button",
                "scaled_mouse_click_no_button",
                "raw_mouse_move_down_up",
                "scaled_mouse_move_down_up",
                "raw_mouse_down_up_with_coords",
                "scaled_mouse_down_up_with_coords",
                "bid_click_double_with_left",
                "bid_click_single_with_left",
                "bid_click_double_no_button",
                "bid_click_single_no_button",
                "playwright_direct_control",
            ]
        ),
        help="Comma-separated methods to run; defaults to all probe methods",
    )
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401

    importlib.import_module("browsergym.miniwob")

    env = gym.make(args.env_id)
    try:
        obs, _, ctx, candidates, failed, scale_factor = _reset_and_extract(env, args.seed)
        instruction = ctx.get("goal_instruction")
        selected = _choose_candidate(candidates, args.target_text, instruction)
        unwrapped = getattr(env, "unwrapped", None)
        _emit(
            {
                "kind": "diagnostics",
                "env_id": args.env_id,
                "seed": args.seed,
                "env.action_space": _safe_repr(getattr(env, "action_space", None), limit=5000),
                "type(env.unwrapped)": f"{type(unwrapped).__module__}.{type(unwrapped).__qualname__}" if unwrapped is not None else None,
                "env.unwrapped_action_attrs": _safe_attr_dump(unwrapped),
                "action_set_description": _describe_action_set(env),
                "observation_dump": _obs_text_dump(obs),
                "instruction": instruction,
                "selected_candidate": selected,
                "selected_candidate_verbose": _candidate_verbose(selected),
                "candidate_extraction_failed": failed,
                "candidate_count": len(candidates),
                "scale_factor": scale_factor,
            }
        )

        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            _emit(_run_method(env, method=method, seed=args.seed, target_text=args.target_text))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
