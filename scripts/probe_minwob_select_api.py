from __future__ import annotations

import argparse
import json
from typing import Any

from app.browsergym_integration.local_extractor import BrowserGymLocalExtractor
from app.browsergym_integration.miniwob_grounding import real_candidate_bid


def _json_arg(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _select_option(bid: str, option: str) -> str:
    return f"select_option({_json_arg(bid)}, {_json_arg(option)})"


def _select_option_list(bid: str, option: str) -> str:
    return f"select_option({_json_arg(bid)}, [{_json_arg(option)}])"


def _click(bid: str) -> str:
    return f"click({_json_arg(bid)}, \"left\")"


def _focus(bid: str) -> str:
    return f"focus({_json_arg(bid)})"


def _press(bid: str, key: str) -> str:
    return f"press({_json_arg(bid)}, {_json_arg(key)})"


def _make_env(env_id: str):
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym

    return gym.make(env_id)


def _reset(env, seed: int | None):
    if seed is None:
        return env.reset()
    return env.reset(seed=seed)


def _extract(env, obs: dict, info: dict) -> list[dict[str, Any]]:
    candidates = []
    try:
        candidates.extend(BrowserGymLocalExtractor(env).extract_candidates())
    except Exception:
        pass
    for key in ("page_clickable_candidates", "clickable_candidates", "dom_object"):
        value = obs.get(key) if isinstance(obs, dict) else None
        if isinstance(value, list):
            candidates.extend(value)
    # De-duplicate by bid/text.
    seen = set()
    unique = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        marker = (real_candidate_bid(c), c.get("role"), c.get("tag"), c.get("text") or c.get("name") or c.get("value"))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(c)
    return unique


def _is_combo(c: dict[str, Any]) -> bool:
    return str(c.get("role", "")).lower() in {"combobox", "listbox"} or str(c.get("tag", "")).lower() == "select"


def _is_option(c: dict[str, Any]) -> bool:
    return str(c.get("role", "")).lower() == "option" or str(c.get("tag", "")).lower() == "option"


def _candidate_text(c: dict[str, Any]) -> str:
    return str(c.get("text") or c.get("name") or c.get("label") or c.get("value") or "").strip()


def _task_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = ["goal", "task_id", "utterance", "reward", "done"]
    nested = info.get("task_info") if isinstance(info.get("task_info"), dict) else {}
    return {key: info.get(key, nested.get(key)) for key in keys if key in info or key in nested}


def _run_method(env_id: str, seed: int | None, method: str) -> dict[str, Any]:
    env = _make_env(env_id)
    actions = []
    reward = None
    terminated = False
    error = None
    info: dict[str, Any] = {}
    selected_combo = None
    selected_option = None
    instruction = ""
    try:
        obs, info = _reset(env, seed)
        instruction = str(info.get("goal") or (info.get("task_info") or {}).get("goal") or obs.get("goal") or "")
        candidates = _extract(env, obs, info)
        combo = next((c for c in candidates if _is_combo(c) and real_candidate_bid(c)), None)
        options = [c for c in candidates if _is_option(c) and real_candidate_bid(c)]
        option = options[0] if options else next((c for c in candidates if _candidate_text(c) and not _is_combo(c) and real_candidate_bid(c)), None)
        if not combo:
            error = "missing combobox bid"
        elif not option:
            error = "missing option bid/text"
        else:
            combo_bid = real_candidate_bid(combo)
            option_bid = real_candidate_bid(option)
            option_text = _candidate_text(option)
            selected_combo = {"bid": combo_bid, "text": _candidate_text(combo), "role": combo.get("role"), "tag": combo.get("tag")}
            selected_option = {"bid": option_bid, "text": option_text, "role": option.get("role"), "tag": option.get("tag")}
            if method == "select_option_text":
                actions = [_select_option(combo_bid, option_text)]
            elif method == "select_option_list_text":
                actions = [_select_option_list(combo_bid, option_text)]
            elif method == "click_combo_click_option":
                actions = [_click(combo_bid), _click(option_bid)]
            elif method == "focus_arrow_enter":
                actions = [_focus(combo_bid), _press(combo_bid, "ArrowDown"), _press(combo_bid, "Enter")]
            elif method == "click_arrow_enter":
                actions = [_click(combo_bid), _press(combo_bid, "ArrowDown"), _press(combo_bid, "Enter")]
            else:
                error = f"unknown method {method}"
        for action in actions:
            if terminated:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            terminated = bool(terminated or truncated)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "method": method,
        "instruction": instruction,
        "selected_combobox": selected_combo,
        "selected_option": selected_option,
        "actions": actions,
        "reward": reward,
        "terminated": terminated,
        "error": error,
        "task_info": _task_info(info),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe BrowserGym MiniWoB choose-list select APIs without LLM")
    parser.add_argument("--env-id", default="browsergym/miniwob.choose-list")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    methods = ["select_option_text", "select_option_list_text", "click_combo_click_option", "focus_arrow_enter", "click_arrow_enter"]
    for method in methods:
        print(json.dumps(_run_method(args.env_id, args.seed, method), ensure_ascii=False))


if __name__ == "__main__":
    main()
