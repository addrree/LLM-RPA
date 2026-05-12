#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner


def _safe(value: Any, *, max_chars: int = 1200) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_chars]
    if getattr(value, "shape", None) is not None:
        return {"kind": type(value).__name__, "shape": tuple(value.shape), "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, dict):
        return {str(k): _safe(v, max_chars=max_chars) for k, v in list(value.items())[:40] if k not in {"screenshot", "image"}}
    if isinstance(value, (list, tuple)):
        return [_safe(v, max_chars=max_chars) for v in list(value)[:40]]
    return str(value)[:max_chars]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a BrowserGym observation without dumping raw screenshots")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--task-kwargs", default="{}", help="JSON task_kwargs passed to gym.make")
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    if "miniwob" in args.env_id.lower():
        importlib.import_module("browsergym.miniwob")
    if "webarena" in args.env_id.lower():
        importlib.import_module("browsergym.webarena")

    task_kwargs = json.loads(args.task_kwargs or "{}")
    try:
        env = gym.make(args.env_id, task_kwargs=task_kwargs)
    except TypeError:
        env = gym.make(args.env_id)
    try:
        obs, info = env.reset()
        context = browsergym_obs_to_page_context(obs, info)
        unwrapped = getattr(env, "unwrapped", None)
        safe_unwrapped = {}
        for name in ("task_id", "task_name", "subdomain", "instance", "action_set", "action_mapping"):
            if unwrapped is not None and hasattr(unwrapped, name):
                safe_unwrapped[name] = _safe(getattr(unwrapped, name))
        payload = {
            "env_id": args.env_id,
            "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
            "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
            "action_space": _safe(getattr(env, "action_space", None)),
            "action_syntax": BrowserGymRunner._extract_action_syntax(env),
            "env_unwrapped_safe_fields": safe_unwrapped,
            "task_info": _safe((info or {}).get("task_info") if isinstance(info, dict) else None),
            "goal_instruction": context.get("goal_instruction"),
            "axtree_excerpt": context.get("axtree_excerpt"),
            "pruned_html_excerpt": context.get("pruned_html_excerpt"),
            "visible_text_excerpt": context.get("visible_text_excerpt"),
            "clickable_candidates_count": context.get("clickable_candidates_count"),
            "clickable_candidates": context.get("clickable_candidates"),
            "observation_summary": context.get("observation_summary"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
