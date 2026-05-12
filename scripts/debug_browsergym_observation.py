#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner


def _safe(value: Any, limit: int = 3000) -> Any:
    if value is None:
        return None
    if getattr(value, "shape", None) is not None:
        return {"kind": type(value).__name__, "shape": tuple(value.shape), "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:limit]
    except Exception:
        return str(value)[:limit]


def _safe_attrs(obj: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for name in ("page", "action_space", "action_mapping", "action_set", "high_level_action_set", "task", "task_entrypoint"):
        try:
            value = getattr(obj, name, None)
        except Exception as exc:
            attrs[name] = f"<error {exc}>"
            continue
        if value is not None:
            attrs[name] = repr(value)[:500]
    return attrs


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one BrowserGym observation without dumping raw screenshot arrays")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    if "miniwob" in args.env_id.lower():
        importlib.import_module("browsergym.miniwob")
    if "webarena" in args.env_id.lower():
        importlib.import_module("browsergym.webarena")

    env = gym.make(args.env_id)
    try:
        obs, info = env.reset()
        ctx = browsergym_obs_to_page_context(obs, info)
        payload = {
            "obs_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
            "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else [],
            "env_action_space": repr(getattr(env, "action_space", None)),
            "action_syntax_examples": BrowserGymRunner._extract_action_syntax(env),
            "unwrapped_type": str(type(getattr(env, "unwrapped", None))),
            "unwrapped_safe_attrs": _safe_attrs(getattr(env, "unwrapped", None)),
            "axtree_txt_first_3000": _safe((obs or {}).get("axtree_txt") if isinstance(obs, dict) else None),
            "axtree_object_first_3000": _safe((obs or {}).get("axtree_object") if isinstance(obs, dict) else None),
            "dom_object_first_3000": _safe((obs or {}).get("dom_object") if isinstance(obs, dict) else None),
            "pruned_html_first_3000": _safe((obs or {}).get("pruned_html") if isinstance(obs, dict) else None),
            "html_first_3000": _safe((obs or {}).get("html") if isinstance(obs, dict) else None),
            "text_first_3000": _safe((obs or {}).get("text") if isinstance(obs, dict) else None),
            "goal": _safe((obs or {}).get("goal") if isinstance(obs, dict) else None),
            "goal_object": _safe((obs or {}).get("goal_object") if isinstance(obs, dict) else None),
            "task_info": _safe((info or {}).get("task_info") if isinstance(info, dict) else None),
            "clickable_candidates_count": ctx.get("clickable_candidates_count"),
            "clickable_candidates": ctx.get("clickable_candidates"),
            "observation_summary": ctx.get("observation_summary"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
