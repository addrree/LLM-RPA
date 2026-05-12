#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_grounding import ground_miniwob_action
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Try a grounded MiniWoB BrowserGym click without an LLM")
    parser.add_argument("--env-id", default="browsergym/miniwob.click-button")
    parser.add_argument("--target-text", default="Submit")
    args = parser.parse_args()

    import gymnasium as gym
    import browsergym.core  # noqa: F401
    importlib.import_module("browsergym.miniwob")

    env = gym.make(args.env_id)
    try:
        obs, info = env.reset()
        obs = BrowserGymRunner._augment_miniwob_observation_with_page_candidates(env, obs, info)
        ctx = browsergym_obs_to_page_context(obs, info)
        print("instruction:", ctx.get("goal_instruction"))
        print("candidates:", json.dumps(ctx.get("clickable_candidates", []), ensure_ascii=False, indent=2, default=str))
        result = ground_miniwob_action(
            action=f'click("{args.target_text}")',
            parsed_response={"target_text": args.target_text},
            candidates=ctx.get("clickable_candidates", []),
            action_syntax=BrowserGymRunner._extract_action_syntax(env),
        )
        print("trying_action:", result.action)
        print("selected_candidate:", json.dumps(result.selected_candidate, ensure_ascii=False, default=str))
        print("mapping_strategy:", result.mapping_strategy)
        print("mapping_error:", result.mapping_error)
        obs, reward, terminated, truncated, info = env.step(result.action)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
