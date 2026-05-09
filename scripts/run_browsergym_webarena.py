#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from app.browsergym_integration import BrowserGymAgentAdapter, BrowserGymRunConfig, BrowserGymRunner
from app.browsergym_integration.config import validate_webarena_env_vars
from app.main import build_llm_client
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run BrowserGym WebArena task")
    p.add_argument("--env-id", required=True)
    p.add_argument("--goal", required=True)
    p.add_argument("--backend", default="ollama_cloud")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--use-vision", action="store_true", help="Send BrowserGym screenshot to the planner LLM payload")
    return p.parse_args(argv)


def main():
    args = parse_args()
    try:
        import gymnasium  # noqa: F401
        import browsergym.core  # noqa: F401
        import browsergym.webarena  # noqa: F401
    except Exception:
        print(json.dumps({"status": "skipped", "reason": "Install browsergym-webarena and dependencies"}, ensure_ascii=False))
        return

    check = validate_webarena_env_vars(args.env_id)
    if not check["ok"]:
        print(json.dumps({"status": "skipped", "missing": check["missing"], "reason": check["message"]}, ensure_ascii=False, indent=2))
        return

    llm = build_llm_client(backend=args.backend)

    def agent_factory():
        return BrowserGymAgentAdapter(Planner(llm), Replanner(llm), PlanValidator(), LLMVerifier(llm), max_steps=args.max_steps, use_vision=args.use_vision)

    report = BrowserGymRunner(agent_factory=agent_factory, config=BrowserGymRunConfig(env_id=args.env_id, goal=args.goal, backend=args.backend, max_steps=args.max_steps, use_vision=args.use_vision)).run_one()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
