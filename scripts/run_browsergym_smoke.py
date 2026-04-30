#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from app.browsergym_integration import BrowserGymAgentAdapter, BrowserGymRunConfig, BrowserGymRunner
from app.main import build_llm_client
from app.planner.planner import Planner
from app.planner.replanner import Replanner
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


def parse_args():
    p = argparse.ArgumentParser(description="Run BrowserGym openended smoke")
    p.add_argument("--env-id", required=True)
    p.add_argument("--start-url", default=None)
    p.add_argument("--goal", required=True)
    p.add_argument("--backend", default="ollama_cloud")
    p.add_argument("--max-steps", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import gymnasium  # noqa: F401
        import browsergym.core  # noqa: F401
    except Exception as exc:
        print(f"Missing BrowserGym dependencies: {exc}\nInstall:\n  pip install browsergym gymnasium\n  playwright install")
        return

    llm = build_llm_client(backend=args.backend)

    def agent_factory():
        return BrowserGymAgentAdapter(
            planner=Planner(llm),
            replanner=Replanner(llm),
            validator=PlanValidator(),
            verifier=LLMVerifier(llm),
            max_steps=args.max_steps,
            two_stage_planning=True,
        )

    task_kwargs = {"start_url": args.start_url} if args.env_id == "browsergym/openended" and args.start_url else None
    runner = BrowserGymRunner(agent_factory=agent_factory, config=BrowserGymRunConfig(env_id=args.env_id, goal=args.goal, backend=args.backend, max_steps=args.max_steps, task_kwargs=task_kwargs))
    report = runner.run_one()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
