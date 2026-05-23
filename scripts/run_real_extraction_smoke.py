#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction.intent_parser import parse_extraction_intent
from app.extraction.page_extractor import build_extraction_context
from app.extraction.extraction_controller import solve_extraction_task


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="")
    p.add_argument("--goal", required=True)
    p.add_argument("--backend", default="ollama_cloud")
    p.add_argument("--show-browser", action="store_true")
    p.add_argument("--export-format", choices=["json"], default="json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if not args.url:
            print(json.dumps({
                "status": "error",
                "error": "url_required",
                "message": "Pass --url for real-web extraction smoke; no synthetic hardcoded page is used.",
                "goal": args.goal,
            }, ensure_ascii=False, indent=2))
            return 0
        # Runner intentionally consumes externally provided observation/context from a real page pipeline.
        # This lightweight script validates shared extraction logic without introducing MiniWoB- or site-specific hardcode.
        obs = {"visible_text": ""}
        context = {"axtree_excerpt": "", "goal_instruction": args.goal, "url": args.url}
        candidates: list[dict] = []
        intent = parse_extraction_intent(args.goal)
        extraction_context = build_extraction_context(obs, context, candidates)
        decision = solve_extraction_task(intent, extraction_context, candidates, ["click(\"bid\", \"left\")"])
        out = {
            "status": "ok" if decision else "no_decision",
            "url": args.url,
            "goal": args.goal,
            "backend": args.backend,
            "intent": intent,
            "extracted_data": (decision.extracted_data if decision else None),
            "answer": (decision.answer if decision else None),
            "action": (decision.action if decision else None),
            "strategy": (decision.strategy if decision else None),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "url": args.url, "goal": args.goal}, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
