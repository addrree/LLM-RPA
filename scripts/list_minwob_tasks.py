#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.browsergym_integration.miniwob_tasks import build_minwob_inventory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="List registered BrowserGym MiniWoB++ tasks")
    parser.add_argument("--output", default="artifacts/browsergym/miniwob_task_inventory.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inventory = build_minwob_inventory()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    if not os.getenv("MINIWOB_URL"):
        print("WARNING: MINIWOB_URL is not set; env list may work, but running tasks requires MINIWOB_URL.")
    print(json.dumps({
        "total_envs": len(inventory),
        "output_path": str(output_path),
        "first_20_env_ids": [item["env_id"] for item in inventory[:20]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
