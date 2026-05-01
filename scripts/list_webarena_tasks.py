#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from app.browsergym_integration.webarena_tasks import discover_webarena_tasks


def main() -> int:
    out = Path("artifacts/browsergym/webarena_task_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tasks, diagnostics = discover_webarena_tasks()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    payload = {"ok": True, "count": len(tasks), "tasks": tasks, "diagnostics": diagnostics}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "count": len(tasks), "output": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
