from __future__ import annotations

import importlib
import importlib.resources as resources
import json
import pkgutil
from pathlib import Path
from typing import Any

LLM_JUDGE_TOKENS = ("fuzzy_match", "llm_fuzzy_match", "llm_judge", "openai", "gpt")
DISCOVERY_PACKAGES = ("browsergym.webarena", "browsergym_webarena", "libwebarena")


def _contains_llm_judge(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_llm_judge(v) or _contains_llm_judge(k) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_llm_judge(v) for v in value)
    if isinstance(value, str):
        low = value.lower()
        return any(token in low for token in LLM_JUDGE_TOKENS)
    return False


def classify_requires_llm_judge(task_config: dict[str, Any]) -> bool:
    return _contains_llm_judge(task_config)


def _iter_candidate_paths(module) -> list[Path]:
    candidates: set[Path] = set()
    module_file = getattr(module, "__file__", None)
    if module_file:
        root = Path(module_file).resolve().parent
        candidates.update([root, root / "configs", root / "config", root / "tasks"])

    for package_name in DISCOVERY_PACKAGES:
        try:
            files = resources.files(package_name)
            candidates.add(Path(str(files)))
        except Exception:
            continue
    return [p for p in candidates if p.exists()]


def discover_webarena_tasks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"packages": {}, "searched_paths": []}

    imported = {}
    for package_name in DISCOVERY_PACKAGES:
        try:
            imported[package_name] = importlib.import_module(package_name)
            diagnostics["packages"][package_name] = {
                "imported": True,
                "file": getattr(imported[package_name], "__file__", None),
            }
        except Exception as exc:
            diagnostics["packages"][package_name] = {"imported": False, "error": str(exc)}

    if "browsergym.webarena" not in imported:
        raise RuntimeError(f"browsergym.webarena import failed: {json.dumps(diagnostics, ensure_ascii=False)}")

    module = imported["browsergym.webarena"]
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Gym registry scan first (most robust across package layouts)
    try:
        import gymnasium as gym

        for env_id in sorted(gym.envs.registry.keys()):
            if "webarena" not in env_id.lower():
                continue
            if env_id in seen:
                continue
            seen.add(env_id)
            tasks.append(
                {
                    "task_id": env_id,
                    "env_id": env_id,
                    "sites": [],
                    "intent": "",
                    "evaluator_types": [],
                    "requires_llm_judge": False,
                    "source": "gym_registry",
                }
            )
    except Exception as exc:
        diagnostics["gym_registry_error"] = str(exc)

    # Config file scan for metadata + llm judge markers
    for base in _iter_candidate_paths(module):
        diagnostics["searched_paths"].append(str(base))
        for p in base.rglob("*.json"):
            if "task" not in p.name.lower() and "config" not in p.name.lower():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries = data if isinstance(data, list) else [data]
            for item in entries:
                if not isinstance(item, dict):
                    continue
                env_id = str(item.get("env_id") or item.get("id") or "")
                task_id = str(item.get("task_id") or item.get("task") or env_id or p.stem)
                if not env_id:
                    env_id = task_id if "webarena" in task_id.lower() else f"browsergym/webarena.{task_id}"
                signature = f"{task_id}::{env_id}"
                if signature in seen:
                    continue
                seen.add(signature)
                evaluators = item.get("evaluator_types") or item.get("evaluators") or []
                if isinstance(evaluators, str):
                    evaluators = [evaluators]
                sites = item.get("sites") or item.get("site") or []
                if isinstance(sites, str):
                    sites = [sites]
                task = {
                    "task_id": task_id,
                    "env_id": env_id,
                    "sites": sites,
                    "intent": str(item.get("intent") or item.get("goal") or ""),
                    "evaluator_types": evaluators,
                    "requires_llm_judge": classify_requires_llm_judge(item),
                    "source": str(p),
                }
                tasks.append(task)

    if not tasks:
        diagnostics["iter_modules"] = [m.name for m in pkgutil.iter_modules() if "browsergym" in m.name or "webarena" in m.name]
        raise RuntimeError(f"WebArena task configs were not discovered. Diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")

    return tasks, diagnostics
