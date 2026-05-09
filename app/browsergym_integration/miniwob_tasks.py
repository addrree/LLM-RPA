from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

MINIWOB_ENV_PREFIX = "browsergym/miniwob."

RECOMMENDED_MINIWOB_TASK_NAMES = [
    "click-button",
    "click-button-sequence",
    "click-checkboxes",
    "click-dialog",
    "click-link",
    "click-menu",
    "click-option",
    "click-test",
    "enter-text",
    "focus-text",
    "login-user",
    "choose-list",
    "choose-date",
    "use-autocomplete",
    "book-flight",
]


def _ensure_miniwob_registered() -> None:
    """Import BrowserGym MiniWoB plugin so its Gymnasium envs are registered."""
    import browsergym.miniwob  # noqa: F401


def task_name_from_env_id(env_id: str) -> str:
    raw = str(env_id)
    if raw.startswith(MINIWOB_ENV_PREFIX):
        return raw[len(MINIWOB_ENV_PREFIX) :]
    return raw.rsplit(".", 1)[-1]


def env_id_for_task_name(task_name: str) -> str:
    task_name = str(task_name).strip()
    if task_name.startswith(MINIWOB_ENV_PREFIX):
        return task_name
    return f"{MINIWOB_ENV_PREFIX}{task_name}"


def list_minwob_env_ids() -> list[str]:
    """Return registered BrowserGym MiniWoB env IDs sorted by task name.

    BrowserGym registers MiniWoB IDs as ``browsergym/miniwob.<task-name>``.
    If the optional ``browsergym-miniwob`` package is unavailable this returns
    an empty list so list/filter tests and skipped reports do not require the
    real benchmark installation.
    """
    try:
        _ensure_miniwob_registered()
        from gymnasium.envs.registration import registry
    except Exception:
        return []

    return sorted(str(env_id) for env_id in registry.keys() if str(env_id).startswith(MINIWOB_ENV_PREFIX))


def build_minwob_inventory() -> list[dict[str, Any]]:
    return [
        {
            "env_id": env_id,
            "task_name": task_name_from_env_id(env_id),
            "benchmark": "miniwob",
            "requires_external_server": False,
            "requires_wa_env": False,
            "requires_llm_judge": False,
        }
        for env_id in list_minwob_env_ids()
    ]


def _normalize_patterns(patterns: str | Iterable[str] | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [part.strip() for part in patterns.split(",") if part.strip()]
    return [str(part).strip() for part in patterns if str(part).strip()]


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def select_minwob_subset(
    env_ids: Iterable[str],
    limit: int | None = None,
    task_ids: str | Iterable[str] | None = None,
    include_patterns: str | Iterable[str] | None = None,
    exclude_patterns: str | Iterable[str] | None = None,
) -> list[str]:
    """Filter MiniWoB env IDs by explicit tasks, include/exclude regexes, and limit.

    ``task_ids`` may contain full env IDs or bare MiniWoB task names. If no
    explicit task/include filter is supplied, a stable recommended subset is
    used when any recommended tasks exist in the registry; otherwise all envs
    are considered. Missing recommended tasks are ignored.
    """
    unique_env_ids = list(dict.fromkeys(str(env_id) for env_id in env_ids))
    by_env = {env_id: env_id for env_id in unique_env_ids}
    by_task = {task_name_from_env_id(env_id): env_id for env_id in unique_env_ids}

    requested = _normalize_patterns(task_ids)
    include = _normalize_patterns(include_patterns)
    exclude = _normalize_patterns(exclude_patterns)

    if requested:
        selected: list[str] = []
        for item in requested:
            env_id = by_env.get(item) or by_task.get(item) or by_env.get(env_id_for_task_name(item))
            if env_id is not None:
                selected.append(env_id)
    elif include:
        selected = [
            env_id
            for env_id in unique_env_ids
            if _matches_any(env_id, include) or _matches_any(task_name_from_env_id(env_id), include)
        ]
    else:
        recommended = [env_id_for_task_name(task_name) for task_name in RECOMMENDED_MINIWOB_TASK_NAMES]
        selected = [env_id for env_id in recommended if env_id in by_env]
        if not selected:
            selected = unique_env_ids

    if exclude:
        selected = [
            env_id
            for env_id in selected
            if not (_matches_any(env_id, exclude) or _matches_any(task_name_from_env_id(env_id), exclude))
        ]

    selected = list(dict.fromkeys(selected))
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected
