from __future__ import annotations

import os
from urllib.parse import urlparse

from app.schemas.task_spec import TaskSpec


MINIWOB_ENV_PREFIX = "browsergym/miniwob."


def is_minwob_context(env_id: str | None = None, benchmark: str | None = None) -> bool:
    return (benchmark or "").lower() == "miniwob" or (env_id or "").lower().startswith(MINIWOB_ENV_PREFIX)


def normalize_minwob_click_targets(plan: TaskSpec, *, env_id: str | None = None, benchmark: str | None = None) -> TaskSpec:
    """Make MiniWoB text clicks deterministic without changing the global validator policy."""
    if not is_minwob_context(env_id=env_id, benchmark=benchmark):
        return plan

    for step in plan.steps:
        if step.action != "click":
            continue
        args = step.args or {}
        if not args.get("text"):
            continue
        has_exact = "exact" in args and args.get("exact") is not None
        has_scope = bool(str(args.get("scope_selector", "")).strip())
        has_role_name = bool(str(args.get("role", "")).strip() and str(args.get("name", "")).strip())
        has_href = bool(str(args.get("href_contains", "")).strip())
        if not (has_exact or has_scope or has_role_name or has_href):
            args["exact"] = True
            step.args = args
    return plan


def _domain_variants_from_url(url: str | None) -> list[str]:
    if not url:
        return []
    parsed = urlparse(str(url))
    hostname = parsed.hostname
    netloc = parsed.netloc
    if not hostname:
        return []
    variants = [hostname]
    if netloc and netloc != hostname:
        variants.append(netloc)
    return variants


def normalize_allowed_domains_for_browsergym(
    plan: TaskSpec,
    env_id: str | None = None,
    current_url: str | None = None,
    benchmark: str | None = None,
) -> TaskSpec:
    """Add hostname and host:port variants for BrowserGym MiniWoB localhost URLs."""
    if not is_minwob_context(env_id=env_id, benchmark=benchmark):
        return plan

    domains = list(plan.allowed_domains or [])
    seen = set(domains)
    candidate_urls = [str(plan.start_url), current_url, os.getenv("MINIWOB_URL")]
    for url in candidate_urls:
        for domain in _domain_variants_from_url(url):
            if domain in {"127.0.0.1", "localhost"} or domain.startswith("127.0.0.1:") or domain.startswith("localhost:"):
                if domain not in seen:
                    domains.append(domain)
                    seen.add(domain)
    plan.allowed_domains = domains
    return plan


def normalize_plan_for_browsergym(
    plan: TaskSpec,
    *,
    env_id: str | None = None,
    benchmark: str | None = None,
    current_url: str | None = None,
) -> TaskSpec:
    if not is_minwob_context(env_id=env_id, benchmark=benchmark):
        return plan
    normalize_minwob_click_targets(plan, env_id=env_id, benchmark=benchmark)
    normalize_allowed_domains_for_browsergym(plan, env_id=env_id, current_url=current_url, benchmark=benchmark)
    return plan
