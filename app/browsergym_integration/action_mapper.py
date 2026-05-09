from __future__ import annotations

from app.browsergym_integration.errors import UnsupportedBrowserGymActionError

# Single policy point for syntax evolution across BrowserGym benchmarks.  The
# installed BrowserGym package is optional in CI; this sidecar keeps the same
# high-level text DSL shape already used by click/type/finish smoke tests.
BROWSERGYM_ACTION_SYNTAX_VERSION = "v1_text_sidecar"


def task_step_to_browsergym_action(step) -> str:
    payload = step.model_dump(mode="json") if hasattr(step, "model_dump") else dict(step)
    action = str(payload.get("action", "")).strip()
    args = payload.get("args") or {}

    if action == "click":
        if args.get("text"):
            return f"click(text={args['text']!r})"
        if args.get("name"):
            return f"click(name={args['name']!r})"
        if args.get("href_contains"):
            return f"click(href_contains={args['href_contains']!r})"
        if args.get("selector"):
            return f"click(selector={args['selector']!r})"
        raise UnsupportedBrowserGymActionError("click step missing supported targeting args")
    if action in {"type", "fill"}:
        selector = args.get("selector")
        text = args.get("text")
        if text is None and "value" in args:
            text = args.get("value")
        if not selector or text is None:
            raise UnsupportedBrowserGymActionError(f"{action} requires selector and text/value")
        return f"type(selector={selector!r}, text={text!r})"
    if action == "press":
        key = args.get("key") if args.get("key") is not None else args.get("text")
        if key is None:
            raise UnsupportedBrowserGymActionError("press requires key or text")
        selector = args.get("selector")
        if selector:
            return f"press(selector={selector!r}, key={key!r})"
        return f"press(key={key!r})"
    if action == "scroll":
        direction = str(args.get("direction") or args.get("scroll_direction") or "down").lower()
        if direction not in {"down", "up", "left", "right"}:
            raise UnsupportedBrowserGymActionError("scroll direction must be one of down/up/left/right")
        return f"scroll(direction={direction!r})"
    if action in {"wait_for", "noop"}:
        return "noop()"
    if action == "finish":
        return browsergym_finish_action(args.get("answer"))
    if action.startswith("extract_"):
        raise UnsupportedBrowserGymActionError("extraction actions are internal-only for BrowserGym loop")
    raise UnsupportedBrowserGymActionError(f"Unsupported action for BrowserGym mapper: {action}")


def browsergym_finish_action(answer: str | None = None) -> str:
    safe_answer = (answer or "").strip()
    if safe_answer:
        return f"finish(answer={safe_answer!r})"
    return "finish(answer='No final answer produced')"


def normalize_browsergym_action(action: str) -> str:
    return " ".join(str(action).strip().split())
