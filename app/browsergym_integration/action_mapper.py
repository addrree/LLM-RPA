from __future__ import annotations

from app.browsergym_integration.errors import UnsupportedBrowserGymActionError


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
    if action == "type":
        selector = args.get("selector")
        text = args.get("text")
        if not selector or text is None:
            raise UnsupportedBrowserGymActionError("type requires selector and text")
        return f"type(selector={selector!r}, text={text!r})"
    if action in {"wait_for", "noop"}:
        return "noop()"
    if action == "finish":
        return browsergym_finish_action(args.get("answer"))
    if action.startswith("extract_"):
        raise UnsupportedBrowserGymActionError("extraction actions are internal-only for BrowserGym loop")
    raise UnsupportedBrowserGymActionError(f"Unsupported action for BrowserGym mapper: {action}")


def browsergym_finish_action(answer: str | None = None) -> str:
    if answer:
        return f"finish(answer={answer!r})"
    return "finish()"


def normalize_browsergym_action(action: str) -> str:
    return " ".join(str(action).strip().split())
