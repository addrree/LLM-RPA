from __future__ import annotations

from dataclasses import dataclass

from app.browsergym_integration.action_mapper import browsergym_finish_action, task_step_to_browsergym_action
from app.browsergym_integration.errors import UnsupportedBrowserGymActionError
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context, page_context_to_snapshot_like
from app.schemas.task_spec import TaskSpec


@dataclass
class BrowserGymAgentDecision:
    action: str
    internal_plan: dict | None = None
    selected_step: dict | None = None
    rationale: str = ""
    finish: bool = False
    answer: str | None = None


class BrowserGymAgentAdapter:
    def __init__(self, planner, replanner, validator, verifier=None, max_steps: int = 15, two_stage_planning: bool = True):
        self.planner = planner
        self.replanner = replanner
        self.validator = validator
        self.verifier = verifier
        self.max_steps = max_steps
        self.two_stage_planning = two_stage_planning

    def act(self, goal: str, obs: dict, info: dict | None, history: list[dict]) -> BrowserGymAgentDecision:
        context = browsergym_obs_to_page_context(obs, info)
        snapshot_like = page_context_to_snapshot_like(context)
        compact_snapshot = {
            "url": snapshot_like.get("url", ""),
            "title": snapshot_like.get("title", ""),
            "visible_headings": snapshot_like.get("visible_headings", [])[:5],
            "links": snapshot_like.get("links", [])[:10],
            "buttons": snapshot_like.get("buttons", [])[:10],
            "page_text_excerpt": str(snapshot_like.get("page_text", ""))[:900],
            "source": "browsergym",
        }
        history_excerpt = history[-5:]
        prompt_goal = (
            f"{goal}\n\n"
            f"Current page context:\n{compact_snapshot}\n\n"
            f"Recent action history (latest last):\n{history_excerpt}"
        )

        plan: TaskSpec = self.planner.build_plan(prompt_goal)
        self.validator.validate(plan)
        step = next((s for s in plan.steps if s.action in {"click", "type", "wait_for", "finish"} or s.action.startswith("extract_")), None)
        if step is None:
            return BrowserGymAgentDecision(action="noop()", internal_plan=plan.model_dump(mode="json"), rationale="no step")
        if step.action.startswith("extract_"):
            answer = compact_snapshot.get("page_text_excerpt", "")[:500]
            return BrowserGymAgentDecision(
                action=browsergym_finish_action(answer),
                internal_plan=plan.model_dump(mode="json"),
                selected_step=step.model_dump(mode="json"),
                finish=True,
                answer=answer,
            )
        try:
            mapped = task_step_to_browsergym_action(step)
        except UnsupportedBrowserGymActionError:
            return BrowserGymAgentDecision(action="noop()", internal_plan=plan.model_dump(mode="json"), selected_step=step.model_dump(mode="json"), rationale="action mapping failure")
        return BrowserGymAgentDecision(action=mapped, internal_plan=plan.model_dump(mode="json"), selected_step=step.model_dump(mode="json"), finish=step.action == "finish", answer=step.args.get("answer") if step.action == "finish" else None)
