from __future__ import annotations

from dataclasses import dataclass

from app.browsergym_integration.action_mapper import browsergym_finish_action, task_step_to_browsergym_action
from app.browsergym_integration.errors import UnsupportedBrowserGymActionError
from app.browsergym_integration.local_extractor import (
    extract_pattern_from_observation,
    extract_structured_items_from_observation,
    extract_text_from_observation,
    extract_value_near_anchor_from_observation,
)
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context, page_context_to_snapshot_like
from app.browsergym_integration.vision import extract_browsergym_image_base64
from app.schemas.task_spec import TaskSpec


@dataclass
class BrowserGymAgentDecision:
    action: str
    internal_plan: dict | None = None
    selected_step: dict | None = None
    rationale: str = ""
    finish: bool = False
    answer: str | None = None
    extracted_value: str | None = None
    vision_used: bool = False
    vision_image_present: bool = False


class BrowserGymAgentAdapter:
    def __init__(
        self,
        planner,
        replanner,
        validator,
        verifier=None,
        max_steps: int = 15,
        two_stage_planning: bool = True,
        use_vision: bool = False,
    ):
        self.planner = planner
        self.replanner = replanner
        self.validator = validator
        self.verifier = verifier
        self.max_steps = max_steps
        self.two_stage_planning = two_stage_planning
        self.use_vision = use_vision

    def _extract_local(self, action: str, args: dict, compact_snapshot: dict, goal: str) -> str:
        if action == "extract_text":
            return extract_text_from_observation(compact_snapshot, selector=args.get("selector"), goal=goal)
        if action == "extract_structured_items":
            return ", ".join(extract_structured_items_from_observation(compact_snapshot))
        if action == "extract_pattern_from_page_text":
            return extract_pattern_from_observation(compact_snapshot, pattern=str(args.get("pattern", "")), case_insensitive=bool(args.get("case_insensitive", False)))
        if action == "extract_value_near_anchor":
            return extract_value_near_anchor_from_observation(compact_snapshot, anchor_candidates=args.get("anchor_candidates"), value_type=args.get("value_type"))
        if action == "extract_section_lines":
            return extract_value_near_anchor_from_observation(compact_snapshot, anchor_candidates=[str(args.get("section", ""))])
        return ""

    def act(self, goal: str, obs: dict, info: dict | None, history: list[dict]) -> BrowserGymAgentDecision:
        context = browsergym_obs_to_page_context(obs, info)
        snapshot_like = page_context_to_snapshot_like(context)
        image_base64 = extract_browsergym_image_base64(obs, info) if self.use_vision else None
        vision_image_present = image_base64 is not None
        compact_snapshot = {
            "url": snapshot_like.get("url", ""),
            "title": snapshot_like.get("title", ""),
            "open_pages_titles": context.get("open_pages_titles", []),
            "visible_headings": snapshot_like.get("visible_headings", [])[:5],
            "links": snapshot_like.get("links", [])[:10],
            "buttons": snapshot_like.get("buttons", [])[:10],
            "text_excerpt": str(snapshot_like.get("page_text", ""))[:900],
            "axtree_excerpt": context.get("axtree_excerpt", "")[:900],
            "source": "browsergym",
        }
        if self.use_vision:
            compact_snapshot["vision_enabled"] = True
            compact_snapshot["vision_image_present"] = vision_image_present
            compact_snapshot["screenshot_summary"] = context.get("screenshot_summary")
            compact_snapshot["image_summary"] = context.get("image_summary")
        history_excerpt = history[-5:]
        vision_note = ""
        if self.use_vision:
            vision_note = (
                "\n\nVision mode: the planner receives the current screenshot image separately in the LLM payload. "
                "Do not include or request base64 image data in the JSON plan."
                if vision_image_present
                else "\n\nVision mode was requested, but no screenshot image was available; use the text/AX context only."
            )
        prompt_goal = f"{goal}{vision_note}\n\nCurrent page context:\n{compact_snapshot}\n\nRecent action history (latest last):\n{history_excerpt}"

        images = [image_base64] if image_base64 is not None else None
        if images is not None:
            plan: TaskSpec = self.planner.build_plan(prompt_goal, images_base64=images)
        else:
            plan = self.planner.build_plan(prompt_goal)
        self.validator.validate(plan)
        step = next((s for s in plan.steps if s.action in {"click", "type", "wait_for", "finish", "press", "fill", "scroll", "noop"} or s.action.startswith("extract_")), None)
        if step is None:
            return BrowserGymAgentDecision(
                action="noop()",
                internal_plan=plan.model_dump(mode="json"),
                rationale="no step",
                vision_used=self.use_vision,
                vision_image_present=vision_image_present,
            )
        if step.action.startswith("extract_"):
            extracted = self._extract_local(step.action, step.args or {}, compact_snapshot, goal).strip()
            final_answer = extracted or "No final answer produced"
            return BrowserGymAgentDecision(
                action=browsergym_finish_action(final_answer),
                internal_plan=plan.model_dump(mode="json"),
                selected_step=step.model_dump(mode="json"),
                finish=bool(extracted),
                answer=final_answer if extracted else None,
                extracted_value=extracted,
                vision_used=self.use_vision,
                vision_image_present=vision_image_present,
            )
        try:
            mapped = task_step_to_browsergym_action(step)
        except UnsupportedBrowserGymActionError:
            return BrowserGymAgentDecision(
                action="noop()",
                internal_plan=plan.model_dump(mode="json"),
                selected_step=step.model_dump(mode="json"),
                rationale="action mapping failure",
                vision_used=self.use_vision,
                vision_image_present=vision_image_present,
            )
        return BrowserGymAgentDecision(
            action=mapped,
            internal_plan=plan.model_dump(mode="json"),
            selected_step=step.model_dump(mode="json"),
            finish=step.action == "finish",
            answer=step.args.get("answer") if step.action == "finish" else None,
            vision_used=self.use_vision,
            vision_image_present=vision_image_present,
        )
