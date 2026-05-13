from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from app.browsergym_integration.action_mapper import browsergym_finish_action, task_step_to_browsergym_action
from app.browsergym_integration.errors import UnsupportedBrowserGymActionError
from app.browsergym_integration.miniwob_grounding import find_submit_button, ground_miniwob_action, map_login_textboxes, parse_quoted_strings, real_candidate_bid, textbox_candidates
from app.browsergym_integration.local_extractor import (
    extract_pattern_from_observation,
    extract_structured_items_from_observation,
    extract_text_from_observation,
    extract_value_near_anchor_from_observation,
)
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context, page_context_to_snapshot_like
from app.browsergym_integration.plan_normalizer import normalize_plan_for_browsergym
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
    miniwob_instruction: str | None = None
    action_string: str | None = None
    mapping_error: str | None = None
    action_string_before_mapping: str | None = None
    action_string_after_mapping: str | None = None
    selected_candidate: dict | None = None
    selected_candidate_bid: str | None = None
    bid_source: str | None = None
    clickable_candidates_count: int | None = None
    page_candidate_extraction_failed: bool | None = None
    mapping_strategy: str | None = None
    fallback_used: bool = False
    fallback_type: str | None = None
    fallback_reward: float | None = None
    fallback_terminated: bool | None = None


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
        env_id: str | None = None,
        benchmark: str | None = None,
    ):
        self.planner = planner
        self.replanner = replanner
        self.validator = validator
        self.verifier = verifier
        self.max_steps = max_steps
        self.two_stage_planning = two_stage_planning
        self.use_vision = use_vision
        self.env_id = env_id
        self.benchmark = benchmark
        self.browsergym_action_syntax: list[str] = []

    def set_browsergym_context(self, *, env_id: str | None = None, benchmark: str | None = None) -> None:
        self.env_id = env_id
        self.benchmark = benchmark

    def set_browsergym_action_syntax(self, action_syntax: list[str] | None = None) -> None:
        self.browsergym_action_syntax = list(action_syntax or [])

    @property
    def uses_direct_action_mode(self) -> bool:
        return self._is_miniwob_context()

    def _is_miniwob_context(self) -> bool:
        env_id = (self.env_id or "").lower()
        benchmark = (self.benchmark or "").lower()
        return benchmark == "miniwob" or env_id.startswith("browsergym/miniwob.")

    def _default_action_syntax_examples(self) -> list[str]:
        if self.browsergym_action_syntax:
            return self.browsergym_action_syntax[:20]
        if self._is_miniwob_context():
            return [
                'click("bid", "left")',
                'click("bid")',
                'mouse_click(x, y, "left")',
                'fill("bid", "text")',
                'press("bid", "Enter")',
                'focus("bid")',
                'clear("bid")',
                'keyboard_type("text")',
                'keyboard_insert_text("text")',
                'noop()',
            ]
        return [
            "click(element_id)",
            "click(x, y)",
            "fill(element_id, 'text')",
            "press('Enter')",
            "scroll(0, 200)",
            "noop()",
        ]

    @staticmethod
    def _json_safe(value: Any, *, max_chars: int = 1200) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:max_chars]
        if getattr(value, "shape", None) is not None:
            return {"kind": type(value).__name__, "shape": tuple(value.shape), "dtype": str(getattr(value, "dtype", ""))}
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in list(value.items())[:20]:
                out[str(key)] = BrowserGymAgentAdapter._json_safe(item, max_chars=max_chars)
            return out
        if isinstance(value, (list, tuple)):
            return [BrowserGymAgentAdapter._json_safe(item, max_chars=max_chars) for item in list(value)[:20]]
        return str(value)[:max_chars]

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _call_direct_action_model(self, system_prompt: str, user_prompt: str, images: list[str] | None) -> dict[str, Any]:
        llm_client = getattr(self.planner, "llm_client", None)
        if llm_client is not None:
            if hasattr(llm_client, "generate_planner_json"):
                return llm_client.generate_planner_json(system_prompt, user_prompt, images_base64=images)
            if hasattr(llm_client, "generate_json"):
                return llm_client.generate_json(system_prompt, user_prompt)
        if hasattr(self.planner, "generate_planner_json"):
            return self.planner.generate_planner_json(system_prompt, user_prompt, images_base64=images)
        raise RuntimeError("MiniWoB direct action mode requires a planner with an llm_client capable of JSON generation")

    @staticmethod
    def _validate_direct_action(action: str) -> tuple[str, str | None]:
        normalized = " ".join(str(action or "").strip().split())
        if not normalized:
            return "noop()", "action_mapping_failure: empty action"
        if normalized.lower().startswith(("finish(", "agent_finish(")):
            return "noop()", "action_mapping_failure: finish is disabled for MiniWoB; success requires reward > 0"
        if not re.match(r"^(click|mouse_click|fill|type|press|focus|clear|keyboard_press|keyboard_type|keyboard_insert_text|scroll|noop|wait)\s*\(.*\)\s*$", normalized):
            return "noop()", f"action_mapping_failure: unsupported MiniWoB action syntax: {normalized[:120]}"
        return normalized, None

    @staticmethod
    def _mini_wob_no_progress_signal(history: list[dict]) -> dict | None:
        if not history:
            return None
        last = history[-1]
        action = str(last.get("action") or "").strip()
        if not action:
            return None
        reward = float(last.get("reward") or 0)
        repeat_count = 0
        for item in reversed(history):
            if str(item.get("action") or "").strip() == action and float(item.get("reward") or 0) <= 0:
                repeat_count += 1
            else:
                break
        if reward <= 0:
            return {"previous_action_had_no_effect": True, "action": action, "repeat_count": repeat_count}
        return None

    def _act_miniwob_direct(self, goal: str, obs: dict, info: dict | None, history: list[dict]) -> BrowserGymAgentDecision:
        context = browsergym_obs_to_page_context(obs, info)
        snapshot_like = page_context_to_snapshot_like(context)
        image_base64 = extract_browsergym_image_base64(obs, info) if self.use_vision else None
        vision_image_present = image_base64 is not None
        miniwob_instruction = context.get("goal_instruction") or goal or "Complete the MiniWoB task according to the page instruction"
        action_examples = self._default_action_syntax_examples()
        candidates_for_state = list(context.get("clickable_candidates") or [])
        submit_candidate = find_submit_button(candidates_for_state)
        login_textbox_map = map_login_textboxes(miniwob_instruction, candidates_for_state)
        text_action_hints = {
            "quoted_strings": parse_quoted_strings(miniwob_instruction),
            "textbox_bids_in_order": [real_candidate_bid(c) for c in textbox_candidates(candidates_for_state) if real_candidate_bid(c)],
            "login_textbox_bids": {key: real_candidate_bid(value) for key, value in login_textbox_map.items() if real_candidate_bid(value)},
            "submit_or_login_bid": real_candidate_bid(submit_candidate) if submit_candidate else "",
        }
        current_state = {
            "benchmark": "MiniWoB++",
            "env_id": self.env_id or "",
            "task_instruction": miniwob_instruction,
            "current_url": snapshot_like.get("url", ""),
            "visible_text_excerpt": str(snapshot_like.get("page_text", "") or context.get("axtree_excerpt", ""))[:1200],
            "axtree_excerpt": str(context.get("axtree_excerpt", ""))[:1200],
            "buttons": snapshot_like.get("buttons", [])[:10],
            "links": snapshot_like.get("links", [])[:10],
            "clickable_candidates": snapshot_like.get("clickable_candidates", [])[:30],
            "clickable_candidates_count": context.get("clickable_candidates_count", 0),
            "page_candidate_extraction_failed": bool(obs.get("page_candidate_extraction_failed")) if isinstance(obs, dict) else False,
            "obs_keys": context.get("obs_keys", []),
            "info_keys": context.get("info_keys", []),
            "vision_enabled": self.use_vision,
            "vision_image_present": vision_image_present,
            "screenshot_summary": context.get("screenshot_summary"),
            "image_summary": context.get("image_summary"),
            "recent_actions": self._json_safe(history[-5:]),
            "no_progress_signal": self._json_safe(self._mini_wob_no_progress_signal(history)),
            "available_action_syntax_examples": action_examples,
            "text_action_hints": text_action_hints,
        }
        system_prompt = (
            "You are controlling a BrowserGym MiniWoB++ environment. "
            "Choose exactly one next browser action for the current observation. "
            "Do not produce a plan and do not call finish; MiniWoB success is determined only by environment reward. "
            "Do NOT invent click(submit) when BrowserGym requires an element id. "
            "Do not output Unicode(). Unicode is only the type of the action space, not an action. "
            "For text input tasks, use fill(\"<textbox_bid>\", \"<required_text>\"). "
            "Do not repeatedly click a textbox when the task requires entering text. "
            "After filling required fields, click Submit/Login by real bid. "
            "For username/password tasks, fill username field first, then password field, then Login. "
            "Prefer real candidate.bid over coordinates. "
            "Prefer real candidate.bid for clicking. "
            "Use click(\"<bid>\", \"left\") when bid exists. "
            "Never infer bid from candidate index. "
            "Use coordinates only when no real bid exists. "
            "Do not treat raw DOM id as BrowserGym bid unless it is explicitly from bid/ref/data-testid/browsergym_id/data-bid. "
            "Coordinates in clickable_candidates include browsergym_center_x/browsergym_center_y; use those for BrowserGym mouse_click and do not use page_center_x/page_center_y directly. "
            "If candidates include a button with name/text matching the instruction, choose that candidate. "
            "Return STRICT JSON only with keys rationale, target_text, target_bid, and action."
        )
        user_prompt = (
            "Select the single next MiniWoB action. Use one of the available action syntaxes exactly as supported by this BrowserGym version.\n"
            "Do not output Unicode(). Unicode is only the type of the action space, not an action.\n"
            "For text input tasks, use fill(\"<textbox_bid>\", \"<required_text>\").\n"
            "Do not repeatedly click a textbox when the task requires entering text.\n"
            "After filling required fields, click Submit/Login by real bid.\n"
            "For username/password tasks, fill username field first, then password field, then Login.\n"
            "Prefer real candidate.bid over coordinates.\n"
            "Never infer bid from candidate index.\n"
            "Use one valid JSON object only. No prose outside JSON.\n"
            "For click actions:\n"
            "- Prefer real candidate.bid for clicking.\n"
            "- Use click(\"<bid>\", \"left\") when bid exists.\n"
            "- Never infer bid from candidate index.\n"
            "- Use coordinates only when no real bid exists.\n"
            "Do not use raw DOM id as BrowserGym bid unless it is explicitly bid/ref/data-testid/browsergym_id/data-bid. "
            "If there is no real bid but a candidate has browsergym_center_x/browsergym_center_y, use mouse_click(browsergym_center_x, browsergym_center_y, \"left\"). "
            "Do NOT use page_center_x/page_center_y directly for BrowserGym mouse_click. If no scaled coordinates are present, fall back to action-space center_x/center_y. "
            "Do NOT return click(submit) unless the action_space explicitly says text labels are valid.\n"
            "If clickable_candidates contains a button whose name/text/label matches target_text and has a real bid, set target_bid to that candidate bid and use it in action.\n"
            "Do not repeat exactly the same previous action after reward=0 unless new evidence changed.\n"
            "If unsure, choose the safest grounded interaction; use noop() only when no valid grounded action is possible.\n\n"
            f"Current state JSON:\n{json.dumps(current_state, ensure_ascii=False, indent=2)}\n\n"
            "Return exactly: {\"rationale\": \"...\", \"target_text\": \"submit\", \"target_bid\": \"...\", \"action\": \"click(\\\"...\\\", \\\"left\\\")\"}"
        )
        images = [image_base64] if image_base64 is not None else None
        mapping_error = None
        try:
            parsed = self._call_direct_action_model(system_prompt, user_prompt, images)
        except Exception as exc:
            parsed = {}
            mapping_error = f"action_mapping_failure: model error: {exc}"
        if isinstance(parsed, str):
            parsed = self._extract_json_object(parsed)
        elif not isinstance(parsed, dict):
            parsed = {}
        rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip()
        raw_action = str(parsed.get("action") or "").strip()
        action, validation_error = self._validate_direct_action(raw_action)
        mapping_error = mapping_error or validation_error
        before_mapping = action
        selected_candidate = None
        mapping_strategy = None
        if not validation_error:
            grounding = ground_miniwob_action(
                action=action,
                parsed_response=parsed,
                candidates=list(context.get("clickable_candidates") or []),
                history=history,
                action_syntax=self.browsergym_action_syntax,
            )
            action = grounding.action
            selected_candidate = grounding.selected_candidate
            mapping_strategy = grounding.mapping_strategy
            if grounding.mapping_error:
                mapping_error = grounding.mapping_error
            if grounding.repeated_warning and rationale:
                rationale = f"{rationale} | {grounding.repeated_warning}"
        if mapping_error and not rationale:
            rationale = mapping_error
        return BrowserGymAgentDecision(
            action=action,
            internal_plan=None,
            selected_step=None,
            rationale=rationale,
            finish=False,
            answer=None,
            vision_used=self.use_vision,
            vision_image_present=vision_image_present,
            miniwob_instruction=miniwob_instruction,
            action_string=action,
            mapping_error=mapping_error,
            action_string_before_mapping=before_mapping,
            action_string_after_mapping=action,
            selected_candidate=selected_candidate,
            selected_candidate_bid=real_candidate_bid(selected_candidate),
            bid_source=selected_candidate.get("bid_source") if isinstance(selected_candidate, dict) else None,
            clickable_candidates_count=int(context.get("clickable_candidates_count", 0) or 0),
            page_candidate_extraction_failed=bool(obs.get("page_candidate_extraction_failed")) if isinstance(obs, dict) else False,
            mapping_strategy=mapping_strategy,
        )

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
        if self._is_miniwob_context():
            return self._act_miniwob_direct(goal, obs, info, history)
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
            "goal_instruction": context.get("goal_instruction", ""),
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
        normalize_plan_for_browsergym(
            plan,
            env_id=self.env_id,
            benchmark=self.benchmark,
            current_url=str(snapshot_like.get("url", "")),
        )
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
