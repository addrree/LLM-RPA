from __future__ import annotations

import importlib
import json
import time
import traceback
from datetime import datetime

from app.browsergym_integration.config import BrowserGymRunConfig
from app.browsergym_integration.miniwob_grounding import real_candidate_bid
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.miniwob_dom_bridge import extract_miniwob_dom_candidates, merge_dom_candidates_with_ax
from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord


class BrowserGymRunner:
    NON_RECOVERABLE_POLICY_ERRORS = {
        "menu_requires_hover_no_supported_action",
        "autocomplete_suggestions_not_found",
        "datepicker_header_not_found",
        "search_no_progress",
    }
    NON_RECOVERABLE_MAPPING_STRATEGIES = {
        "policy_click_menu_hover_required",
        "policy_use_autocomplete_not_found",
        "policy_choose_date_header_not_found",
        "policy_choose_date_invalid_state",
        "policy_book_flight_search_no_progress",
        "policy_click_link_target_not_found",
        "policy_choose_date_fill_no_progress",
    }
    NON_RECOVERABLE_DIAGNOSTICS = {
        "menu_requires_hover_no_supported_action",
        "autocomplete_suggestions_not_found",
        "datepicker_header_not_found",
        "search_no_progress",
    }
    def __init__(self, agent_factory, config: BrowserGymRunConfig):
        self.agent_factory = agent_factory
        self.config = config

    @staticmethod
    def _is_rawish_answer(value: str | None) -> bool:
        if not value:
            return True
        low = value.lower()
        return any(token in low for token in ["screenshot", "array(", "open_pages_urls", "chat_messages", "ndarray", "{'url':"])

    @staticmethod
    def _is_miniwob_config(config: BrowserGymRunConfig) -> bool:
        return (config.benchmark or "").lower() == "miniwob" or config.env_id.lower().startswith("browsergym/miniwob.")

    @staticmethod
    def _browsergym_default_action_syntax() -> list[str]:
        return ['click("bid", "left")', 'click("bid")', 'mouse_click(x, y, "left")', 'mouse_move(x, y)', 'fill("bid", "text")', 'select_option("bid", ["option_text"])', 'select_option("bid", "option_text")', 'clear("bid")', 'focus("bid")', 'press("bid", "Enter")', 'keyboard_type("text")', 'keyboard_insert_text("text")', 'noop()']

    @staticmethod
    def _looks_like_action_space_repr(value: str) -> bool:
        compact = " ".join(str(value or "").strip().split())
        return compact.startswith(("Unicode(", "Text(", "String(")) or compact in {"Unicode()", "Text()", "String()"}

    @classmethod
    def _extract_action_syntax(cls, env) -> list[str]:
        examples: list[str] = []
        unwrapped = getattr(env, "unwrapped", None)
        candidates = [
            getattr(unwrapped, "action_mapping", None),
            getattr(unwrapped, "action_set", None),
            getattr(unwrapped, "high_level_action_set", None),
            getattr(unwrapped, "actions", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            for attr in ("get_action_description", "describe", "action_description", "to_python_code", "docs"):
                member = getattr(candidate, attr, None)
                if callable(member):
                    try:
                        desc = member()
                    except TypeError:
                        continue
                    if desc:
                        examples.extend(str(desc).splitlines())
                elif member:
                    examples.extend(str(member).splitlines())
            if isinstance(candidate, dict):
                examples.extend(str(key) for key in candidate.keys())
            elif isinstance(candidate, (list, tuple, set)):
                examples.extend(str(item) for item in candidate)
        cleaned: list[str] = []
        for item in examples:
            value = " ".join(str(item).strip().split())
            if value and not cls._looks_like_action_space_repr(value) and value not in cleaned:
                cleaned.append(value)
        return (cleaned or cls._browsergym_default_action_syntax())[:25]

    @staticmethod
    def _find_page(env):
        for obj in (getattr(env, "unwrapped", None), env):
            page = getattr(obj, "page", None) if obj is not None else None
            if page is not None and hasattr(page, "evaluate"):
                return page
        return None

    @staticmethod
    def _browsergym_scaled_bbox(bbox: dict, scale_factor: float) -> dict:
        scaled = {}
        for key in ("x", "y", "width", "height", "left", "top", "right", "bottom"):
            value = bbox.get(key)
            if isinstance(value, (int, float)):
                scaled[key] = value * scale_factor
            else:
                try:
                    scaled[key] = float(value) * scale_factor
                except (TypeError, ValueError):
                    pass
        return scaled

    @classmethod
    def _augment_page_candidate_coordinates(cls, candidates: list[dict], scale_factor: float) -> list[dict]:
        augmented: list[dict] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            item = dict(candidate)
            center_x = item.get("center_x")
            center_y = item.get("center_y")
            try:
                page_center_x = float(center_x)
                page_center_y = float(center_y)
            except (TypeError, ValueError):
                page_center_x = None
                page_center_y = None
            if page_center_x is not None and page_center_y is not None:
                item["page_center_x"] = page_center_x
                item["page_center_y"] = page_center_y
                item["browsergym_center_x"] = page_center_x * scale_factor
                item["browsergym_center_y"] = page_center_y * scale_factor
            item["browsergym_scale_factor"] = scale_factor
            item["coordinate_space"] = "page_css"
            item["action_coordinate_space"] = "browsergym_scaled"
            bbox = item.get("bbox")
            if isinstance(bbox, dict):
                item["browsergym_bbox"] = cls._browsergym_scaled_bbox(bbox, scale_factor)
            augmented.append(item)
        return augmented

    @classmethod
    def _extract_page_clickable_candidates(cls, env) -> tuple[list[dict], bool]:
        page = cls._find_page(env)
        if page is None:
            return [], True
        try:
            dom = extract_miniwob_dom_candidates(page)
            return dom, False
        except Exception:
            return [], True

    @classmethod
    def _augment_miniwob_observation_with_page_candidates(cls, env, obs, info):
        if not isinstance(obs, dict):
            return obs
        current = browsergym_obs_to_page_context(obs, info if isinstance(info, dict) else {})
        candidates, failed = cls._extract_page_clickable_candidates(env)
        augmented = dict(obs)
        existing_page = list(obs.get("page_clickable_candidates") or []) if isinstance(obs, dict) else []
        if (not candidates) and existing_page:
            candidates = existing_page
            failed = False
        ax = list(obs.get("clickable_candidates") or []) if isinstance(obs, dict) else []
        merged = merge_dom_candidates_with_ax(ax, candidates or [])
        seen = set()
        def key(c):
            if not isinstance(c, dict):
                return ("",)
            bid = str(real_candidate_bid(c) or "").strip()
            if bid:
                return ("bid", bid.lower())
            href = str(c.get("href") or "").strip().lower()
            text = str(c.get("text") or c.get("innerText") or c.get("textContent") or c.get("name") or "").strip().lower()
            tag = str(c.get("tag") or c.get("role") or "").strip().lower()
            if href:
                return ("href_text_tag", href, text, tag)
            bbox = c.get("bbox") or c.get("browsergym_bbox") or {}
            bbox_key = tuple((bbox or {}).get(k) for k in ("x", "y", "width", "height", "left", "top", "right", "bottom")) if isinstance(bbox, dict) else ("",)
            return ("bbox_text_tag", bbox_key, text, tag)
        for c in merged:
            seen.add(key(c))
        for c in candidates or []:
            k = key(c)
            if k in seen:
                continue
            seen.add(k)
            merged.append(c)
        augmented["page_clickable_candidates"] = (candidates or [])[:300]
        augmented["page_clickable_candidates_count"] = len(candidates or [])
        augmented["page_candidate_extraction_failed"] = bool(failed)
        augmented["clickable_candidates"] = merged[:300]
        augmented["clickable_candidates_count"] = len(merged)
        if failed:
            augmented["page_candidate_extraction_error"] = "dom_candidate_extraction_failed"
        return augmented

    def _persist_report(self, report: BrowserGymRunReport) -> BrowserGymRunReport:
        if not self.config.save_artifacts:
            return report
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        env_id_s = self.config.env_id.replace("/", "_").replace(".", "_")
        out = self.config.output_dir / f"browsergym_run_{env_id_s}_{ts}.json"
        out.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        report.output_path = str(out)
        return report


    @staticmethod
    def _candidate_verbose_summary(candidate: dict | None) -> dict | None:
        if not isinstance(candidate, dict):
            return None
        return {
            "bid": candidate.get("bid"),
            "bid_source": candidate.get("bid_source"),
            "text": candidate.get("text") or candidate.get("name") or candidate.get("value") or candidate.get("label") or candidate.get("ariaLabel"),
            "bbox": candidate.get("bbox"),
            "browsergym_bbox": candidate.get("browsergym_bbox"),
            "centers": {
                "center_x": candidate.get("center_x"),
                "center_y": candidate.get("center_y"),
                "page_center_x": candidate.get("page_center_x"),
                "page_center_y": candidate.get("page_center_y"),
                "browsergym_center_x": candidate.get("browsergym_center_x"),
                "browsergym_center_y": candidate.get("browsergym_center_y"),
            },
        }

    @staticmethod
    def _make_step_record(idx: int, obs, info, action: str, reward, terminated: bool, truncated: bool, decision) -> BrowserGymStepRecord:
        selected_candidate = getattr(decision, "selected_candidate", None)
        selected_bid_source = selected_candidate.get("bid_source") if isinstance(selected_candidate, dict) else None
        return BrowserGymStepRecord(
            step_idx=idx,
            url=str((obs or {}).get("url", "")) if isinstance(obs, dict) else "",
            action=action,
            reward=float(reward) if reward is not None else None,
            terminated=terminated,
            truncated=truncated,
            info_summary={"keys": sorted(list((info or {}).keys()))} if isinstance(info, dict) else {},
            internal_plan=decision.internal_plan,
            selected_step=decision.selected_step,
            extracted_value=getattr(decision, "extracted_value", None),
            rationale=getattr(decision, "rationale", None),
            action_rationale=getattr(decision, "rationale", None),
            action_string=getattr(decision, "action_string", None) or action,
            miniwob_instruction=getattr(decision, "miniwob_instruction", None),
            mapping_error=getattr(decision, "mapping_error", None),
            action_string_before_mapping=getattr(decision, "action_string_before_mapping", None),
            action_string_after_mapping=getattr(decision, "action_string_after_mapping", None),
            selected_candidate=selected_candidate,
            selected_candidate_bid=getattr(decision, "selected_candidate_bid", None) or real_candidate_bid(selected_candidate),
            bid_source=getattr(decision, "bid_source", None) or selected_bid_source,
            selected_candidate_verbose=BrowserGymRunner._candidate_verbose_summary(selected_candidate),
            clickable_candidates_count=getattr(decision, "clickable_candidates_count", None),
            page_candidate_extraction_failed=getattr(decision, "page_candidate_extraction_failed", None),
            mapping_strategy=getattr(decision, "mapping_strategy", None),
            mapping_diagnostics=getattr(decision, "mapping_diagnostics", None),
            fallback_used=bool(getattr(decision, "fallback_used", False)),
            fallback_type=getattr(decision, "fallback_type", None),
            fallback_reward=getattr(decision, "fallback_reward", None),
            fallback_terminated=getattr(decision, "fallback_terminated", None),
            error=getattr(decision, "mapping_error", None),
            vision_used=bool(getattr(decision, "vision_used", False)),
            vision_image_present=bool(getattr(decision, "vision_image_present", False)),
        )


    def _try_miniwob_playwright_fallback(self, env, decision, obs, reward, terminated: bool, truncated: bool, info):
        if not self.config.allow_playwright_fallback or not self._is_miniwob_config(self.config):
            return obs, reward, terminated, truncated, info
        if terminated or truncated or float(reward or 0) > 0:
            return obs, reward, terminated, truncated, info
        if getattr(decision, "mapping_strategy", None) != "coordinate_scaled":
            return obs, reward, terminated, truncated, info
        candidate = getattr(decision, "selected_candidate", None)
        if not isinstance(candidate, dict):
            return obs, reward, terminated, truncated, info
        try:
            page_x = float(candidate.get("page_center_x", candidate.get("center_x")))
            page_y = float(candidate.get("page_center_y", candidate.get("center_y")))
        except (TypeError, ValueError):
            return obs, reward, terminated, truncated, info
        page = self._find_page(env)
        mouse = getattr(page, "mouse", None) if page is not None else None
        click = getattr(mouse, "click", None)
        if not callable(click):
            return obs, reward, terminated, truncated, info
        try:
            click(page_x, page_y)
            fallback_obs, fallback_reward, fallback_terminated, fallback_truncated, fallback_info = env.step("noop()")
        except Exception as exc:
            try:
                decision.mapping_error = f"{getattr(decision, 'mapping_error', '') or ''} playwright_fallback_failed: {exc}".strip()
            except Exception:
                pass
            return obs, reward, terminated, truncated, info
        try:
            decision.fallback_used = True
            decision.fallback_type = "playwright_direct_click"
            decision.fallback_reward = float(fallback_reward) if fallback_reward is not None else None
            decision.fallback_terminated = bool(fallback_terminated)
        except Exception:
            pass
        return fallback_obs, fallback_reward, bool(fallback_terminated), bool(fallback_truncated), fallback_info

    def run_one(self) -> BrowserGymRunReport:
        started = time.time()
        try:
            import gymnasium as gym
            import browsergym.core  # noqa: F401
            if "miniwob" in self.config.env_id.lower():
                importlib.import_module("browsergym.miniwob")
        except Exception as exc:
            failure_stage = "miniwob_import" if "miniwob" in self.config.env_id.lower() else "imports"
            message = f"browsergym.miniwob import failed: {exc}" if failure_stage == "miniwob_import" else f"Install browsergym dependencies first: {exc}"
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage=failure_stage, error_message=message, runtime_sec=time.time() - started, benchmark=self.config.benchmark, task_name=self.config.task_name))

        try:
            env = gym.make(self.config.env_id, task_kwargs=self.config.task_kwargs or {})
        except TypeError:
            env = gym.make(self.config.env_id)
        except Exception as exc:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage="env_creation", error_message=str(exc), error_traceback=traceback.format_exc(), runtime_sec=time.time() - started, benchmark=self.config.benchmark, task_name=self.config.task_name))
        agent = self.agent_factory()
        if hasattr(agent, "set_browsergym_context"):
            agent.set_browsergym_context(env_id=self.config.env_id, benchmark=self.config.benchmark)
        if hasattr(agent, "set_browsergym_action_syntax"):
            agent.set_browsergym_action_syntax(self._extract_action_syntax(env))
        steps = []
        reward = None
        terminated = False
        truncated = False
        final_answer = None
        status = "partial"
        try:
            obs, info = env.reset()
            history = []
            if self._is_miniwob_config(self.config):
                print(f"[MiniWoB] task {self.config.env_id} env reset", flush=True)
            for idx in range(self.config.max_steps):
                if self._is_miniwob_config(self.config):
                    obs = self._augment_miniwob_observation_with_page_candidates(env, obs, info)
                    miniwob_ctx = browsergym_obs_to_page_context(obs, info if isinstance(info, dict) else {})
                    print(f"[MiniWoB] step {idx + 1} clickable_candidates_count={miniwob_ctx.get('clickable_candidates_count', 0)} candidates={miniwob_ctx.get('clickable_candidates', [])[:5]}", flush=True)
                decision = agent.act(self.config.goal or "", obs, info, history)
                action = decision.action
                if self._is_miniwob_config(self.config) and (getattr(decision, "finish", False) or str(action).strip().lower().startswith(("finish(", "agent_finish("))):
                    action = "noop()"
                    try:
                        decision.action = action
                        decision.action_string = action
                        decision.finish = False
                        decision.mapping_error = getattr(decision, "mapping_error", None) or "action_mapping_failure: finish is disabled for MiniWoB; success requires reward > 0"
                        decision.rationale = getattr(decision, "rationale", None) or decision.mapping_error
                    except Exception:
                        pass
                if self._is_miniwob_config(self.config):
                    before = getattr(decision, "action_string_before_mapping", None)
                    after = getattr(decision, "action_string_after_mapping", None) or action
                    print(f"[MiniWoB] step {idx + 1}/{self.config.max_steps} action={action} before_grounding={before} after_grounding={after} mapping_strategy={getattr(decision, 'mapping_strategy', None)}", flush=True)
                if decision.finish and self.config.stop_on_agent_finish and not self._is_miniwob_config(self.config):
                    final_answer = (decision.answer or "").strip() or None
                    finish_status = "success_by_agent_finish"
                    if self._is_rawish_answer(final_answer):
                        finish_status = "invalid_agent_finish"
                    steps.append(self._make_step_record(idx, obs, info, action, reward, False, False, decision))
                    status = finish_status
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                if self._is_miniwob_config(self.config):
                    obs, reward, terminated, truncated, info = self._try_miniwob_playwright_fallback(env, decision, obs, reward, terminated, truncated, info)
                    if getattr(decision, "mapping_strategy", None) == "select_option_control":
                        try:
                            after_ctx = browsergym_obs_to_page_context(obs if isinstance(obs, dict) else {}, info if isinstance(info, dict) else {})
                            controls = after_ctx.get("select_control_candidates") or []
                            selected = getattr(decision, "selected_candidate", None)
                            selected_bid = real_candidate_bid(selected) if isinstance(selected, dict) else ""
                            after_control = next((c for c in controls if str(c.get("bid") or "") == selected_bid), controls[0] if controls else None)
                            after_value = None
                            if isinstance(after_control, dict):
                                after_value = after_control.get("value") or after_control.get("selected_value") or after_control.get("current_value") or after_control.get("name") or after_control.get("text")
                            diagnostics = getattr(decision, "mapping_diagnostics", None)
                            if isinstance(diagnostics, dict):
                                diagnostics["current_select_value_after"] = after_value
                        except Exception:
                            pass
                    selected = getattr(decision, "selected_candidate", None)
                    selected_verbose = self._candidate_verbose_summary(selected)
                    selected_bid = selected.get("bid") if isinstance(selected, dict) else None
                    selected_bid_source = selected.get("bid_source") if isinstance(selected, dict) else None
                    print(f"[MiniWoB] step {idx + 1} selected_candidate.bid={selected_bid} bid_source={selected_bid_source} selected_candidate={selected} selected_candidate_verbose={selected_verbose} mapping_strategy={getattr(decision, 'mapping_strategy', None)}", flush=True)
                    print(f"[MiniWoB] step {idx + 1} reward={reward} terminated={terminated} truncated={truncated}", flush=True)
                history_note = None
                if self._is_miniwob_config(self.config) and getattr(decision, "mapping_error", None):
                    history_note = f"previous target_text {getattr(decision, 'action_string_before_mapping', '')} had no candidate, choose coordinate fallback or inspect candidates."
                sc = getattr(decision, "selected_candidate", None) if isinstance(getattr(decision, "selected_candidate", None), dict) else {}
                history.append({"action": action, "reward": reward, "error": getattr(decision, "mapping_error", None), "rationale": getattr(decision, "rationale", None), "url": (obs or {}).get("url", "") if isinstance(obs, dict) else "", "instruction": getattr(decision, "miniwob_instruction", None), "note": history_note, "selected_candidate_bid": real_candidate_bid(sc), "selected_candidate_text": str(sc.get("text") or sc.get("name") or "").strip().lower(), "selected_candidate_role": sc.get("role"), "mapping_strategy": getattr(decision, "mapping_strategy", None), "terminated": terminated, "truncated": truncated})
                steps.append(self._make_step_record(idx, obs, info, action, reward, terminated, truncated, decision))
                mapping_error = str(getattr(decision, "mapping_error", "") or "").strip()
                mapping_strategy = str(getattr(decision, "mapping_strategy", "") or "").strip()
                diagnostics = getattr(decision, "mapping_diagnostics", None) if isinstance(getattr(decision, "mapping_diagnostics", None), dict) else {}
                diag_nonrecoverable = any(bool(diagnostics.get(k)) for k in self.NON_RECOVERABLE_DIAGNOSTICS)
                if self._is_miniwob_config(self.config) and (mapping_error in self.NON_RECOVERABLE_POLICY_ERRORS or mapping_strategy in self.NON_RECOVERABLE_MAPPING_STRATEGIES or diag_nonrecoverable):
                    status = "failed"
                    terminated = True
                    break
                if terminated or truncated:
                    break
        except Exception as exc:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="failed", steps=steps, failure_stage="runtime", error_message=str(exc), error_traceback=traceback.format_exc(), runtime_sec=time.time() - started, final_answer=final_answer, steps_count=len(steps), success=False, benchmark=self.config.benchmark, task_name=self.config.task_name))
        finally:
            env.close()

        if status not in {"success_by_agent_finish", "invalid_agent_finish", "failed"}:
            reward_value_for_status = float(reward) if reward is not None else None
            if self._is_miniwob_config(self.config):
                status = "success" if reward_value_for_status is not None and reward_value_for_status > 0 else "partial"
            else:
                status = "success" if terminated else "partial"
        if status == "invalid_agent_finish":
            status = "failed"
        reward_value = float(reward) if reward is not None else None
        success_value = (reward_value is not None and reward_value > 0) if self._is_miniwob_config(self.config) else (status in {"success", "success_by_agent_finish"})
        if self._is_miniwob_config(self.config):
            print(f"[MiniWoB] task done success={success_value} reward={reward_value}", flush=True)
        failure_stage = None
        error_message = None
        if status == "failed" and steps:
            last_error = steps[-1].mapping_error
            if last_error in self.NON_RECOVERABLE_POLICY_ERRORS:
                failure_stage = "unsupported_action" if last_error == "menu_requires_hover_no_supported_action" else "action_mapping_failure"
                error_message = f"non-recoverable policy error: {last_error}"
        report = BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status=status, reward=reward_value, terminated=terminated, truncated=truncated, steps=steps, runtime_sec=time.time() - started, final_answer=final_answer, steps_count=len(steps), success=success_value, benchmark=self.config.benchmark, task_name=self.config.task_name, failure_stage=failure_stage, error_message=error_message)
        return self._persist_report(report)
