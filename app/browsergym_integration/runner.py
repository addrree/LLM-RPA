from __future__ import annotations

import importlib
import json
import time
import traceback
from datetime import datetime

from app.browsergym_integration.config import BrowserGymRunConfig, validate_webarena_env_vars
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord


class BrowserGymRunner:
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
        return ['click("bid")', 'mouse_click(x, y, "left")', 'fill("bid", "text")', 'keyboard_press("Enter")', 'noop()']

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
    def _extract_page_clickable_candidates(env) -> tuple[list[dict], bool]:
        page = BrowserGymRunner._find_page(env)
        if page is None:
            return [], True
        script = """
        () => {
          const selectors = [
            'button','input[type=button]','input[type=submit]','input[type=checkbox]','input[type=radio]',
            'a','select','textarea','[role=button]','[role=link]','[onclick]','label'
          ];
          const seen = new Set();
          return Array.from(document.querySelectorAll(selectors.join(','))).map((el) => {
            if (seen.has(el)) return null;
            seen.add(el);
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            const visible = !!(rect.width || rect.height) && style.visibility !== 'hidden' && style.display !== 'none';
            const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || el.name || el.id || '').trim();
            return {
              tag: el.tagName.toLowerCase(),
              type: el.getAttribute('type') || '',
              role: el.getAttribute('role') || '',
              text,
              value: el.value || '',
              ariaLabel: el.getAttribute('aria-label') || '',
              id: el.id || '',
              name: el.getAttribute('name') || '',
              disabled: !!el.disabled,
              enabled: !el.disabled,
              visible,
              clickable: true,
              bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height, left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom},
              center_x: rect.x + rect.width / 2,
              center_y: rect.y + rect.height / 2
            };
          }).filter(Boolean);
        }
        """
        try:
            candidates = page.evaluate(script)
        except Exception:
            return [], True
        return (candidates if isinstance(candidates, list) else []), False

    @classmethod
    def _augment_miniwob_observation_with_page_candidates(cls, env, obs, info):
        if not isinstance(obs, dict):
            return obs
        current = browsergym_obs_to_page_context(obs, info if isinstance(info, dict) else {})
        if current.get("clickable_candidates_count", 0):
            return obs
        candidates, failed = cls._extract_page_clickable_candidates(env)
        augmented = dict(obs)
        if candidates:
            augmented["page_clickable_candidates"] = candidates[:30]
        if failed or not candidates:
            augmented["page_candidate_extraction_failed"] = True
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
    def _make_step_record(idx: int, obs, info, action: str, reward, terminated: bool, truncated: bool, decision) -> BrowserGymStepRecord:
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
            selected_candidate=getattr(decision, "selected_candidate", None),
            clickable_candidates_count=getattr(decision, "clickable_candidates_count", None),
            page_candidate_extraction_failed=getattr(decision, "page_candidate_extraction_failed", None),
            mapping_strategy=getattr(decision, "mapping_strategy", None),
            error=getattr(decision, "mapping_error", None),
            vision_used=bool(getattr(decision, "vision_used", False)),
            vision_image_present=bool(getattr(decision, "vision_image_present", False)),
        )

    def run_one(self) -> BrowserGymRunReport:
        started = time.time()
        check = validate_webarena_env_vars(self.config.env_id)
        if not check["ok"]:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage="env_validation", error_message=check["message"], runtime_sec=time.time() - started, benchmark=self.config.benchmark, task_name=self.config.task_name))
        try:
            import gymnasium as gym
            import browsergym.core  # noqa: F401
            normalized_env_id = self.config.env_id.lower()
            if "webarena" in normalized_env_id:
                importlib.import_module("browsergym.webarena")
            if "miniwob" in normalized_env_id:
                importlib.import_module("browsergym.miniwob")
        except Exception as exc:
            normalized_env_id = self.config.env_id.lower()
            failure_stage = "webarena_import" if "webarena" in normalized_env_id else ("miniwob_import" if "miniwob" in normalized_env_id else "imports")
            message = f"browsergym.webarena import failed: {exc}" if failure_stage == "webarena_import" else (f"browsergym.miniwob import failed: {exc}" if failure_stage == "miniwob_import" else f"Install browsergym dependencies first: {exc}")
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
                    print(f"[MiniWoB] step {idx + 1}/{self.config.max_steps} action={action} before_grounding={before} after_grounding={after}", flush=True)
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
                    print(f"[MiniWoB] step {idx + 1} reward={reward} terminated={terminated} truncated={truncated}", flush=True)
                history_note = None
                if self._is_miniwob_config(self.config) and getattr(decision, "mapping_error", None):
                    history_note = f"previous target_text {getattr(decision, 'action_string_before_mapping', '')} had no candidate, choose coordinate fallback or inspect candidates."
                history.append({"action": action, "reward": reward, "error": getattr(decision, "mapping_error", None), "rationale": getattr(decision, "rationale", None), "url": (obs or {}).get("url", "") if isinstance(obs, dict) else "", "instruction": getattr(decision, "miniwob_instruction", None), "note": history_note})
                steps.append(self._make_step_record(idx, obs, info, action, reward, terminated, truncated, decision))
                if terminated or truncated:
                    break
        except Exception as exc:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="failed", steps=steps, failure_stage="runtime", error_message=str(exc), error_traceback=traceback.format_exc(), runtime_sec=time.time() - started, final_answer=final_answer, steps_count=len(steps), success=False, benchmark=self.config.benchmark, task_name=self.config.task_name))
        finally:
            env.close()

        if status not in {"success_by_agent_finish", "invalid_agent_finish"}:
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
        report = BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status=status, reward=reward_value, terminated=terminated, truncated=truncated, steps=steps, runtime_sec=time.time() - started, final_answer=final_answer, steps_count=len(steps), success=success_value, benchmark=self.config.benchmark, task_name=self.config.task_name)
        return self._persist_report(report)
