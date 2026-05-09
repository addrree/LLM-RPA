from __future__ import annotations

import importlib
import json
import time
import traceback
from datetime import datetime

from app.browsergym_integration.config import BrowserGymRunConfig, validate_webarena_env_vars
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
            vision_used=bool(getattr(decision, "vision_used", False)),
            vision_image_present=bool(getattr(decision, "vision_image_present", False)),
        )

    def run_one(self) -> BrowserGymRunReport:
        started = time.time()
        check = validate_webarena_env_vars(self.config.env_id)
        if not check["ok"]:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage="env_validation", error_message=check["message"], runtime_sec=time.time() - started))
        try:
            import gymnasium as gym
            import browsergym.core  # noqa: F401
            if "webarena" in self.config.env_id.lower():
                importlib.import_module("browsergym.webarena")
        except Exception as exc:
            failure_stage = "webarena_import" if "webarena" in self.config.env_id.lower() else "imports"
            message = f"browsergym.webarena import failed: {exc}" if failure_stage == "webarena_import" else f"Install browsergym dependencies first: {exc}"
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage=failure_stage, error_message=message, runtime_sec=time.time() - started))

        env = gym.make(self.config.env_id, task_kwargs=self.config.task_kwargs or {})
        agent = self.agent_factory()
        steps = []
        reward = None
        terminated = False
        truncated = False
        final_answer = None
        status = "partial"
        try:
            obs, info = env.reset()
            history = []
            for idx in range(self.config.max_steps):
                decision = agent.act(self.config.goal or "", obs, info, history)
                action = decision.action
                if decision.finish and self.config.stop_on_agent_finish:
                    final_answer = (decision.answer or "").strip() or None
                    finish_status = "success_by_agent_finish"
                    if self._is_rawish_answer(final_answer):
                        finish_status = "invalid_agent_finish"
                    steps.append(self._make_step_record(idx, obs, info, action, reward, False, False, decision))
                    status = finish_status
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                history.append({"action": action, "reward": reward})
                steps.append(self._make_step_record(idx, obs, info, action, reward, terminated, truncated, decision))
                if terminated or truncated:
                    break
        except Exception as exc:
            return self._persist_report(BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="failed", steps=steps, failure_stage="runtime", error_message=str(exc), error_traceback=traceback.format_exc(), runtime_sec=time.time() - started, final_answer=final_answer))
        finally:
            env.close()

        if status not in {"success_by_agent_finish", "invalid_agent_finish"}:
            status = "success" if terminated else "partial"
        if status == "invalid_agent_finish":
            status = "failed"
        report = BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status=status, reward=float(reward) if reward is not None else None, terminated=terminated, truncated=truncated, steps=steps, runtime_sec=time.time() - started, final_answer=final_answer)
        return self._persist_report(report)
