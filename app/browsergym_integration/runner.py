from __future__ import annotations

import json
import time
from datetime import datetime

from app.browsergym_integration.config import BrowserGymRunConfig, validate_webarena_env_vars
from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord


class BrowserGymRunner:
    def __init__(self, agent_factory, config: BrowserGymRunConfig):
        self.agent_factory = agent_factory
        self.config = config

    def run_one(self) -> BrowserGymRunReport:
        started = time.time()
        check = validate_webarena_env_vars(self.config.env_id)
        if not check["ok"]:
            return BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage="env_validation", error_message=check["message"])
        try:
            import gymnasium as gym
            import browsergym.core  # noqa: F401
        except Exception as exc:
            return BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="skipped", failure_stage="imports", error_message=f"Install browsergym dependencies first: {exc}")

        env = gym.make(self.config.env_id, task_kwargs=self.config.task_kwargs or {})
        agent = self.agent_factory()
        steps = []
        reward = None
        terminated = False
        truncated = False
        try:
            obs, info = env.reset()
            history = []
            for idx in range(self.config.max_steps):
                decision = agent.act(self.config.goal or "", obs, info, history)
                action = decision.action
                obs, reward, terminated, truncated, info = env.step(action)
                history.append({"action": action, "reward": reward})
                steps.append(BrowserGymStepRecord(step_idx=idx, url=str((obs or {}).get("url", "")) if isinstance(obs, dict) else "", action=action, reward=float(reward) if reward is not None else None, terminated=terminated, truncated=truncated, info_summary={"keys": sorted(list((info or {}).keys())) if isinstance(info, dict) else []}, internal_plan=decision.internal_plan, selected_step=decision.selected_step))
                if terminated or truncated or (decision.finish and self.config.stop_on_agent_finish):
                    break
        except Exception as exc:
            return BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status="failed", steps=steps, failure_stage="runtime", error_message=str(exc), runtime_sec=time.time() - started)
        finally:
            env.close()

        status = "success" if terminated else "partial"
        report = BrowserGymRunReport(env_id=self.config.env_id, goal=self.config.goal or "", status=status, reward=float(reward) if reward is not None else None, terminated=terminated, truncated=truncated, steps=steps, runtime_sec=time.time() - started)
        if self.config.save_artifacts:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            env_id_s = self.config.env_id.replace("/", "_").replace(".", "_")
            out = self.config.output_dir / f"browsergym_run_{env_id_s}_{ts}.json"
            out.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
            report.output_path = str(out)
        return report
