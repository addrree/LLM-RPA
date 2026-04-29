import types

from app.browsergym_integration.config import BrowserGymRunConfig
from app.browsergym_integration.runner import BrowserGymRunner


class _Agent:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(action="finish(answer='done')", finish=True, answer="done", internal_plan=None, selected_step=None)


class _Env:
    def __init__(self):
        self.step_calls = 0

    def reset(self):
        return {"url": "https://example.com"}, {"k": 1}

    def step(self, action):
        self.step_calls += 1
        return {}, 0.0, False, False, {}

    def close(self):
        return None


def test_finish_decision_does_not_call_env_step(monkeypatch, tmp_path):
    env = _Env()

    gym_mod = types.SimpleNamespace(make=lambda env_id, task_kwargs=None: env)
    monkeypatch.setitem(__import__("sys").modules, "gymnasium", gym_mod)
    monkeypatch.setitem(__import__("sys").modules, "browsergym", types.SimpleNamespace(core=types.SimpleNamespace()))
    monkeypatch.setitem(__import__("sys").modules, "browsergym.core", types.SimpleNamespace())

    runner = BrowserGymRunner(agent_factory=lambda: _Agent(), config=BrowserGymRunConfig(env_id="browsergym/openended", goal="g", save_artifacts=True, output_dir=tmp_path))
    report = runner.run_one()

    assert report.status == "success_by_agent_finish"
    assert report.final_answer == "done"
    assert env.step_calls == 0
    assert report.output_path is not None


def test_skipped_report_is_persisted(tmp_path):
    cfg = BrowserGymRunConfig(env_id="browsergym/webarena.10", goal="g", save_artifacts=True, output_dir=tmp_path)
    report = BrowserGymRunner(agent_factory=lambda: _Agent(), config=cfg).run_one()
    assert report.status == "skipped"
    assert report.output_path is not None
