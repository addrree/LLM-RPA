import types

from app.browsergym_integration.config import BrowserGymRunConfig
from app.browsergym_integration.observation_adapter import browsergym_obs_to_page_context
from app.browsergym_integration.runner import BrowserGymRunner


class _FakeArray:
    def __init__(self, shape=(10, 10, 3), dtype="uint8"):
        self.shape = shape
        self.dtype = dtype

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous.")


class _Agent:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(action="finish(answer='done')", finish=True, answer="done", internal_plan=None, selected_step=None, extracted_value="done")


class _RawObsFinishAgent:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(action="finish(answer='{" + "screenshot: array(" + "}')", finish=True, answer="{'screenshot': 'array('}", internal_plan=None, selected_step=None)


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


class _FailingEnv(_Env):
    def step(self, action):
        raise RuntimeError("boom")


class _AgentUsingAdapter:
    def act(self, goal, obs, info, history):
        ctx = browsergym_obs_to_page_context(obs, info)
        assert ctx["screenshot_summary"] is not None
        return types.SimpleNamespace(action="noop()", finish=False, answer=None, internal_plan=None, selected_step=None)


class _AgentNoFinish:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(action="noop()", finish=False, answer=None, internal_plan=None, selected_step=None)


class _ArrayEnv(_Env):
    def reset(self):
        return {"url": "https://example.com", "screenshot": _FakeArray((10, 10, 3), "uint8")}, {"k": 1}

    def step(self, action):
        self.step_calls += 1
        return {"url": "https://example.com", "screenshot": _FakeArray((10, 10, 3), "uint8")}, 0.0, True, False, {}


def _patch_env(monkeypatch, env):
    gym_mod = types.SimpleNamespace(make=lambda env_id, task_kwargs=None: env)
    monkeypatch.setitem(__import__("sys").modules, "gymnasium", gym_mod)
    monkeypatch.setitem(__import__("sys").modules, "browsergym", types.SimpleNamespace(core=types.SimpleNamespace()))
    monkeypatch.setitem(__import__("sys").modules, "browsergym.core", types.SimpleNamespace())


def test_finish_decision_does_not_call_env_step(monkeypatch, tmp_path):
    env = _Env()
    _patch_env(monkeypatch, env)
    runner = BrowserGymRunner(agent_factory=lambda: _Agent(), config=BrowserGymRunConfig(env_id="browsergym/openended", goal="g", save_artifacts=True, output_dir=tmp_path))
    report = runner.run_one()
    assert report.status == "success_by_agent_finish"
    assert report.final_answer == "done"
    assert env.step_calls == 0
    assert report.output_path is not None


def test_raw_observation_guardrail(monkeypatch):
    env = _Env()
    _patch_env(monkeypatch, env)
    runner = BrowserGymRunner(agent_factory=lambda: _RawObsFinishAgent(), config=BrowserGymRunConfig(env_id="browsergym/openended", goal="g"))
    report = runner.run_one()
    assert report.status == "failed"


def test_runner_saves_runtime_traceback(monkeypatch, tmp_path):
    env = _FailingEnv()
    _patch_env(monkeypatch, env)
    runner = BrowserGymRunner(agent_factory=lambda: _AgentNoFinish(), config=BrowserGymRunConfig(env_id="browsergym/openended", goal="g", save_artifacts=True, output_dir=tmp_path))
    report = runner.run_one()
    assert report.status == "failed"
    assert "RuntimeError: boom" in (report.error_traceback or "")


def test_smoke_like_env_with_ndarray_observation(monkeypatch):
    env = _ArrayEnv()
    _patch_env(monkeypatch, env)
    runner = BrowserGymRunner(agent_factory=lambda: _AgentUsingAdapter(), config=BrowserGymRunConfig(env_id="browsergym/openended", goal="g", max_steps=2))
    report = runner.run_one()
    assert report.status == "success"
