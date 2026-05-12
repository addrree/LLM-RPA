import sys
import types

from app.browsergym_integration.agent_adapter import BrowserGymAgentAdapter
from app.browsergym_integration.config import BrowserGymRunConfig
from app.browsergym_integration.runner import BrowserGymRunner
from scripts.run_minwob_subset import result_from_report


class _DirectLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_planner_json(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "kwargs": kwargs})
        return self.payload


class _Planner:
    def __init__(self, payload):
        self.llm_client = _DirectLLM(payload)
        self.build_plan_calls = 0

    def build_plan(self, *args, **kwargs):
        self.build_plan_calls += 1
        raise AssertionError("MiniWoB direct action mode must not build TaskSpec plans")


class _PlanPlanner:
    def __init__(self):
        self.calls = 0

    def build_plan(self, *args, **kwargs):
        self.calls += 1
        from app.schemas.task_spec import TaskSpec

        return TaskSpec.model_validate(
            {
                "goal": "g",
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "constraints": {"max_steps": 1, "max_replans": 1, "max_verification_retries": 1, "timeout_sec": 1},
                "expected_result": {"description": "d", "required_fields": []},
                "steps": [{"step_id": 1, "action": "wait_for", "args": {"text": "hello"}}],
            }
        )


class _Validator:
    def validate(self, plan):
        return None


def test_miniwob_env_id_uses_direct_action_mode_not_taskspec():
    planner = _Planner({"rationale": "click target", "action": "click('button-1')"})
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.click-button")

    decision = adapter.act("goal", {"goal": "Click the button", "url": "http://miniwob/", "text": "OK"}, {}, [])

    assert adapter.uses_direct_action_mode is True
    assert planner.build_plan_calls == 0
    assert planner.llm_client.calls
    assert decision.action == "click('button-1')"
    assert decision.miniwob_instruction == "Click the button"
    assert decision.selected_step is None


def test_non_miniwob_env_id_keeps_taskspec_mode():
    planner = _PlanPlanner()
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/openended")

    decision = adapter.act("goal", {"url": "https://example.com", "text": "hello"}, {}, [])

    assert adapter.uses_direct_action_mode is False
    assert planner.calls == 1
    assert decision.action == "noop()"


class _FinishAgent:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(
            action="finish(answer='done')",
            finish=True,
            answer="done",
            internal_plan=None,
            selected_step=None,
            rationale="done",
            action_string="finish(answer='done')",
        )


class _InvalidDirectAgent:
    def act(self, goal, obs, info, history):
        return types.SimpleNamespace(
            action="noop()",
            finish=False,
            answer=None,
            internal_plan=None,
            selected_step=None,
            rationale="bad action",
            action_string="noop()",
            mapping_error="action_mapping_failure: unsupported MiniWoB action syntax",
            miniwob_instruction="Click the button",
        )


class _MiniWoBEnv:
    action_space = ["click('bid')", "noop()"]

    def __init__(self):
        self.actions = []

    def reset(self):
        return {"url": "http://127.0.0.1:8765/miniwob/", "goal": "Click the button", "text": "OK"}, {"task_info": {"goal": "Click the button"}}

    def step(self, action):
        self.actions.append(action)
        return {"url": "http://127.0.0.1:8765/miniwob/", "goal": "Click the button", "text": "OK"}, 0.0, False, False, {}

    def close(self):
        return None


def _patch_miniwob_env(monkeypatch, env):
    gym_mod = types.SimpleNamespace(make=lambda env_id, task_kwargs=None: env)
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "browsergym", types.SimpleNamespace(core=types.SimpleNamespace(), miniwob=types.SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "browsergym.core", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "browsergym.miniwob", types.SimpleNamespace())


def test_miniwob_finish_action_does_not_count_success_without_reward(monkeypatch):
    env = _MiniWoBEnv()
    _patch_miniwob_env(monkeypatch, env)

    report = BrowserGymRunner(
        agent_factory=lambda: _FinishAgent(),
        config=BrowserGymRunConfig(env_id="browsergym/miniwob.click-button", goal="g", benchmark="miniwob", max_steps=1),
    ).run_one()

    assert env.actions == ["noop()"]
    assert report.success is False
    assert report.status == "partial"
    assert report.reward == 0.0
    assert report.steps[0].mapping_error == "action_mapping_failure: finish is disabled for MiniWoB; success requires reward > 0"


def test_invalid_model_action_noops_and_batch_report_contains_step(monkeypatch):
    env = _MiniWoBEnv()
    _patch_miniwob_env(monkeypatch, env)

    report = BrowserGymRunner(
        agent_factory=lambda: _InvalidDirectAgent(),
        config=BrowserGymRunConfig(env_id="browsergym/miniwob.click-button", goal="g", benchmark="miniwob", max_steps=1),
    ).run_one()
    result = result_from_report(report, env_id="browsergym/miniwob.click-button", use_vision=False)

    assert report.status == "partial"
    assert env.actions == ["noop()"]
    assert result["failure_stage"] == "action_mapping_failure"
    assert result["steps"][0]["action_string"] == "noop()"
    assert result["steps"][0]["action_rationale"] == "bad action"
    assert result["steps"][0]["reward"] == 0.0
    assert result["steps"][0]["mapping_error"] == "action_mapping_failure: unsupported MiniWoB action syntax"

from app.browsergym_integration.miniwob_grounding import ground_miniwob_action


def test_miniwob_grounding_click_bare_text_to_bid():
    result = ground_miniwob_action(
        action="click(submit)",
        parsed_response={"target_text": "submit"},
        candidates=[{"bid": "7", "role": "button", "name": "submit"}],
    )
    assert result.action == 'click("7")'
    assert result.selected_candidate["bid"] == "7"


def test_miniwob_grounding_click_quoted_text_to_bid():
    result = ground_miniwob_action(
        action='click("submit")',
        parsed_response={},
        candidates=[{"bid": "7", "role": "button", "name": "submit"}],
    )
    assert result.action == 'click("7")'


def test_miniwob_grounding_exact_match_precedes_fuzzy():
    result = ground_miniwob_action(
        action="click(submit)",
        parsed_response={"target_text": "submit"},
        candidates=[
            {"bid": "8", "role": "button", "name": "submit form"},
            {"bid": "7", "role": "button", "name": "submit"},
        ],
    )
    assert result.action == 'click("7")'


def test_miniwob_grounding_no_candidate_noops_with_mapping_error():
    result = ground_miniwob_action(action='click("submit")', parsed_response={}, candidates=[])
    assert result.action == "noop()"
    assert "no clickable candidate matched" in result.mapping_error


def test_miniwob_grounding_blocks_repeated_ineffective_action():
    result = ground_miniwob_action(
        action='click("submit")',
        parsed_response={},
        candidates=[{"bid": "7", "role": "button", "name": "submit"}],
        history=[{"action": 'click("submit")', "reward": 0.0}, {"action": 'click("submit")', "reward": 0.0}],
    )
    assert result.action == "noop()"
    assert "repeated ineffective action" in result.mapping_error
