import sys
import types

from app.browsergym_integration.agent_adapter import BrowserGymAgentAdapter
from app.browsergym_integration.config import BrowserGymRunConfig
from app.browsergym_integration.runner import BrowserGymRunner
from app.utils.llm_client import LLMClientError
from scripts.run_minwob_subset import result_from_report


class _DirectLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_planner_json(self, system_prompt, user_prompt, **kwargs):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "kwargs": kwargs})
        return self.payload


class _FailingDirectLLM(_DirectLLM):
    def generate_planner_json(self, system_prompt, user_prompt, **kwargs):
        raise LLMClientError("LLM returned reasoning/thinking but no JSON content")


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


def test_miniwob_direct_non_json_llm_response_returns_noop():
    planner = _Planner({"rationale": "unused", "action": "noop()"})
    planner.llm_client = _FailingDirectLLM({})
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.click-button")
    decision = adapter.act("goal", {"goal": "Click the button", "url": "http://miniwob/", "text": "OK"}, {}, [])
    assert decision.action == "noop()"
    assert decision.mapping_error == "action_mapping_failure: llm_non_json_response"
    assert decision.mapping_strategy == "llm_non_json_response"

from app.browsergym_integration.miniwob_grounding import ground_miniwob_action


def test_miniwob_grounding_click_bare_text_to_bid():
    result = ground_miniwob_action(
        action="click(submit)",
        parsed_response={"target_text": "submit"},
        candidates=[{"bid": "7", "role": "button", "name": "submit"}],
    )
    assert result.action == 'click("7", "left")'
    assert result.selected_candidate["bid"] == "7"


def test_miniwob_grounding_click_quoted_text_to_bid():
    result = ground_miniwob_action(
        action='click("submit")',
        parsed_response={},
        candidates=[{"bid": "7", "role": "button", "name": "submit"}],
    )
    assert result.action == 'click("7", "left")'


def test_miniwob_grounding_exact_match_precedes_fuzzy():
    result = ground_miniwob_action(
        action="click(submit)",
        parsed_response={"target_text": "submit"},
        candidates=[
            {"bid": "8", "role": "button", "name": "submit form"},
            {"bid": "7", "role": "button", "name": "submit"},
        ],
    )
    assert result.action == 'click("7", "left")'


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


def test_choose_date_policy_runs_pre_llm_and_skips_llm_call():
    planner = _Planner({"rationale": "fallback", "action": 'fill("17", "06/28/2016")'})
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.choose-date")
    obs = {
        "goal": "Select 06/28/2016 as the date and hit submit.",
        "url": "http://miniwob/",
        "page_clickable_candidates": [
            {"bid": "17", "tag": "input", "type": "text", "id": "datepicker", "className": "hasDatepicker", "role": "", "source": "dom"}
        ],
    }

    decision = adapter.act("goal", obs, {}, [])

    assert decision.action == 'click("17", "left")'
    assert decision.mapping_strategy == "policy_choose_date_open"
    assert (decision.mapping_diagnostics or {}).get("policy_pre_llm_used") is True
    assert (decision.mapping_diagnostics or {}).get("policy_name") == "choose-date"
    assert (decision.mapping_diagnostics or {}).get("chosen_stage") == "open"
    assert (decision.mapping_diagnostics or {}).get("date_input_bid") == "17"
    assert planner.llm_client.calls == []


def test_grounding_keeps_click_on_datepicker_input():
    result = ground_miniwob_action(
        action='click("17", "left")',
        parsed_response={"instruction": "Select 06/28/2016 as the date and hit submit.", "env_id": "browsergym/miniwob.choose-date"},
        candidates=[{"bid": "17", "tag": "input", "type": "text", "id": "datepicker", "className": "hasDatepicker", "parent_text": "Date field"}],
    )
    assert result.action == 'click("17", "left")'
    assert result.mapping_strategy == "datepicker_input_click"


def test_action_space_unicode_repr_not_in_action_syntax_examples():
    class _UnicodeSpace:
        def __repr__(self):
            return "Unicode()"

        __str__ = __repr__

    env = types.SimpleNamespace(action_space=_UnicodeSpace(), unwrapped=types.SimpleNamespace(action_space=_UnicodeSpace()))
    examples = BrowserGymRunner._extract_action_syntax(env)
    assert "Unicode()" not in examples
    assert 'click("bid", "left")' in examples


def test_dom_candidate_with_bbox_maps_to_mouse_click():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"role": "button", "text": "Submit", "visible": True, "enabled": True, "center_x": 11, "center_y": 22}],
    )
    assert result.action == 'mouse_click(11, 22, "left")'
    assert result.mapping_strategy == "coordinate_raw"


def test_candidate_with_bid_maps_to_click_bid():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"bid": "a12", "role": "button", "name": "Submit", "center_x": 11, "center_y": 22}],
    )
    assert result.action == 'click("a12", "left")'
    assert result.mapping_strategy == "bid_click"


def test_click_submit_goes_through_grounding_before_env_step(monkeypatch):
    class _LLMAgentFactory:
        def __call__(self):
            planner = _Planner({"rationale": "click submit", "target_text": "Submit", "action": 'click("Submit")'})
            return BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.click-button", benchmark="miniwob")

    class _SubmitEnv(_MiniWoBEnv):
        action_space = "Unicode()"

        def reset(self):
            return {"goal": "Click Submit", "axtree_object": {"role": "button", "name": "Submit", "bid": "real_bid"}}, {}

    env = _SubmitEnv()
    _patch_miniwob_env(monkeypatch, env)
    report = BrowserGymRunner(
        agent_factory=_LLMAgentFactory(),
        config=BrowserGymRunConfig(env_id="browsergym/miniwob.click-button", goal="g", benchmark="miniwob", max_steps=1),
    ).run_one()
    assert env.actions == ['click("real_bid", "left")']
    assert report.steps[0].action_string_before_mapping == 'click("Submit")'
    assert report.steps[0].action_string_after_mapping == 'click("real_bid", "left")'


def test_non_miniwob_action_syntax_defaults_do_not_change_mode():
    adapter = BrowserGymAgentAdapter(_PlanPlanner(), None, _Validator(), env_id="browsergym/openended")
    assert adapter.uses_direct_action_mode is False


def test_page_candidate_augmentation_adds_browsergym_scaled_coordinates():
    candidates = BrowserGymRunner._augment_page_candidate_coordinates(
        [{"center_x": 20, "center_y": 100, "bbox": {"x": 2, "y": 10, "width": 30, "height": 40, "left": 2, "top": 10, "right": 32, "bottom": 50}}],
        2.0,
    )

    candidate = candidates[0]
    assert candidate["page_center_x"] == 20.0
    assert candidate["page_center_y"] == 100.0
    assert candidate["browsergym_scale_factor"] == 2.0
    assert candidate["browsergym_center_x"] == 40.0
    assert candidate["browsergym_center_y"] == 200.0
    assert candidate["coordinate_space"] == "page_css"
    assert candidate["action_coordinate_space"] == "browsergym_scaled"
    assert candidate["browsergym_bbox"] == {"x": 4.0, "y": 20.0, "width": 60.0, "height": 80.0, "left": 4.0, "top": 20.0, "right": 64.0, "bottom": 100.0}


def test_candidate_center_prefers_browsergym_center_over_raw_center():
    from app.browsergym_integration.miniwob_grounding import candidate_center

    candidate = {"center_x": 20, "center_y": 100, "browsergym_center_x": 40, "browsergym_center_y": 200}

    assert candidate_center(candidate) == (40.0, 200.0)


def test_candidate_center_prefers_browsergym_bbox_over_raw_bbox():
    from app.browsergym_integration.miniwob_grounding import candidate_center

    candidate = {
        "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
        "browsergym_bbox": {"x": 20, "y": 40, "width": 100, "height": 80},
    }

    assert candidate_center(candidate) == (70.0, 80.0)


def test_candidate_center_falls_back_to_legacy_center_coordinates():
    from app.browsergym_integration.miniwob_grounding import candidate_center

    assert candidate_center({"center_x": 20, "center_y": 100}) == (20.0, 100.0)


def test_candidate_center_prefers_action_over_browsergym_over_raw_center():
    from app.browsergym_integration.miniwob_grounding import candidate_center, candidate_center_with_strategy

    candidate = {
        "center_x": 20,
        "center_y": 100,
        "browsergym_center_x": 40,
        "browsergym_center_y": 200,
        "action_x": 50,
        "action_y": 250,
    }

    assert candidate_center(candidate) == (50.0, 250.0)
    assert candidate_center_with_strategy(candidate) == (50.0, 250.0, "coordinate_scaled")


def test_grounding_scaled_mapping_strategy_and_selected_candidate_in_report(monkeypatch):
    class _LLMAgentFactory:
        def __call__(self):
            planner = _Planner({"rationale": "click ok", "target_text": "OK", "action": 'click("OK")'})
            return BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.click-button", benchmark="miniwob")

    class _ScaledEnv(_MiniWoBEnv):
        def reset(self):
            return {
                "url": "http://127.0.0.1:8765/miniwob/",
                "goal": "Click OK",
                "page_clickable_candidates": [
                    {"role": "button", "text": "OK", "center_x": 10, "center_y": 20, "browsergym_center_x": 15, "browsergym_center_y": 30}
                ],
            }, {}

    env = _ScaledEnv()
    _patch_miniwob_env(monkeypatch, env)
    report = BrowserGymRunner(
        agent_factory=_LLMAgentFactory(),
        config=BrowserGymRunConfig(env_id="browsergym/miniwob.click-button", goal="g", benchmark="miniwob", max_steps=1),
    ).run_one()

    assert env.actions == ['mouse_click(15, 30, "left")']
    assert report.steps[0].selected_candidate["text"] == "OK"
    assert report.steps[0].mapping_strategy == "coordinate_scaled"
    assert report.steps[0].action_string_before_mapping == 'click("OK")'
    assert report.steps[0].action_string_after_mapping == 'mouse_click(15, 30, "left")'
    result = result_from_report(report, env_id="browsergym/miniwob.click-button", use_vision=False)
    assert result["steps"][0]["selected_candidate"]["text"] == "OK"
    assert result["steps"][0]["mapping_strategy"] == "coordinate_scaled"

def test_grounding_remaps_mouse_click_when_target_text_selects_scaled_candidate():
    result = ground_miniwob_action(
        action='mouse_click(25.56, 147.5, "left")',
        parsed_response={"target_text": "Okay"},
        candidates=[{"role": "button", "text": "Okay", "center_x": 25.56, "center_y": 147.5, "browsergym_center_x": 38.34, "browsergym_center_y": 221.25}],
    )
    assert result.action == 'mouse_click(38.34, 221.25, "left")'
    assert result.mapping_strategy == "coordinate_scaled"


def test_page_candidate_extractor_preserves_plain_bid_and_source(monkeypatch):
    class _Page:
        _bgym_scale_factor = 1.0

        def evaluate(self, script):
            assert "el.getAttribute('bid')" in script
            assert "el.getAttribute('data-testid')" in script
            assert "el.getAttribute('browsergym_id')" in script
            assert "el.getAttribute('data-bid')" in script
            assert "el.getAttribute('ref')" in script
            return [
                {
                    "tag": "button",
                    "text": "Submit",
                    "id": "plain-dom-id",
                    "bid": "12",
                    "bid_source": "bid",
                    "bbox": {"x": 1, "y": 2, "width": 10, "height": 20},
                    "center_x": 6,
                    "center_y": 12,
                }
            ]

    monkeypatch.setattr(BrowserGymRunner, "_find_page", classmethod(lambda cls, env: _Page()))
    candidates, failed = BrowserGymRunner._extract_page_clickable_candidates(object())

    assert failed is False
    assert candidates[0]["bid"] == "12"
    assert candidates[0]["bid_source"] == "bid"
    assert candidates[0]["id"] == "plain-dom-id"


def test_candidate_index_and_plain_dom_id_are_not_used_as_bid():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"index": 12, "id": "plain-dom-id", "role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )

    assert result.action == 'mouse_click(11, 22, "left")'
    assert result.mapping_strategy == "coordinate_raw"


def test_grounding_prefers_real_bid_over_coordinates():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"bid": "12", "bid_source": "bid", "role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )

    assert result.action == 'click("12", "left")'
    assert result.mapping_strategy == "bid_click"


def test_candidate_with_bid_18_maps_to_browsergym_bid_click():
    result = ground_miniwob_action(
        action='click("cancel")',
        parsed_response={"target_text": "cancel"},
        candidates=[{"text": "cancel", "bid": "18", "bid_source": "bid", "role": "button"}],
    )

    assert result.action == 'click("18", "left")'
    assert result.selected_candidate["bid"] == "18"
    assert result.selected_candidate["bid_source"] == "bid"
    assert result.mapping_strategy == "bid_click"


def test_grounding_rewrites_case_text_numeric_and_mouse_click_to_selected_bid():
    candidates = [{"text": "cancel", "bid": "18", "bid_source": "bid", "role": "button", "browsergym_center_x": 1, "browsergym_center_y": 2}]

    for action, parsed in (
        ('click("cancel")', {}),
        ('click("Cancel")', {}),
        ('click("2")', {"target_text": "cancel"}),
        ('mouse_click(1, 2, "left")', {"target_text": "cancel"}),
        ('noop()', {"target_text": "cancel"}),
    ):
        result = ground_miniwob_action(action=action, parsed_response=parsed, candidates=candidates)
        assert result.action == 'click("18", "left")'
        assert result.mapping_strategy == "bid_click"


def test_fake_bid_source_index_is_forbidden_and_falls_back_to_coordinates():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit", "target_bid": "2"},
        candidates=[{"bid": "2", "bid_source": "index", "role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )

    assert result.action == 'mouse_click(11, 22, "left")'
    assert result.mapping_strategy == "coordinate_raw"


def test_dom_id_is_not_browsergym_bid_even_when_matching_target_bid():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit", "target_bid": "plain-dom-id"},
        candidates=[{"id": "plain-dom-id", "role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )

    assert result.action == 'mouse_click(11, 22, "left")'
    assert result.mapping_strategy == "coordinate_raw"


def test_coordinate_fallback_only_when_no_real_bid():
    with_bid = ground_miniwob_action(
        action='mouse_click(11, 22, "left")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"bid": "18", "bid_source": "bid", "role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )
    without_bid = ground_miniwob_action(
        action='mouse_click(11, 22, "left")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"role": "button", "text": "Submit", "center_x": 11, "center_y": 22}],
    )

    assert with_bid.action == 'click("18", "left")'
    assert with_bid.mapping_strategy == "bid_click"
    assert without_bid.action == 'mouse_click(11, 22, "left")'
    assert without_bid.mapping_strategy == "coordinate_raw"


def test_fill_action_with_bid_remains_fill_not_click():
    result = ground_miniwob_action(
        action='fill("16", "michel")',
        parsed_response={},
        candidates=[{"bid": "16", "bid_source": "bid", "role": "textbox"}],
    )

    assert result.action == 'fill("16", "michel")'
    assert result.mapping_strategy == "bid_fill"


def test_textbox_fill_intent_maps_to_bid_fill():
    result = ground_miniwob_action(
        action='type("16", "michel")',
        parsed_response={"target_bid": "16", "target_text": "michel", "intent": "enter text"},
        candidates=[{"bid": "16", "bid_source": "bid", "role": "textbox"}],
    )

    assert result.action == 'fill("16", "michel")'
    assert result.mapping_strategy == "bid_fill"


def test_button_click_intent_still_maps_to_bid_click():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"target_text": "Submit"},
        candidates=[{"bid": "7", "bid_source": "bid", "role": "button", "name": "Submit"}],
    )

    assert result.action == 'click("7", "left")'
    assert result.mapping_strategy == "bid_click"


def test_username_password_instruction_parser_extracts_quoted_values():
    from app.browsergym_integration.miniwob_grounding import parse_username_password_instruction

    values = parse_username_password_instruction('Enter the username "michel" and the password "c3" into the text fields and press login.')

    assert values == {"username": "michel", "password": "c3"}


def test_two_textboxes_are_mapped_in_order_for_login_style_forms():
    from app.browsergym_integration.miniwob_grounding import map_login_textboxes

    first = {"bid": "16", "bid_source": "bid", "role": "textbox"}
    second = {"bid": "17", "bid_source": "bid", "role": "textbox"}
    mapped = map_login_textboxes('Enter the username "u" and the password "p".', [first, second])

    assert mapped["username"] is first
    assert mapped["password"] is second


def test_submit_login_button_is_selected_by_role_and_name():
    from app.browsergym_integration.miniwob_grounding import find_submit_button

    login = {"bid": "20", "bid_source": "bid", "role": "button", "name": "Login"}

    assert find_submit_button([{"role": "link", "name": "Login"}, login]) is login


def test_repeated_textbox_click_no_progress_is_detected():
    result = ground_miniwob_action(
        action='click("16", "left")',
        parsed_response={},
        candidates=[{"bid": "16", "bid_source": "bid", "role": "textbox"}],
        history=[{"action": 'click("16", "left")', "reward": 0.0}, {"action": 'click("16", "left")', "reward": 0.0}],
    )

    assert result.action == "noop()"
    assert "no_progress repeated textbox click" in result.mapping_error


def test_select_instruction_extracts_target_option():
    from app.browsergym_integration.miniwob_grounding import extract_select_target_from_instruction

    assert extract_select_target_from_instruction('Choose "Orlando" from the list.') == "Orlando"
    assert extract_select_target_from_instruction("Select Boston from the dropdown.") == "Boston"
    assert extract_select_target_from_instruction("Select Azerbaijan from the list and click Submit.") == "Azerbaijan"


def test_option_candidate_matching_text_maps_to_select_option_control_strategy():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"miniwob_instruction": "Choose orange from the list."},
        candidates=[
            {"bid": "combo", "bid_source": "bid", "role": "combobox", "text": "choices"},
            {"bid": "opt2", "bid_source": "bid", "role": "option", "text": "orange", "parent_bid": "combo"},
            {"bid": "submit", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("combo", "orange")'
    assert result.mapping_strategy == "select_option_control"
    assert result.selected_candidate["bid"] == "combo"



def test_explicit_option_bid_click_is_allowed_when_option_lookup_would_fail():
    result = ground_miniwob_action(
        action='click("19", "left")',
        parsed_response={
            "miniwob_instruction": "Select Azerbaijan from the list and click Submit.",
            "rationale": 'The option "Azerbaijan" is visible and has bid "19".',
        },
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "text": "countries"},
            {"bid": "19", "bid_source": "bid", "role": "option", "text": "", "parent_bid": "13"},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("13", "Azerbaijan")'
    assert result.mapping_strategy == "select_option_control"
    assert result.selected_candidate["bid"] == "13"
    assert result.mapping_error is None
    assert result.mapping_diagnostics["target_option"] == "Azerbaijan"
    assert result.mapping_diagnostics["clicked_bid"] == "19"


def test_explicit_option_bid_click_with_matching_text_uses_select_option_strategy():
    result = ground_miniwob_action(
        action='click("19")',
        parsed_response={"miniwob_instruction": "Select Azerbaijan from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "text": "countries"},
            {"bid": "19", "bid_source": "bid", "role": "option", "text": "Azerbaijan", "parent_bid": "13"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("13", "Azerbaijan")'
    assert result.mapping_strategy == "select_option_control"
    assert result.selected_candidate["bid"] == "13"


def test_unknown_explicit_select_bid_is_blocked():
    result = ground_miniwob_action(
        action='click("999", "left")',
        parsed_response={"miniwob_instruction": "Select Azerbaijan from the list and click Submit."},
        candidates=[{"bid": "19", "bid_source": "bid", "role": "option", "text": "Azerbaijan"}],
    )

    assert result.action == "noop()"
    assert "clicked bid '999' not found" in result.mapping_error
    assert result.mapping_strategy == "none"


def test_submit_before_select_option_is_blocked_even_with_explicit_bid():
    result = ground_miniwob_action(
        action='click("20", "left")',
        parsed_response={"miniwob_instruction": "Select Azerbaijan from the list and click Submit."},
        candidates=[
            {"bid": "19", "bid_source": "bid", "role": "option", "text": "Azerbaijan", "selected": False},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
    )

    assert result.action == "noop()"
    assert "submit_before_select" in result.mapping_error


def test_repeated_combobox_open_click_no_progress_is_detected():
    result = ground_miniwob_action(
        action='click("13", "left")',
        parsed_response={"miniwob_instruction": "Select Azerbaijan from the list and click Submit."},
        candidates=[{"bid": "13", "bid_source": "bid", "role": "combobox", "text": "countries"}],
        history=[
            {"action": 'click("13", "left")', "reward": 0.0},
            {"action": 'click("13", "left")', "reward": 0.0},
        ],
    )

    assert result.action == "noop()"
    assert "no_progress_repeated_select" in result.mapping_error

def test_select_option_syntax_is_used_when_supported():
    result = ground_miniwob_action(
        action='click("blue")',
        parsed_response={"miniwob_instruction": "Select blue from the dropdown."},
        candidates=[
            {"bid": "combo", "bid_source": "bid", "role": "combobox", "text": "choices"},
            {"bid": "opt-blue", "bid_source": "bid", "role": "option", "text": "blue", "parent_bid": "combo"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("combo", "blue")'
    assert result.mapping_strategy == "select_option_control"
    assert result.selected_candidate["bid"] == "combo"


def test_unknown_select_option_does_not_map_to_submit():
    result = ground_miniwob_action(
        action='click("Submit")',
        parsed_response={"miniwob_instruction": "Choose purple from the list."},
        candidates=[
            {"bid": "combo", "bid_source": "bid", "role": "combobox", "text": "choices"},
            {"bid": "opt2", "bid_source": "bid", "role": "option", "text": "orange", "parent_bid": "combo"},
            {"bid": "submit", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("combo", "purple")'
    assert result.mapping_strategy == "select_option_control"


def test_repeated_combobox_option_loop_is_detected():
    result = ground_miniwob_action(
        action='click("green")',
        parsed_response={"miniwob_instruction": "Choose green from the list."},
        candidates=[
            {"bid": "combo", "bid_source": "bid", "role": "combobox", "text": "choices"},
            {"bid": "opt-green", "bid_source": "bid", "role": "option", "text": "green", "parent_bid": "combo"},
        ],
        history=[
            {"action": 'select_option("combo", "green")', "reward": 0.0},
            {"action": 'select_option("combo", "green")', "reward": 0.0},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == "noop()"
    assert "select_option_no_progress" in result.mapping_error


def test_choose_list_instruction_extracts_denmark_target_option():
    from app.browsergym_integration.miniwob_grounding import extract_select_target_from_instruction

    assert extract_select_target_from_instruction("Select Denmark from the list and click Submit.") == "Denmark"


def test_choose_list_control_bid_maps_to_select_option_list_syntax():
    result = ground_miniwob_action(
        action='click("16", "left")',
        parsed_response={"miniwob_instruction": "Select Denmark from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": "Latvia", "text": "countries"},
            {"bid": "16", "bid_source": "bid", "role": "option", "text": "Denmark", "parent_bid": "13"},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", ["option_text"])'],
    )

    assert result.action == 'select_option("13", ["Denmark"])'
    assert result.mapping_strategy == "select_option_control"
    assert result.selected_candidate["bid"] == "13"
    assert result.mapping_diagnostics["target_option"] == "Denmark"
    assert result.mapping_diagnostics["select_control_bid"] == "13"
    assert result.mapping_diagnostics["current_select_value_before"] == "Latvia"


def test_choose_list_combobox_click_is_overridden_to_select_option_control():
    result = ground_miniwob_action(
        action='click("13", "left")',
        parsed_response={"miniwob_instruction": "Select Denmark from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": {"value": "combobox"}, "value": "Latvia"},
            {"bid": "16", "bid_source": "bid", "role": "option", "text": "Denmark", "parent_bid": "13"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'select_option("13", "Denmark")'
    assert result.mapping_strategy == "select_option_control"


def test_click_link_instruction_matches_link_text():
    result = ground_miniwob_action(
        action='click("news")',
        parsed_response={"miniwob_instruction": 'Click the link "News".'},
        candidates=[{"bid": "42", "bid_source": "bid", "role": "link", "text": "News", "href": "/news"}],
    )
    assert result.action == 'click("42", "left")'
    assert result.mapping_strategy == "link_bid_click"


def test_click_link_instruction_can_match_href_title_or_aria_label():
    result = ground_miniwob_action(
        action='click("profile")',
        parsed_response={"miniwob_instruction": "Open link profile page."},
        candidates=[{"bid": "7", "bid_source": "bid", "role": "link", "text": "", "href": "/profile", "title": "Profile page", "aria_label": "profile"}],
    )
    assert result.action == 'click("7", "left")'
    assert result.mapping_strategy == "link_bid_click"


def test_click_link_ambiguous_targets_are_blocked():
    result = ground_miniwob_action(
        action='click("news")',
        parsed_response={"miniwob_instruction": 'Click the link "News".'},
        candidates=[
            {"bid": "1", "bid_source": "bid", "role": "link", "text": "News"},
            {"bid": "2", "bid_source": "bid", "role": "link", "text": "News"},
        ],
    )
    assert result.action == "noop()"
    assert "ambiguous_link_target" in (result.mapping_error or "")


def test_click_link_unknown_target_is_blocked():
    result = ground_miniwob_action(
        action='click("unknown")',
        parsed_response={"miniwob_instruction": 'Follow link "Unknown".'},
        candidates=[{"bid": "1", "bid_source": "bid", "role": "link", "text": "Home"}],
    )
    assert result.action == "noop()"
    assert "link_target_not_found" in (result.mapping_error or "")


def test_choose_list_submit_allowed_after_control_value_equals_target():
    result = ground_miniwob_action(
        action='click("20", "left")',
        parsed_response={"miniwob_instruction": "Select Denmark from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": "Denmark"},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == 'click("20", "left")'
    assert result.mapping_strategy == "select_submit_after_match"


def test_choose_list_submit_after_failed_select_option_reports_no_state_change():
    result = ground_miniwob_action(
        action='click("20", "left")',
        parsed_response={"miniwob_instruction": "Select Denmark from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": "Latvia"},
            {"bid": "16", "bid_source": "bid", "role": "option", "text": "Denmark", "parent_bid": "13"},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        history=[{"action": 'select_option("13", "Denmark")', "reward": 0.0}],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == "noop()"
    assert "select_option_no_state_change" in result.mapping_error
    assert result.mapping_strategy == "select_option_control"


def test_choose_list_select_option_no_progress_stops_after_two_attempts():
    result = ground_miniwob_action(
        action='click("16", "left")',
        parsed_response={"miniwob_instruction": "Select Denmark from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": "Latvia"},
            {"bid": "16", "bid_source": "bid", "role": "option", "text": "Denmark", "parent_bid": "13"},
        ],
        history=[
            {"action": 'select_option("13", "Denmark")', "reward": 0.0},
            {"action": 'select_option("13", "Denmark")', "reward": 0.0},
        ],
        action_syntax=['select_option("bid", "option_text")'],
    )

    assert result.action == "noop()"
    assert "select_option_no_progress" in result.mapping_error


def test_normalize_candidate_value_dict_and_stringified_dict():
    from app.browsergym_integration.miniwob_grounding import normalize_candidate_text, normalize_candidate_value

    assert normalize_candidate_value({"type": "string", "value": "Alfreda"}) == "Alfreda"
    assert normalize_candidate_value("{'type': 'string', 'value': 'Alfreda'}") == "Alfreda"
    assert normalize_candidate_text({"type": "computedString", "value": "Submit"}) == "submit"


def test_is_submit_like_candidate_accepts_button_with_submit_name():
    from app.browsergym_integration.miniwob_grounding import is_submit_like_candidate

    assert is_submit_like_candidate({"role": {"value": "button"}, "name": {"value": "Submit"}}) is True


def test_choose_list_selected_value_match_with_dict_value_allows_submit():
    result = ground_miniwob_action(
        action='click("20", "left")',
        parsed_response={"miniwob_instruction": "Select Alfreda from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": {"type": "string", "value": "Alfreda"}},
            {"bid": "20", "bid_source": "bid", "role": "button", "name": {"type": "computedString", "value": "Submit"}},
        ],
        action_syntax=['select_option("bid", ["option_text"])'],
    )

    assert result.action == 'click("20", "left")'
    assert result.mapping_strategy == "select_submit_after_match"
    assert result.mapping_diagnostics["selected_value_matches_target"] is True
    assert result.mapping_diagnostics["submit_allowed"] is True
    assert result.mapping_diagnostics["submit_source"] == "clicked_bid_candidate"


def test_choose_list_submit_blocked_when_selected_value_not_matching_target():
    result = ground_miniwob_action(
        action='click("20", "left")',
        parsed_response={"miniwob_instruction": "Select Alfreda from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": {"type": "string", "value": "Gavra"}},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        action_syntax=['select_option("bid", ["option_text"])'],
    )

    assert result.action == "noop()"
    assert "submit_before_select" in (result.mapping_error or "")


def test_select_option_no_progress_not_triggered_when_target_already_selected():
    result = ground_miniwob_action(
        action='click("16", "left")',
        parsed_response={"miniwob_instruction": "Select Alfreda from the list and click Submit."},
        candidates=[
            {"bid": "13", "bid_source": "bid", "role": "combobox", "value": {"type": "string", "value": "Alfreda"}},
            {"bid": "16", "bid_source": "bid", "role": "option", "text": "Alfreda", "parent_bid": "13"},
            {"bid": "20", "bid_source": "bid", "role": "button", "text": "Submit"},
        ],
        history=[
            {"action": 'select_option("13", ["Alfreda"])', "reward": 0.0},
            {"action": 'select_option("13", ["Alfreda"])', "reward": 0.0},
        ],
        action_syntax=['select_option("bid", ["option_text"])'],
    )

    assert "select_option_no_progress" not in (result.mapping_error or "")
    assert result.action == 'click("20", "left")'

def test_checkbox_select_instruction_does_not_trigger_native_select_guard():
    result = ground_miniwob_action(
        action='click("21", "left")',
        parsed_response={"miniwob_instruction": "Select HWy32jZ and click Submit."},
        candidates=[{"bid": "21", "bid_source": "bid", "role": "checkbox", "name": "HWy32jZ"}],
    )
    assert result.action == 'click("21", "left")'
    assert result.mapping_strategy == "checkbox_bid_click"


def test_radio_select_instruction_does_not_trigger_native_select_guard():
    result = ground_miniwob_action(
        action='click("24", "left")',
        parsed_response={"miniwob_instruction": "Select Ix2km and click Submit."},
        candidates=[{"bid": "24", "bid_source": "bid", "role": "radio", "name": "Ix2km"}],
    )
    assert result.action == 'click("24", "left")'
    assert result.mapping_strategy == "radio_bid_click"


def test_menu_path_instruction_does_not_trigger_native_select_guard():
    result = ground_miniwob_action(
        action='click("19", "left")',
        parsed_response={"miniwob_instruction": "Select Pris>Cherlyn>Libbi"},
        candidates=[{"bid": "19", "bid_source": "bid", "role": "menuitem", "name": "Pris"}],
    )
    assert result.action == 'click("19", "left")'
    assert result.mapping_strategy == "menuitem_bid_click"


def test_date_fill_does_not_trigger_native_select_guard():
    result = ground_miniwob_action(
        action='fill("17", "12/28/2016")',
        parsed_response={"miniwob_instruction": "Select 12/28/2016 as the date and hit submit."},
        candidates=[{"bid": "17", "bid_source": "bid", "role": "textbox"}],
    )
    assert result.action == 'fill("17", "12/28/2016")'
    assert result.mapping_strategy == "date_or_text_bid_fill"


def test_autocomplete_click_without_select_control_does_not_trigger_native_select_guard():
    result = ground_miniwob_action(
        action='click("19", "left")',
        parsed_response={"miniwob_instruction": 'Enter an item that starts with "Po" and ends with "land".'},
        candidates=[{"bid": "19", "bid_source": "bid", "role": "list", "name": "Poland"}],
    )
    assert result.action == 'click("19", "left")'
    assert result.mapping_strategy == "autocomplete_suggestion_click"
