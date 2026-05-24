from app.browsergym_integration.agent_adapter import BrowserGymAgentAdapter
from scripts.run_minwob_subset import parse_args


class _Planner:
    def __init__(self):
        self.llm_client = self
        self.calls = 0

    def generate_planner_json(self, *args, **kwargs):
        self.calls += 1
        return {"action": "noop()"}


class _Validator:
    def validate(self, plan):
        return None


def test_extraction_subset_default_timeout_and_no_find_midpoint():
    args = parse_args(["--subset", "extraction"])
    assert args.subset == "extraction"


def test_compact_candidates_for_llm_limits_and_truncates():
    candidates = [{"bid": str(i), "text": ("x" * 2000) if i == 0 else f"btn {i}", "role": "button", "className": "email-thread" if i % 2 else "wrap"} for i in range(30)]
    compact = BrowserGymAgentAdapter.compact_candidates_for_llm(candidates, limit=20)
    assert len(compact) <= 20
    assert all(len(str(c.get("text") or "")) <= 120 for c in compact)


def test_known_extraction_no_decision_skips_llm_call():
    planner = _Planner()
    adapter = BrowserGymAgentAdapter(planner, None, _Validator(), env_id="browsergym/miniwob.daily-calendar")
    adapter.allow_extraction_llm_fallback = False
    obs = {"goal": "Find calendar event", "url": "http://miniwob/", "page_clickable_candidates": []}
    decision = adapter.act("goal", obs, {}, [])
    assert decision.mapping_strategy == "extraction_no_decision"
    assert planner.calls == 0
