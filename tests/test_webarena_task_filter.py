from app.browsergym_integration.webarena_tasks import classify_requires_llm_judge


def test_marks_llm_judge_tokens():
    cfg = {"evaluator_types": ["fuzzy_match", "rule_based"], "judge": "openai"}
    assert classify_requires_llm_judge(cfg) is True


def test_non_llm_evaluator_not_marked():
    cfg = {"evaluator_types": ["exact_match"], "intent": "click submit"}
    assert classify_requires_llm_judge(cfg) is False
