from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec
from app.validator.plan_validator import PlanValidator
from app.verifier.llm_verifier import LLMVerifier


class _FailIfCalledClient:
    def generate_verifier_artifact(self, *args, **kwargs):
        raise AssertionError("Verifier LLM should not be called for deterministic shape checks.")


def _anchor_object_plan() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "Extract a label and nearby count.",
            "start_url": "https://example.org",
            "allowed_domains": ["example.org"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {
                "description": "Language object",
                "required_fields": ["language_name", "article_count"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://example.org"}},
                {
                    "step_id": 2,
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "anchor_object",
                        "fields": {
                            "language_name": {"type": "text"},
                            "article_count": {"type": "number"},
                        },
                    },
                    "save_as": "metadata",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )


def _cards_plan() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "Extract cards with title, description, and href.",
            "start_url": "https://cards.sample.test",
            "allowed_domains": ["cards.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {"description": "Cards", "required_fields": ["cards"]},
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://cards.sample.test"}},
                {
                    "step_id": 2,
                    "action": "extract_by_intent",
                    "args": {
                        "intent": "card_items",
                        "output_key": "cards",
                        "fields": {
                            "title": {"type": "text"},
                            "description": {"type": "description"},
                            "href": {"type": "url"},
                        },
                    },
                    "save_as": "cards",
                },
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )


def test_pf_a2_like_anchor_object_rejects_swapped_semantic_shape():
    result = ExecutionResult(
        status="success",
        extracted_data={
            "metadata": {
                "language_name": "2 103 000+ articles",
                "article_count": "Francais",
            }
        },
        logs=[StepLog(step_id=2, action="extract_by_intent", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_anchor_object_plan(), result)

    assert verdict.verdict == "reject"
    assert any("language_name" in issue for issue in verdict.issues)
    assert any("article_count" in issue for issue in verdict.issues)


def test_anchor_object_accepts_populated_label_and_count_without_llm():
    result = ExecutionResult(
        status="success",
        extracted_data={
            "metadata": {
                "language_name": "Français",
                "article_count": "2 761 000+ articles",
            }
        },
        logs=[StepLog(step_id=2, action="extract_by_intent", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_anchor_object_plan(), result)

    assert verdict.verdict == "accept"
    assert verdict.confidence >= 0.9


def test_anchor_object_accepts_parent_metadata_required_field_without_llm():
    plan = _anchor_object_plan()
    plan.expected_result.required_fields = ["www_wikipedia_org", "metadata"]
    result = ExecutionResult(
        status="success",
        extracted_data={
            "www_wikipedia_org": "https://www.wikipedia.org/",
            "metadata": {
                "language_name": "Français",
                "article_count": "2 761 000+ articles",
            },
        },
        logs=[StepLog(step_id=2, action="extract_by_intent", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(plan, result)

    assert verdict.verdict == "accept"
    assert verdict.confidence >= 0.9


def test_plan_validator_accepts_anchor_object_nested_required_fields():
    plan = _anchor_object_plan()
    plan.steps[1].save_as = None
    plan.steps[1].args["output_key"] = "metadata"

    PlanValidator().validate(plan)


def test_pf_g2_like_raw_only_cards_are_rejected():
    result = ExecutionResult(
        status="success",
        extracted_data={
            "cards": [
                {
                    "title": None,
                    "description": None,
                    "href": None,
                    "raw_text": "Python on Arm: 2025 Update Other unrelated story text",
                    "selector": "main",
                }
            ]
        },
        logs=[StepLog(step_id=2, action="extract_by_intent", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_cards_plan(), result)

    assert verdict.verdict == "reject"
    assert "raw container text" in verdict.issues[0]


def test_contact_field_shapes_accept_plausible_values():
    issues = LLMVerifier._validate_semantic_value_quality(
        required_fields=["address", "phone", "email"],
        extracted_data={
            "address": "199034, University Embankment, 7-9",
            "phone": "+7 (812) 328-20-00",
            "email": "office@example.org",
        },
        goal="Extract contact fields.",
    )

    assert issues == []
