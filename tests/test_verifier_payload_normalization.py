from app.schemas.execution import ExecutionResult, StepLog
from app.schemas.task_spec import TaskSpec
from app.verifier.llm_verifier import LLMVerifier


class _FailIfCalledClient:
    def generate_verifier_artifact(self, *args, **kwargs):
        raise AssertionError("Verifier LLM should not be called for deterministic preprocessing coverage.")


def _metadata_plan(required_fields=None) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "Extract object metadata.",
            "start_url": "https://docs.sample.test/item",
            "allowed_domains": ["docs.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {
                "description": "Object metadata",
                "required_fields": required_fields or ["name", "description", "url"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://docs.sample.test/item"}},
                {"step_id": 2, "action": "extract_by_intent", "args": {"intent": "package_metadata"}, "save_as": "package_metadata"},
                {"step_id": 3, "action": "finish", "args": {}},
            ],
        }
    )


def _visual_count_plan() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "goal": "Visually count the links.",
            "start_url": "https://visual.sample.test/",
            "allowed_domains": ["visual.sample.test"],
            "constraints": {"max_steps": 5, "max_replans": 1, "timeout_sec": 20},
            "expected_result": {
                "description": "Link count",
                "required_fields": ["language_link_count"],
            },
            "steps": [
                {"step_id": 1, "action": "open_url", "args": {"url": "https://visual.sample.test/"}},
                {"step_id": 2, "action": "visual_observe", "args": {}, "save_as": "page_snapshot"},
                {
                    "step_id": 3,
                    "action": "visual_extract_object_count",
                    "args": {"target": "link"},
                    "save_as": "language_link_count",
                },
                {"step_id": 4, "action": "finish", "args": {}},
            ],
        }
    )


def test_verifier_preprocessing_accepts_nested_metadata_required_fields():
    result = ExecutionResult(
        status="success",
        extracted_data={
            "package_metadata": {
                "name": "Sample Tool",
                "description": "A compact metadata fixture.",
                "url": "https://docs.sample.test/item",
            }
        },
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    normalized = LLMVerifier._normalized_extracted_data_for_verifier(
        required_fields=["name", "description", "url"],
        result=result,
    )
    assert normalized["name"] == "Sample Tool"
    assert normalized["description"] == "A compact metadata fixture."
    assert normalized["url"] == "https://docs.sample.test/item"

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_metadata_plan(), result)

    assert verdict.verdict == "accept"
    assert verdict.task_completed is True


def test_verifier_preprocessing_reports_partial_metadata_without_rejecting():
    result = ExecutionResult(
        status="success",
        extracted_data={"package_metadata": {"name": "Sample Tool"}},
        logs=[StepLog(step_id=1, action="open_url", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_metadata_plan(), result)

    assert verdict.verdict == "uncertain"
    assert verdict.task_completed is False
    assert "missing: description, url" in verdict.issues[0]


def test_verifier_preprocessing_accepts_populated_count_fields_without_llm():
    result = ExecutionResult(
        status="success",
        extracted_data={"language_link_count": 10},
        logs=[StepLog(step_id=3, action="visual_extract_object_count", status="success")],
    )

    verdict = LLMVerifier(_FailIfCalledClient()).verify(_visual_count_plan(), result)

    assert verdict.verdict == "accept"
    assert verdict.task_completed is True


def test_verifier_payload_normalization_adds_missing_summary():
    payload = LLMVerifier._normalize_verdict_payload(
        {
            "task_completed": False,
            "confidence": 0.9,
            "verdict": "reject",
            "issues": ["Missing one required field."],
        }
    )

    assert payload["summary"] == "Missing one required field."
    assert payload["verdict"] == "reject"
    assert payload["confidence"] == 0.9


def test_verifier_payload_normalization_defaults_invalid_payload():
    payload = LLMVerifier._normalize_verdict_payload({"issues": "bad shape"})

    assert payload["task_completed"] is False
    assert payload["verdict"] == "uncertain"
    assert payload["issues"] == ["bad shape"]
    assert payload["summary"] == "bad shape"
