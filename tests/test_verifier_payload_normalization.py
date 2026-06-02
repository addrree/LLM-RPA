from app.verifier.llm_verifier import LLMVerifier


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
