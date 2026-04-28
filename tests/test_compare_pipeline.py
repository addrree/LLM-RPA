import asyncio

from app.executor.action_handlers import ActionHandlers
from app.orchestrator.workflow_manager import WorkflowManager
from app.schemas.execution import ExecutionResult


def test_compare_structured_values_populates_structured_outputs():
    handlers = ActionHandlers()
    runtime_state = {
        "extracted_data": {
            "section_a_data": {"name": "A", "value": 10},
            "section_b_data": {"name": "A", "value": 12},
        }
    }
    args = {"left_key": "section_a_data", "right_key": "section_b_data"}

    comparison = asyncio.run(handlers.compare_structured_values(page=None, args=args, runtime_state=runtime_state))

    extracted = runtime_state["extracted_data"]
    assert comparison["status"] == "different"
    assert extracted["structured_comparison"]["status"] == "different"
    assert extracted["comparison"]["exact_match"] is False
    assert extracted["compare_status"] == "different"
    assert extracted["comparison_left_summary"]["label"] == "section_a_data"
    assert extracted["comparison_right_summary"]["label"] == "section_b_data"


def test_augment_multi_step_comparison_sets_compare_status():
    result = ExecutionResult(
        status="success",
        extracted_data={
            "structured_comparison": {
                "left_key": "section_a_data",
                "right_key": "section_b_data",
                "status": "equal",
                "exact_match": True,
            },
            "section_a_data": {"x": 1},
            "section_b_data": {"x": 1},
        },
    )

    WorkflowManager._augment_multi_step_comparison(result)

    assert result.extracted_data["compare_status"] == "equal"
    assert result.extracted_data["comparison"]["exact_match"] is True
    assert "combined_result" in result.extracted_data


def test_compare_structured_values_empty_sources_not_equal():
    handlers = ActionHandlers()
    runtime_state = {
        "extracted_data": {
            "source_a": {},
            "source_b": {},
        }
    }
    args = {"left_key": "source_a", "right_key": "source_b"}
    comparison = asyncio.run(handlers.compare_structured_values(page=None, args=args, runtime_state=runtime_state))
    assert comparison["status"] == "insufficient_data"
    assert comparison["exact_match"] is False
    assert comparison["reason"] == "empty_source"
