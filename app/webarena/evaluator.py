from __future__ import annotations

from typing import Protocol

from app.benchmark.runner import BenchmarkScenarioResult


class WebArenaEvaluator(Protocol):
    def evaluate(self, result: BenchmarkScenarioResult, *, expected: dict | None = None) -> dict: ...


class WebArenaEvaluationAdapter:
    """Produces WebArena-style normalized outcome tags from internal benchmark results."""

    @staticmethod
    def evaluate(result: BenchmarkScenarioResult, *, expected: dict | None = None) -> dict:
        expected = expected or {}
        success = result.execution_status == "success" and result.verifier_verdict == "accept"
        return {
            "task_id": result.scenario_id,
            "success": success,
            "verdict": result.verifier_verdict,
            "execution_status": result.execution_status,
            "failure_stage": result.failure_stage,
            "runtime_sec": result.runtime_sec,
            "metadata": expected,
        }
