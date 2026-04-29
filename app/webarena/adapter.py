from __future__ import annotations

from urllib.parse import urlparse

from app.benchmark.scenario_loader import BenchmarkScenario
from app.webarena.loader import WebArenaTaskConfig


class WebArenaTaskAdapter:
    """Maps WebArena-like task configs into internal BenchmarkScenario-compatible objects."""

    @staticmethod
    def to_scenario(task: WebArenaTaskConfig, *, category: str = "navigation_then_extraction") -> BenchmarkScenario:
        domain = urlparse(task.start_url).netloc
        preconditions = list(task.constraints)
        if task.allowed_domains:
            preconditions.append(f"Stay within domains: {', '.join(task.allowed_domains)}")
        elif domain:
            preconditions.append(f"Stay within domain: {domain}")

        return BenchmarkScenario(
            scenario_id=task.task_id,
            goal=task.objective,
            start_url=task.start_url,
            category=category,
            task_family=category,
            description=f"WebArena-like task from split={task.split}, site={task.site}",
            expected_output_type="mixed",
            required_top_level_fields=[],
            expected_min_items=0,
            expected_item_fields=[],
            should_succeed=True,
            notes="generated_by_webarena_adapter",
            preconditions=preconditions,
        )

    @staticmethod
    def benchmark_metadata(task: WebArenaTaskConfig) -> dict:
        return {
            "webarena_task_id": task.task_id,
            "webarena_split": task.split,
            "webarena_site": task.site,
            "adapter_version": "v1",
        }
