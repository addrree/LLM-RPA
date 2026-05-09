from scripts.run_minwob_subset import build_aggregate


def test_aggregate_metrics_and_failure_buckets():
    aggregate = build_aggregate(
        [
            {"success": True, "reward": 1.0, "steps_count": 2, "runtime_sec": 1.0, "status": "success"},
            {"success": False, "reward": 0.0, "steps_count": 4, "runtime_sec": 3.0, "failure_stage": "runtime", "status": "failed"},
            {"success": False, "reward": None, "steps_count": 0, "runtime_sec": 0.5, "failure_stage": "env_validation", "status": "skipped"},
        ],
        use_vision=True,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert aggregate["total_tasks"] == 3
    assert aggregate["success_count"] == 1
    assert aggregate["success_rate"] == 1 / 3
    assert aggregate["mean_reward"] == 0.5
    assert aggregate["mean_steps"] == 2.0
    assert aggregate["mean_runtime_sec"] == 1.5
    assert aggregate["failure_buckets"] == {"env_validation": 1, "runtime": 1}
    assert aggregate["use_vision"] is True
