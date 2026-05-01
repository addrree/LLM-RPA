from scripts.run_webarena_deterministic_subset import compute_summary


def test_compute_summary_rate_and_buckets():
    rows = [
        {"status": "success", "steps": 3, "runtime_sec": 1.0},
        {"status": "failed", "failure_stage": "runtime", "steps": 2, "runtime_sec": 2.0},
        {"status": "partial", "steps": 4, "runtime_sec": 3.0},
        {"status": "skipped", "skip_reason": "llm_judge", "steps": 0, "runtime_sec": 0.0},
    ]
    s = compute_summary(rows)
    assert s["attempted_tasks"] == 3
    assert s["success_count"] == 1
    assert s["success_rate"] == 1 / 3
    assert s["skipped_llm_judge_tasks"] == 1
    assert s["failure_buckets"]["runtime"] == 1
    assert s["failure_buckets"]["partial"] == 1
