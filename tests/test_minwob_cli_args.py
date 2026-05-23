from scripts.run_minwob_subset import parse_args


def test_parse_minwob_cli_args():
    args = parse_args([
        "--use-vision",
        "--limit",
        "3",
        "--task-ids",
        "click-button,enter-text",
        "--backend",
        "ollama_cloud",
    ])
    assert args.use_vision is True
    assert args.limit == 3
    assert args.task_ids == "click-button,enter-text"
    assert args.backend == "ollama_cloud"


def test_parse_minwob_subset_arg():
    args = parse_args(["--subset", "basic"])
    assert args.subset == "basic"
    args2 = parse_args(["--subset", "extraction"])
    assert args2.subset == "extraction"
    args3 = parse_args(["--subset", "visual"])
    assert args3.subset == "visual"
