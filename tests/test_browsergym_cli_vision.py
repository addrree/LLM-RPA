from app.browsergym_integration.config import BrowserGymRunConfig
from scripts import run_browsergym_smoke, run_browsergym_webarena, run_webarena_deterministic_subset


def test_config_default_use_vision_false():
    assert BrowserGymRunConfig(env_id="browsergym/openended").use_vision is False


def test_smoke_parse_use_vision():
    args = run_browsergym_smoke.parse_args(["--env-id", "browsergym/openended", "--goal", "g", "--use-vision"])
    assert args.use_vision is True


def test_webarena_parse_use_vision():
    args = run_browsergym_webarena.parse_args(["--env-id", "browsergym/webarena.10", "--goal", "g", "--use-vision"])
    assert args.use_vision is True


def test_subset_parse_use_vision():
    args = run_webarena_deterministic_subset.parse_args(["--use-vision"])
    assert args.use_vision is True
