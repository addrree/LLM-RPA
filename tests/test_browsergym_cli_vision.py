from app.browsergym_integration.config import BrowserGymRunConfig
from scripts import run_minwob_subset


def test_config_default_use_vision_false():
    assert BrowserGymRunConfig(env_id="browsergym/miniwob.click-button").use_vision is False


def test_minwob_subset_parse_use_vision():
    args = run_minwob_subset.parse_args(["--use-vision", "--limit", "1"])
    assert args.use_vision is True
