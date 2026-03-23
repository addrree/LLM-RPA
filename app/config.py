from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"
RESULTS_DIR = ARTIFACTS_DIR / "results"
LOGS_DIR = ARTIFACTS_DIR / "logs"

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_MAX_STEPS = 20
GLOBAL_MAX_REPLANS = 2
GLOBAL_TIMEOUT_SEC = 60