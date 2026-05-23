# LLM-RPA

LLM-RPA is a compact browser automation pipeline driven by an LLM planner, a Playwright executor, verifier, and artifact exporters. The repository currently keeps two supported runtime paths:

1. Main pipeline: `app.main → WorkflowManager → Planner/Replanner → PlanValidator → PlaywrightExecutor → LLMVerifier → artifacts/export`.
2. MiniWoB benchmark: `scripts/run_minwob_subset.py → BrowserGymRunner → BrowserGymAgentAdapter → MiniWoB grounding/action mapping → BrowserGym env.step`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

For MiniWoB/BrowserGym runs, install the BrowserGym extras used by the benchmark:

```bash
pip install -r requirements-browsergym.txt
```

Configure one of the production LLM backends:

```bash
export LLM_BACKEND=ollama_cloud
export OLLAMA_API_KEY="..."
# Optional model overrides:
export OLLAMA_MODEL=qwen3-vl:4b
export OLLAMA_TIMEOUT_SEC=300
```

Supported production backends are `ollama` and `ollama_cloud`.

## Main pipeline

Run the production Playwright workflow with two-stage planning and JSON export:

```bash
python -m app.main \
  --backend ollama_cloud \
  --goal "Открой https://www.wikipedia.org и найди число статей на английском на главной странице" \
  --two-stage-planning \
  --export-format json
```

Useful runtime options:

```bash
python -m app.main \
  --backend ollama_cloud \
  --goal "Open https://www.wikipedia.org and extract the English article count" \
  --two-stage-planning \
  --show-browser \
  --slow-mo 300 \
  --export-format json
```

## MiniWoB benchmark

List available MiniWoB tasks:

```bash
python scripts/list_minwob_tasks.py
```

Run the supported MiniWoB subset benchmark:

```bash
python scripts/run_minwob_subset.py \
  --backend ollama_cloud \
  --max-steps 15 \
  --limit 15 \
  --use-vision
```

The MiniWoB runner writes aggregate JSON/CSV results under `artifacts/browsergym/` by default.

MiniWoB subset split:
- `--subset action` / `--subset basic`: UI action grounding tasks.
- `--subset extraction`: text/data/list/grid/email/calendar/tree extraction tasks (no canvas geometry).
- `--subset visual`: visual-spatial canvas tasks (currently `find-midpoint`), reported separately.

## Artifacts

Main pipeline artifacts are written under `artifacts/`, including:

- `artifacts/runs/` for per-run workflow payloads and diagnostics.
- `artifacts/exports/` for JSON/CSV exports requested with `--export-format`.
- `artifacts/screenshots/` for screenshots captured by workflow steps.
- `artifacts/videos/` when `--record-video` is enabled.
- `artifacts/browsergym/` for MiniWoB benchmark reports.
