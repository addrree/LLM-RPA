# BrowserGym MiniWoB runtime

The supported BrowserGym integration is the MiniWoB benchmark sidecar. It is intentionally separate from the main Playwright workflow and uses this path:

`scripts/run_minwob_subset.py → BrowserGymRunner → BrowserGymAgentAdapter → MiniWoB grounding/action mapping → BrowserGym env.step`.

## Setup

```bash
pip install -r requirements-browsergym.txt
export LLM_BACKEND=ollama_cloud
export OLLAMA_API_KEY="..."
```

## List tasks

```bash
python scripts/list_minwob_tasks.py
```

## Run MiniWoB subset

```bash
python scripts/run_minwob_subset.py \
  --backend ollama_cloud \
  --max-steps 15 \
  --limit 15 \
  --use-vision
```

Outputs are written to `artifacts/browsergym/` as JSON and CSV aggregate reports. The runner stores per-step action mapping diagnostics, selected MiniWoB candidates, rewards, termination flags, and failure buckets.

## Action mapping

The MiniWoB mapper prefers BrowserGym-native action strings when reliable element identifiers are available:

- `click("<bid>", "left")`
- `fill("<bid>", "<text>")`

When MiniWoB observations expose only coordinates, the mapper can use scaled coordinate actions and records the mapping strategy in the report.
