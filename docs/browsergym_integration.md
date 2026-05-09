# BrowserGym / WebArena external evaluation layer

Internal benchmark suites (`v1/v2/v3`) remain the reproducible in-project benchmark layer.

BrowserGym/WebArena is integrated as a **separate external evaluation layer** via `app/browsergym_integration`.

- `app/webarena/*` remains an internal WebArena-like adapter, not a true BrowserGym WebArena score.
- True BrowserGym WebArena requires self-hosted WebArena services and `WA_*` env vars.

## Commands

Openended smoke:

```bash
python scripts/run_browsergym_smoke.py \
  --env-id browsergym/openended \
  --start-url https://www.python.org/ \
  --goal "Find the main heading of the page" \
  --backend ollama_cloud \
  --max-steps 5
```

WebArena availability/run:

```bash
python scripts/run_browsergym_webarena.py \
  --env-id browsergym/webarena.10 \
  --goal "Complete the task according to the environment instruction" \
  --backend ollama_cloud \
  --max-steps 15
```

Tests:

```bash
python -m pytest -q
python -m pytest tests/test_browsergym_observation_adapter.py tests/test_browsergym_action_mapper.py tests/test_browsergym_report.py -q
```

## Artifacts

Reports are saved to `artifacts/browsergym/browsergym_run_<env>_<timestamp>.json`.

## MVP limitations

- Minimal action mapping (`click/type/fill/press/scroll/noop/finish`).
- Full WebArena score depends on external services and environment setup.
- This does not claim universal any-site agent capability.
- AgentLab can be added later as a larger experiment orchestration layer.

## WebArena deterministic subset without OpenAI API

Этот режим запускает только детерминированное подмножество WebArena задач и **не является полным официальным WebArena score**.

Исключаются задачи, где в конфиге есть `llm_fuzzy_match`, `fuzzy_match`, `llm_judge`, `openai` или `gpt`-зависимые evaluator-ы.

Команды:

```bash
python scripts/check_webarena_env.py
python scripts/list_webarena_tasks.py
python scripts/run_webarena_deterministic_subset.py --backend ollama_cloud --max-steps 20 --limit 10
```

Артефакты сохраняются в `artifacts/browsergym/`:
- `webarena_task_inventory.json`
- `webarena_deterministic_subset_report.json`
- `webarena_deterministic_subset_report.csv`

## BrowserGym vision mode (opt-in)

BrowserGym sidecar can run the planner as a vision-aware agent when a screenshot/image is available in the BrowserGym observation. This is **opt-in** and does not replace the main `WorkflowManager` / internal benchmark suites / `PlaywrightExecutor` pipeline.

In vision mode the agent sends:

```text
goal + compact text/AX snapshot + screenshot image
→ qwen3-vl planner
→ next BrowserGym action or local extraction
→ BrowserGym report
```

Safety/serialization guarantees:

- enable it only with `--use-vision`;
- raw BrowserGym `numpy` arrays are converted to PNG base64 only for the LLM API payload;
- base64 screenshots are not added to prompts, `internal_plan`, `selected_step`, JSON reports, or artifacts;
- reports only record safe diagnostics: `vision_used` and `vision_image_present`;
- when no image is present, the sidecar falls back to the existing text/AX snapshot path without dummy/local model fallback.

Vision smoke:

```bash
python scripts/run_browsergym_smoke.py \
  --env-id browsergym/openended \
  --start-url https://www.python.org/ \
  --goal "Using the screenshot, find the main heading of the page" \
  --backend ollama_cloud \
  --max-steps 5 \
  --use-vision
```

Deterministic WebArena subset with vision enabled:

```bash
python scripts/run_webarena_deterministic_subset.py \
  --backend ollama_cloud \
  --max-steps 20 \
  --limit 5 \
  --use-vision
```

WebArena still requires self-hosted services and the `WA_*` environment variables; `--use-vision` only changes how BrowserGym observations are passed to the planner.

## MiniWoB++ via BrowserGym

MiniWoB++ is supported as an **external BrowserGym sidecar benchmark**. It does not use the internal benchmark suites (`v1`/`v2`/`v3`), `WorkflowManager`, `BenchmarkRunner`, or `PlaywrightExecutor`, and it does not require WebArena `WA_*` variables or Docker-hosted WebArena services.

### Install and serve MiniWoB++

Install the BrowserGym MiniWoB plugin and serve the static MiniWoB HTML files locally:

```bash
pip install browsergym-miniwob
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git external/miniwob-plusplus
git -C external/miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
python -m http.server 8765 --directory .\external\miniwob-plusplus\miniwob\html\miniwob
```

In PowerShell, point BrowserGym MiniWoB to the local static server:

```powershell
$env:MINIWOB_URL="http://127.0.0.1:8765"
```

On POSIX shells, use:

```bash
export MINIWOB_URL="http://127.0.0.1:8765"
```

`MINIWOB_URL` is required for actually running tasks. Listing the Gymnasium registry can still work without it; the list script prints a warning instead of failing.

### List available MiniWoB task IDs

```bash
python scripts/list_minwob_tasks.py
```

The inventory is saved by default to:

```text
artifacts/browsergym/miniwob_task_inventory.json
```

Each inventory entry contains the BrowserGym env ID, parsed MiniWoB task name, benchmark name, and flags documenting that MiniWoB does not require WebArena environment variables or an LLM judge.

### Run a text-only MiniWoB subset

```bash
python scripts/run_minwob_subset.py \
  --backend ollama_cloud \
  --max-steps 10 \
  --limit 3
```

You can select explicit tasks by full env ID or bare task name:

```bash
python scripts/run_minwob_subset.py \
  --task-ids browsergym/miniwob.click-button,enter-text \
  --max-steps 10
```

You can also filter with regex patterns:

```bash
python scripts/run_minwob_subset.py --include click --exclude sequence --limit 5
```

### Run a vision MiniWoB subset

Vision mode is opt-in and follows the same BrowserGym sidecar guarantees as the openended/WebArena smoke paths: screenshots are sent to the planner only through the LLM image payload and raw screenshots/base64 are not written to reports.

```bash
python scripts/run_minwob_subset.py \
  --backend ollama_cloud \
  --max-steps 10 \
  --limit 3 \
  --use-vision
```

### MiniWoB artifacts and metrics

`run_minwob_subset.py` writes both JSON and CSV artifacts under `artifacts/browsergym/` by default:

- `miniwob_results_<timestamp>.json`
- `miniwob_results_<timestamp>.csv`

The JSON aggregate includes:

- `suite_id="browsergym_miniwob_subset_v1"`
- `total_tasks`
- `success_count`
- `success_rate`
- `mean_reward`
- `mean_steps`
- `mean_runtime_sec`
- `failure_buckets`
- `use_vision`
- per-task result rows with reward, success, termination/truncation, step count, runtime, failure stage, error message, final answer, vision diagnostics, and any per-run artifact path.

If `MINIWOB_URL` is missing, the runner writes a skipped JSON/CSV report with a clear `env_validation` message instead of raising a stacktrace.
