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
