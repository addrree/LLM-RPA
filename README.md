# LLM-RPA (MVP)

Минимальный end-to-end прототип агентной веб-автоматизации:

1. **Planner (LLM)**: строит JSON-план из цели пользователя.
2. **Validator**: проверяет безопасность/корректность плана.
3. **Executor (Playwright)**: исполняет шаги на странице.
4. **Verifier (LLM)**: оценивает, достигнута ли цель (включая screenshot через vision).

## Быстрый запуск

### 1) Установить зависимости

```bash
pip install -r requirements.txt
playwright install
```

### 2) Выбрать backend (local или cloud)

Проект поддерживает:
- `ollama` — локальный Ollama backend (`http://localhost:11434`)
- `ollama_cloud` — прямой Ollama Cloud API backend (`https://ollama.com/api/...`, через `OLLAMA_API_KEY`)
- `dummy` — заглушка без реальных LLM вызовов

Для локального режима проверьте, что Ollama запущена:

```bash
ollama serve
```

По умолчанию проект ожидает API на `http://localhost:11434`.

### 3) Проверить, что модель доступна

```bash
ollama list
ollama pull qwen3-vl:4b
```

### 4) Настроить переменные окружения

```bash
cp .env.example .env
```

Минимальный `.env`:

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:4b
OLLAMA_TIMEOUT_SEC=300
```

Пример `.env` для **Ollama Cloud**:

```env
LLM_BACKEND=ollama_cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_MODEL=qwen3-vl:4b
# optional per-role overrides:
# OLLAMA_PLANNER_MODEL=qwen3-vl:4b
# OLLAMA_VERIFIER_MODEL=qwen3-vl:4b
OLLAMA_TIMEOUT_SEC=300
```

### 5) Запуск MVP в dummy-режиме

```bash
python -m app.main --dummy
```

### 6) Запуск MVP с Ollama

```bash
python -m app.main --backend ollama
```

Запуск с **Ollama Cloud**:

```bash
python -m app.main --backend ollama_cloud
```

С кастомным timeout (если локальная модель отвечает долго):

```bash
OLLAMA_TIMEOUT_SEC=300 python -m app.main --backend ollama
```

Или просто:

```bash
python -m app.main
```

(если `LLM_BACKEND=ollama` в env).

### 7) Пример простого сценария

```bash
python -m app.main --goal "Open https://www.wikipedia.org, extract the h1 text, take screenshot and finish."
```

Рабочий пример для cloud backend:

```bash
python -m app.main --backend ollama_cloud --goal "Open https://www.wikipedia.org, extract the h1 text, take a screenshot, and finish." --show-browser --slow-mo 500 --export-format json
```

## Что появляется в artifacts

После каждого запуска сохраняются:
- `artifacts/results/plan_<timestamp>.json`
- `artifacts/results/execution_<timestamp>.json`
- `artifacts/results/verdict_<timestamp>.json`
- `artifacts/logs/logs_<timestamp>.json`

Скриншот (`artifacts/screenshots/...`) появляется только если шаг `screenshot` реально выполнен.
Если выполнение упало раньше (например, `open_url` с DNS/сеть ошибкой), скриншота не будет.


## BrowserGym sidecar vision mode

The BrowserGym integration remains a separate sidecar evaluation layer; it does not replace the main `WorkflowManager` → `Planner` → `Validator` → `PlaywrightExecutor` pipeline or the internal benchmark suites.

Vision mode is opt-in via `--use-vision`. When enabled, BrowserGym screenshots/images are converted to PNG base64 and sent only in the Ollama chat API payload for the planner (for example `qwen3-vl:235b-cloud` through Ollama Cloud). Raw arrays and base64 screenshots are not saved in prompts, JSON reports, `internal_plan`, `selected_step`, or artifacts. Reports only contain safe diagnostics such as `vision_used` and `vision_image_present`.

Non-vision BrowserGym smoke:

```bash
python scripts/run_browsergym_smoke.py \
  --env-id browsergym/openended \
  --start-url https://www.python.org/ \
  --goal "Find the main heading of the page" \
  --backend ollama_cloud \
  --max-steps 5
```

Vision BrowserGym smoke:

```bash
python scripts/run_browsergym_smoke.py \
  --env-id browsergym/openended \
  --start-url https://www.python.org/ \
  --goal "Using the screenshot, find the main heading of the page" \
  --backend ollama_cloud \
  --max-steps 5 \
  --use-vision
```

Vision WebArena deterministic subset:

```bash
python scripts/run_webarena_deterministic_subset.py \
  --backend ollama_cloud \
  --max-steps 20 \
  --limit 5 \
  --use-vision
```

BrowserGym WebArena still requires self-hosted WebArena services and `WA_*` URLs.

## Benchmark / evaluation layer

Добавлен компактный воспроизводимый benchmark-suite с обобщёнными типами задач:

- `single_value_extraction`
- `anchored_value_extraction`
- `repeated_structured_items`
- `navigation_then_extraction`
- `multi_step_information_retrieval`
- `negative_or_ambiguous_case`

Сценарии v1 лежат в `benchmarks/scenarios/core_task_suite.json` (алиас: `benchmarks/scenarios/core_task_suite_v1.json`) и не привязаны к одному домену.
Дополнительно есть расширенная версия v2: `benchmarks/scenarios/core_task_suite_v2.json` (совместимый алиас: `benchmarks/scenarios/extended_generalized_task_suite.json`)
с теми же task families и дополнительными стабильными публичными сайтами.

Дополнительно есть быстрый smoke-suite: `benchmarks/scenarios/smoke_generalized_suite.json`
с тремя категориями:
- `single_value_extraction`
- `anchored_value_extraction`
- `repeated_structured_items`

### Baseline: запуск core suites

Запустить все сценарии **core_task_suite_v1**:

```bash
python -m app.main --benchmark-all --benchmark-suite benchmarks/scenarios/core_task_suite_v1.json --backend ollama
```

Запустить все сценарии **core_task_suite_v2**:

```bash
python -m app.main --benchmark-all --benchmark-suite benchmarks/scenarios/core_task_suite_v2.json --backend ollama
```

Запустить smoke-suite отдельно:

```bash
python -m app.main --benchmark-all --benchmark-suite benchmarks/scenarios/smoke_generalized_suite.json --backend ollama
```

Запустить один сценарий:

```bash
python -m app.main --benchmark-scenario repeated_listing_cards --backend ollama
```

Запустить категорию:

```bash
python -m app.main --benchmark-category navigation_then_extraction --backend ollama
```

После запуска формируются агрегированные отчёты:

- `artifacts/benchmarks/benchmark_summary_<suite_id>_<timestamp>.json`
- `artifacts/benchmarks/benchmark_summary_<suite_id>_<timestamp>.csv`
- `artifacts/benchmarks/benchmark_multi_run_summary_<suite_id>_<timestamp>.json` (для `--benchmark-runs > 1` или `--benchmark-summarize-report`)

Также теперь можно запустить стабильностный анализ повторов:

```bash
python scripts/run_benchmark_repeats.py --suite benchmarks/scenarios/core_task_suite_v2.json --runs 5 --backend ollama
```

Скрипт не меняет benchmark-логику исполнения; он запускает тот же runner N раз и строит aggregate:
- pass rate by scenario;
- verifier accept rate by scenario;
- mean/std runtime by scenario;
- correction usage by scenario;
- failure buckets by scenario.

Сделать multi-run и автоматически получить summary по нескольким прогонам:

```bash
python -m app.main --benchmark-all --benchmark-suite benchmarks/scenarios/extended_generalized_task_suite.json --benchmark-runs 5 --backend ollama
```

Построить multi-run summary из уже сохранённых `benchmark_summary_*.json`:

```bash
python -m app.main   --benchmark-summarize-report artifacts/benchmarks/benchmark_summary_core_generalized_task_suite_v2_20260424_100000.json   --benchmark-summarize-report artifacts/benchmarks/benchmark_summary_core_generalized_task_suite_v2_20260424_103000.json
```

### Метрики benchmark summary

В summary считаются:

- total scenarios
- positive execution success rate
- positive verifier accept rate
- negative expected reject rate
- plan validation pass rate
- correction retry usage rate
- correction recovery rate
- export success rate
- mean runtime
- mean planning time
- mean execution time
- mean verification time
- mean correction time

Краткие определения:
- `positive_execution_success_rate` — доля позитивных (`should_succeed=true`) сценариев, где `execution_status=success`.
- `positive_verifier_accept_rate` / `verifier_accept_rate` — доля позитивных сценариев, где verifier дал `accept`.
- `negative_expected_reject_rate` — доля семантических негативных (`should_succeed=false`, без technical failure), где verifier дал ожидаемый `reject`.
- `correction_recovery_rate` — доля сценариев с corrective attempts, которые завершились ожидаемым исходом.
- `export_success_rate` — доля сценариев, где экспорт отчетов прошел успешно.

## Smoke test

```bash
pytest -q
```

> Тест не вызывает Ollama API и проверяет совместимость `planner -> validator -> verifier` на `DummyLLMClient`.

## BrowserGym / WebArena external evaluation

Для внешней оценки добавлен отдельный слой BrowserGym-интеграции (не заменяет внутренние suites v1/v2/v3).

- Документация: `docs/browsergym_integration.md`
- Openended smoke: `python scripts/run_browsergym_smoke.py ...`
- Реальный BrowserGym WebArena: `python scripts/run_browsergym_webarena.py ...` (требует self-hosted WebArena и `WA_*` env vars).

## Deterministic WebArena subset (without OpenAI API)

Для запуска только deterministic subset (без fuzzy/LLM judge задач):

```bash
python scripts/check_webarena_env.py
python scripts/list_webarena_tasks.py
python scripts/run_webarena_deterministic_subset.py --backend ollama_cloud --max-steps 20 --limit 10
```

Результаты сохраняются в `artifacts/browsergym/` (JSON/CSV отчёты и inventory задач).

## BrowserGym MiniWoB++ external benchmark

The BrowserGym sidecar can run the official MiniWoB++ benchmark subset without touching the internal benchmark suites or the main Playwright workflow. MiniWoB++ does **not** require WebArena `WA_*` variables or Docker-hosted WebArena services.

Setup:

```bash
pip install browsergym-miniwob
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git external/miniwob-plusplus
git -C external/miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
python -m http.server 8765 --directory .\external\miniwob-plusplus\miniwob\html\miniwob
```

PowerShell:

```powershell
$env:MINIWOB_URL="http://127.0.0.1:8765"
```

Commands:

```bash
python scripts/list_minwob_tasks.py
python scripts/run_minwob_subset.py --backend ollama_cloud --max-steps 10 --limit 3
python scripts/run_minwob_subset.py --backend ollama_cloud --max-steps 10 --limit 3 --use-vision
```

Results are saved as JSON and CSV under `artifacts/browsergym/` with success rate, mean reward, mean steps, mean runtime, and failure buckets.
