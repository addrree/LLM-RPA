# WebArena-like quickstart

## 1) Подготовьте JSON с задачами

Поддерживаются два формата:

### A. Массив задач
```json
[
  {
    "task_id": "wa_001",
    "objective": "Open the docs page and extract API base URL.",
    "start_url": "https://example.com/docs",
    "allowed_domains": ["example.com"],
    "constraints": ["Do not navigate outside docs section"],
    "split": "dev",
    "site": "example_docs"
  }
]
```

### B. Объект с `tasks`
```json
{
  "tasks": [
    {
      "task_id": "wa_001",
      "objective": "Open the docs page and extract API base URL.",
      "start_url": "https://example.com/docs"
    }
  ]
}
```

## 2) Запустите WebArena-like suite

```bash
python scripts/run_webarena_suite.py \
  --input /path/to/webarena_tasks.json \
  --backend ollama
```

Опционально:
- `--category navigation_then_extraction` (по умолчанию)
- `--show-browser`
- `--two-stage-planning`
- `--scenario-id wa_001` (фильтр)

## 3) Что происходит внутри

1. `app/webarena/loader.py` загружает задачи.
2. `app/webarena/adapter.py` превращает их в `BenchmarkScenario`.
3. Запускается стандартный `BenchmarkRunner` (без изменения v1/v2/v3 пайплайна).
4. Метрики и отчёты пишутся в `artifacts/benchmarks`.
