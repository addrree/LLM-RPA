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

### 2) Проверить, что Ollama запущена локально

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
```

### 5) Запуск MVP в dummy-режиме

```bash
python -m app.main --dummy
```

### 6) Запуск MVP с Ollama

```bash
python -m app.main --backend ollama
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

## Что появляется в artifacts

После каждого запуска сохраняются:
- `artifacts/results/plan_<timestamp>.json`
- `artifacts/results/execution_<timestamp>.json`
- `artifacts/results/verdict_<timestamp>.json`
- `artifacts/logs/logs_<timestamp>.json`

Скриншот (`artifacts/screenshots/...`) появляется только если шаг `screenshot` реально выполнен.
Если выполнение упало раньше (например, `open_url` с DNS/сеть ошибкой), скриншота не будет.

## Smoke test

```bash
pytest -q
```

> Тест не вызывает Ollama API и проверяет совместимость `planner -> validator -> verifier` на `DummyLLMClient`.
