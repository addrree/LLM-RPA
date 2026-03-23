# LLM-RPA (MVP)

Минимальный end-to-end прототип агентной веб-автоматизации:

1. **Planner (LLM)**: строит JSON-план из цели пользователя.
2. **Validator**: проверяет безопасность/корректность плана.
3. **Executor (Playwright)**: исполняет шаги на странице.
4. **Verifier (LLM)**: оценивает, достигнута ли цель.

## Быстрый запуск

### 1) Установить зависимости

```bash
pip install -r requirements.txt
playwright install
```

### 2) Настроить переменные окружения

```bash
cp .env.example .env
# затем заполнить GEMINI_API_KEY в .env
```

Или экспортом:

```bash
export GEMINI_API_KEY="..."
```

### 3) Запустить MVP

```bash
python -m app.main
```

По умолчанию цель:

- открыть `https://www.wikipedia.org`
- извлечь `h1`
- сделать скриншот
- завершить сценарий

### 4) Запуск в dummy-режиме (без Gemini)

```bash
python -m app.main --dummy
```

### 5) Что появляется в artifacts

После каждого запуска сохраняются:
- `artifacts/results/plan_<timestamp>.json`
- `artifacts/results/execution_<timestamp>.json`
- `artifacts/results/verdict_<timestamp>.json`
- `artifacts/logs/logs_<timestamp>.json`

Скриншот (`artifacts/screenshots/...`) появляется только если шаг `screenshot` реально выполнен.
Если выполнение упало раньше (например, `open_url` с DNS/сеть ошибкой), скриншота не будет.

## Пример своей цели

```bash
python -m app.main --goal "Open https://www.wikipedia.org, extract the h1 text, take screenshot and finish."
```

## Smoke test

```bash
pytest -q
```

> Тест не вызывает Gemini API и проверяет совместимость `planner -> validator -> verifier` на `DummyLLMClient`.
