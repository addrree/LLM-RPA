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

- открыть `https://example.com`
- извлечь `h1`
- сделать скриншот
- завершить сценарий

### 4) Запуск в dummy-режиме (без Gemini)

```bash
python -m app.main --dummy
```

## Пример своей цели

```bash
python -m app.main --goal "Open https://example.com, extract the h1 text, take screenshot and finish."
```

## Smoke test

```bash
pytest -q
```

> Тест не вызывает Gemini API и проверяет совместимость `planner -> validator -> verifier` на `DummyLLMClient`.
