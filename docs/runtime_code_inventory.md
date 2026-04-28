# Runtime/Test/Unused Inventory (2026-04-28)

## Method

- Collected repository files via `rg --files`.
- Classified by import/use from CLI entrypoints (`app/main.py`) and benchmark workflow modules.
- Cross-checked test-only dependencies from `tests/` imports.

## Runtime path (actively used in app/benchmark execution)

- `app/main.py`
- `app/config.py`
- `app/orchestrator/workflow_manager.py`
- `app/orchestrator/persistence.py`
- `app/planner/*`
- `app/validator/plan_validator.py`
- `app/executor/playwright_executor.py`
- `app/executor/action_handlers.py`
- `app/verifier/llm_verifier.py`
- `app/benchmark/*`
- `app/exporters/*`
- `app/schemas/*`
- `benchmarks/scenarios/*.json` (when selected in CLI)

## Test-only / primarily test-support modules

- `tests/*`
- `tests/fixtures/*`
- `app/utils/llm_client.py::DummyLLMClient` is used both by tests and runtime `--dummy` backend, so **kept in runtime module**.

## Not used / cleanup candidates

- No standalone temporary scratch files found in repository root scope during this pass.
- No removable dead runtime modules were identified with high confidence without risking benchmark behavior.

## Cleanup action taken

- Removed **none** of core runtime handlers/planner/validator/executor/verifier.
- Preserved legacy scenario files as compatibility aliases.
