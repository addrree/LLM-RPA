# WebArena / WebArena-Verified Integration Plan (Design Only)

## Scope guard

This document is intentionally **design-only**. No changes to the current benchmark runner pipeline are required for this phase.

## 1) Mapping WebArena task -> `user_goal`

Each WebArena task can be mapped into a canonical `user_goal` string with preserved constraints:

1. `intent` / `task_description` -> plain-language goal sentence.
2. `start_url` / `start_state` -> append explicit open-first instruction.
3. Constraints (allowed domains, disallowed actions, stop condition) -> appended as explicit constraints.
4. Success rubric -> converted into verifier expectations (without leaking answer literals into planner prompt).

Template:

- `Task: <intent>.`
- `Open this URL first: <start_url>.`
- `Stay within domains: <...>.`
- `Stop when: <observable success condition>.`

## 2) Mapping our `TaskSpec`/actions to WebArena-style tasks

Proposed adapter layer (`WebArenaTaskAdapter`) should translate tasks into current action space:

- Navigation:
  - WebArena click/nav instruction -> `click`, `wait_for`, `observe_page`
- Information retrieval:
  - extraction task -> `extract_pattern_from_page_text`, `extract_structured_items`, `extract_value_near_anchor`
- Simple search/form:
  - query input -> `input_text`
  - submit -> `press_key` or `click`

Adapter output remains standard `TaskSpec`; planner/validator/executor/verifier stay unchanged.

## 3) First task classes to onboard

1. Navigation tasks (single-hop and short multi-hop).
2. Information retrieval tasks (single value, repeated listing, section compare-lite).
3. Simple form/search tasks (public search forms without auth).

## 4) Task classes to exclude initially

Until dedicated safeguards are added, exclude:

- Purchase / checkout flows.
- Login/session-dependent tasks.
- Destructive state changes (delete/update/publish/send).
- Any task requiring private credentials.

## 5) Metrics to compare

For A/B against current core suites, compare:

- Execution success (`execution_status=success`).
- Verifier accept/reject behavior.
- Negative expected reject rate where applicable.
- Plan validation pass rate.
- Correction usage/recovery.
- Export success rate.
- Runtime diagnostics: planning/execution/verification/correction.

## 6) Artifact persistence plan

Keep existing artifact format; add WebArena metadata in benchmark context only:

- `webarena_task_id`
- `webarena_split`
- `webarena_site`
- `adapter_version`

Persist with existing paths:

- `artifacts/results/plan_*.json`
- `artifacts/results/execution_*.json`
- `artifacts/results/verdict_*.json`
- `artifacts/benchmarks/benchmark_summary_*.json`

## 7) Rollout sequence

1. Add dataset loader + adapter (offline).
2. Add optional suite generator that emits local scenario JSON.
3. Run in isolated experimental suite file (not default runner).
4. Validate metrics parity and artifact quality.
5. Only then discuss gradual promotion into broader benchmark portfolio.
