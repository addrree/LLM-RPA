# Live Content Evaluation Matrix

Controlled live evaluation for `python -m app.main --backend ollama_cloud --two-stage-planning --goal "<USER_GOAL>" --export-format json`.

Scope rules followed in this pass:

- `scripts/run_real_web_skill_suite.py` was not run.
- Full pytest, MiniWoB, and BrowserGym were not run.
- `app/browsergym_integration/*` was not changed.
- Screenshots, traces, and videos were not opened.
- Result inspection used only `artifacts/results/execution_*.json`; verifier verdicts below are from the CLI workflow summary printed by the same run.
- Main content fields are separated from auxiliary provenance/debug fields. URL/title/page snapshot alone is not counted as content success.

## Batch A

| ID | user_goal | task_type | site | status | artifact | main_content_fields | auxiliary_fields | verifier | full_vocabulary_was_used | what it proves | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Открой https://www.wikipedia.org и извлеки число статей рядом с языком Deutsch. | direct_value_extraction | wikipedia.org | content_success | `artifacts/results/execution_20260604_233842.json` | `article_count="3.125.000+ "` | none in main result | accept / 0.99 | false | Generic anchor-near-value extraction can return a real content scalar from a live page. | The run produced `article_count` only; `language_name` was implicit in the goal and not materialized as a separate field. |
| B1 | Открой https://developer.mozilla.org/en-US/docs/Web/API/WebSocket и выгрузи краткое описание WebSocket API: первое содержательное предложение или абзац описания. | direct_value_extraction | developer.mozilla.org | content_failure | `artifacts/results/execution_20260604_234942.json` | requested description field was populated as `"100"` | `page_snapshot` | reject / 0.20 | false | Failure shows description goals can be misrouted into scalar anchor-value extraction. | Generic planner/router issue: description/paragraph request was routed as `direct_value_extraction`; `extract_value_near_anchor` returned an unrelated numeric value. Corrective replanner also hit backend `prompt too long`. |
| C1 | Открой https://developer.mozilla.org/en-US/search?q=websocket, открой первый релевантный результат и выгрузи краткое описание найденной технологии: название и первое предложение описания. | n/a | developer.mozilla.org | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Failure exercises search-result + content field planning before execution. | Generic plan/output normalization issue: validator stopped before execution because required field `название` was not produced by any step. |
| D1 | Открой https://www.python.org, перейди по ссылке Documentation и выгрузи краткое описание раздела документации Python: заголовок раздела и первое содержательное предложение. | n/a | python.org | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Failure exercises semantic navigation + content field planning before execution. | Generic plan/output normalization issue: validator stopped before execution because required field `заголовок_раздела` was not produced by any step. |
| E1 | Открой https://www.w3schools.com/html/html_tables.asp и найди строку таблицы, где Company содержит Alfreds. Верни Company, Contact и Country. | single_entity_metadata | w3schools.com | pre_execution_failure | no execution artifact | none | none | n/a | false in validator diagnostic | Failure exercises table-row routing before execution. | Generic router/planner profile issue: table-row goal was routed as `single_entity_metadata`, and a `card_items` intent was rejected by that restricted profile. |
| F1 | Открой https://docs.python.org/3/library/index.html и выгрузи несколько модулей стандартной библиотеки Python: название модуля и краткое описание, если оно видно. | single_entity_metadata | docs.python.org | content_failure | `artifacts/results/execution_20260604_235337.json` | `extracted_fields.modules="next \|"` | none | reject / 0.20 | false | Failure exercises repeated-list extraction on a documentation index. | Generic repeated-items routing/extraction issue: list goal was routed as metadata, then field extraction selected navigation text instead of module rows. Corrective replanner hit backend `prompt too long`. |
| G1 | Открой https://www.python.org/success-stories/ и выгрузи несколько карточек историй использования Python: название истории и краткое описание. | catalog_or_card_extraction | python.org | content_success_with_limitation | `artifacts/results/execution_20260604_235421.json` | `cards[{title/name, description/snippet, href}]` including "Using Python to build a solution for instant tokenized real estate redemptions" and "Zama Concrete ML: Simplifying Homomorphic Encryption for Python Machine Learning" | `page_snapshot`, `href/link` as provenance | accept / 0.95 | false | Generic catalog/card extraction can return repeated live content cards. | Card segmentation is imperfect: the first and one later item aggregate a section/container instead of a single card. |
| H1 | Открой https://www.w3schools.com/howto/howto_js_todolist.asp и удали строку с текстом Pay bills. После действия выгрузи оставшиеся элементы списка. | structured_table_extraction | w3schools.com | content_failure | `artifacts/results/execution_20260604_235548.json` | `deleted_row` points to a large tutorial container; no `remaining_items` | `page_snapshot` | reject / 0.80 | false | Failure exercises row/item action on a list-like page. | Generic routing/candidate-narrowing issue: action targeted a broad content container containing "Pay bills" instead of the specific todo `li`; remaining list items were not extracted. Corrective replanner hit backend `prompt too long`. |
| I1 | Открой https://www.wikipedia.org и визуально посчитай крупные языковые блоки в центральной части страницы и верни количество и названия видимых языков. | visual_or_spatial_task | wikipedia.org | controlled_failure | `artifacts/results/execution_20260604_235639.json` | none; URL-keyed value is a page observation payload, not `count`/`visible_language_names` | observation payload contains `url`, `title`, `screenshot_path`, `page_text_excerpt`, DOM lists | reject / 0.00 | false | Failure exercises visual/DOM geometry counting without reading screenshots. | Generic visual controller limitation: execution reported `visual_spatial_no_decision` because object count was not exposed by reliable SVG/DOM geometry. |
| J1 | Открой официальный сайт СПбГУ и выгрузи общие контактные данные университета: адрес, телефон, email при наличии и название раздела или страницы, где эти данные опубликованы. | single_entity_metadata | spbu.ru | content_success | `artifacts/results/execution_20260605_000057.json` | `адрес`, `телефон`, `email`, `название_раздела_или_страницы`, `где_эти_данные_опубликованы` | none in main result | accept / 0.95 | false | Generic field-schema/semantic-region extraction can find live contact fields without a site-specific branch. | Routed as `single_entity_metadata` rather than a distinct semantic-region type, but extracted the requested content. Corrective replanner hit backend `prompt too long` after verifier retry flow. |

## Batch B

| ID | user_goal | task_type | site | status | artifact | main_content_fields | auxiliary_fields | verifier | full_vocabulary_was_used | what it proves | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 | Открой https://www.wikipedia.org и извлеки название языка Français и число статей рядом с этим языком. Верни language_name и article_count. | single_entity_metadata | wikipedia.org | content_failure | `artifacts/results/execution_20260605_070833.json` | none; `extracted_data.status="not_found"` with `article_count` missing | `page_snapshot` | reject / 0.90 | false | Second wording shows the scalar anchor-value path does not generalize to mixed object output. | Generic router/profile issue: explicit `language_name + article_count` shifted the task into `single_entity_metadata`, and field-schema extraction did not use the visible language block as an anchor-value object. Corrective validation also rejected plans missing `anchor_text` or `language_name`. |
| G2 | Открой https://www.python.org/success-stories/ и выгрузи карточки историй, в названии которых есть Arm: название и краткое описание. | n/a | python.org | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Second wording exercises card filtering by title condition. | Generic plan normalization/validator issue: validation stopped before execution because `extract_items` missed required `limit` and `container_selector` args. |
| J2 | Открой официальный сайт СПбГУ и найди сведения для связи с университетом: почтовый адрес, телефон и электронную почту при наличии. | n/a | spbu.ru | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Second wording exercises contact field-schema aliases without the phrase "контактные данные". | Generic plan/output normalization issue: validator stopped before execution because required field `телефон` was not produced by any step. |

## Stopped Types

No task type has yet been stopped by the "two identical failures in a row" rule during this pass. Batch A produced single failures for description, search-result content, semantic navigation, table row, repeated items, row action, and visual counting.

## Generic Issues Observed

| area | evidence | generic diagnosis |
|---|---|---|
| Router/profile selection | B1, E1, F1, H1 | Several content/list/table/action requests were routed into an incompatible restricted profile, causing either wrong extraction intent or validator rejection. |
| Plan/output field normalization | C1, D1, J2 | Required Russian field names were not tied to fields produced by extraction steps, so validation failed before execution and no execution artifact was saved. |
| Corrective replanner prompt sizing | B1, F1, H1, J1 | Corrective replanning sent very large observation/context payloads and hit Ollama Cloud `prompt too long` errors. This is backend/context-limit pressure, not a 429/session limit. |
| Repeated item extraction and item plan defaults | F1, G1 limitation, G2 | Repeated content extraction works for cards but can over-select container blocks; documentation module rows can be routed as metadata; item extraction plans can fail validation when missing generic defaults such as `limit`/`container_selector`. |
| Row/item action candidate narrowing | H1 | Candidate selection can choose a broad ancestor region when the target text appears inside tutorial/source-code content. |
| Visual/DOM geometry extraction | I1 | Visual count profile can stop as controlled failure when the requested objects are not represented by a reliable generic geometry counter. |

## Post-Fix Batch A

Rerun date: 2026-06-05. Same `app.main` command shape, one live run per major type. No screenshots, traces, or videos were opened; artifact inspection below used only `artifacts/results/execution_*.json`. CLI verifier summaries are recorded from the run output.

| ID | user_goal | task_type | site | status | artifact | main_content_fields | auxiliary_fields | verifier | full_vocabulary_was_used | what it proves | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PF-A1 | Wikipedia Deutsch article count. | direct_value_extraction | wikipedia.org | content_success_with_limitation | `artifacts/results/execution_20260605_091850.json` | `www_wikipedia_org="3.125.000+ "` | `page_snapshot` | accept / 0.95 | false | Scalar anchor-near-value extraction still returns the live article count after fixes. | Output key is not normalized to requested `article_count`; content is correct but schema shape is weak. |
| PF-B1 | MDN WebSocket short description. | direct_value_extraction | developer.mozilla.org | content_failure | `artifacts/results/execution_20260605_092048.json` | none | `page_snapshot` | reject / 0.00 | false | Corrective prompt compaction avoided the previous `prompt too long` class in this run. | Description extraction still failed generically: `h1 + p` locator timed out and only the snapshot remained. |
| PF-C1 | MDN search first result title + description. | n/a | developer.mozilla.org | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Field-normalization failure changed shape after fixes, so the original missing-field bug moved forward. | Generic planner/validator gap: `wait_for` with bare text is rejected as too weak without selector, exact match, or URL condition. |
| PF-D1 | python.org Documentation section heading + first sentence. | repeated_items_extraction | python.org | content_failure | `artifacts/results/execution_20260605_092357.json` | `extracted_items` contained documentation/sidebar links such as `Python Periodicals`, `Python Packaging User Guide`, `Browse Python 3.14.5 Documentation` | none | reject / 0.80 | false | Navigation scenario now reaches execution instead of failing required-field validation. | Generic semantic navigation/extraction gap: it extracted link/list items, not the requested page/section summary. Corrective plans still emitted malformed `extract_items`. |
| PF-E1 | W3Schools table row where Company contains Alfreds. | single_entity_metadata | w3schools.com | content_failure | `artifacts/results/execution_20260605_092728.json` | `result_object` fields all contained `Centro comercial Moctezuma / Francisco Chang / Mexico` | `page_snapshot` | reject / 0.90 | false | Table-row scenario now reaches execution instead of restricted-profile validation failure. | Generic row-condition gap: the Alfreds condition was not applied; wrong table row was returned and field splitting collapsed whole rows into every field. |
| PF-F1 | Python docs standard-library modules: name + short description. | single_entity_metadata | docs.python.org | content_success_with_limitation | `artifacts/results/execution_20260605_093004.json` | `modules` and `module_names` include standard-library TOC entries such as `Built-in Functions`, `datetime`, `math`, `collections` with partial descriptions | none | accept / 0.85 | false | Repeated/list extraction improved from `next |` to a real documentation TOC payload. | Still mixes navigation/category nodes with module rows; descriptions are incomplete or category-derived. |
| PF-G1 | python.org success-story cards: title + description. | catalog_or_card_extraction | python.org | content_success_with_limitation | `artifacts/results/execution_20260605_093041.json` | `cards[{title, description, href}]` including real success stories | `page_snapshot`, `href/link` provenance | accept / 0.95 | false | Generic card/catalog extraction remains a stable passing type. | Segmentation still over-selects some section/category containers as cards. |
| PF-H1 | W3Schools todo: delete Pay bills, then list remaining items. | n/a | w3schools.com | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Action scenario no longer produced the broad-container runtime deletion artifact. | Generic action-plan normalization gap: validator rejected `find_row_by_condition` because `condition` was missing. |
| PF-I1 | Wikipedia visual count of central language blocks. | n/a | wikipedia.org | pre_execution_failure | no execution artifact | none | none | n/a | n/a | Visual scenario did not consume visual artifacts. | Generic controlled-skip regression: validator rejected `visual_extract_object_count` because `object/shape/target` was missing, so it did not reach the prior controlled runtime classification. |
| PF-J1 | SPbU common contacts: address, phone, email, source page. | single_entity_metadata | spbu.ru | content_success | `artifacts/results/execution_20260605_093518.json` | `contact_data.address`, `contact_data.phone`, `contact_data.email`, source page fields | `contacts_snapshot` | accept / 0.95 | false | Generic field-schema/semantic navigation can extract real contact content on a live Cyrillic site after fixes. | Still routed as metadata, but content/schema are good enough for this goal. |

Post-fix Batch A summary:

- Clear content successes: PF-A1, PF-G1, PF-J1.
- Improved but limited: PF-F1.
- Reached execution but content wrong: PF-D1, PF-E1.
- Still blocked before execution by generic validation/planning gaps: PF-C1, PF-H1, PF-I1.
- Description extraction remains a runtime content failure: PF-B1.
- No Ollama 429/session-limit occurred in this batch.

## Post-Fix Batch B

Second wording / second-shape checks for types that passed or improved in Post-Fix Batch A.

| ID | user_goal | task_type | site | status | artifact | main_content_fields | auxiliary_fields | verifier | full_vocabulary_was_used | what it proves | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PF-A2 | Wikipedia Francais language block: return `language_name` and `article_count`. | single_entity_metadata | wikipedia.org | content_failure_despite_verifier_accept | `artifacts/results/execution_20260605_094030.json` | `metadata.article_count="2 103 000+"`; language-name field contained an article-count-like value instead of `Francais`; `www_wikipedia_org` was the URL | `page_snapshot` | accept / 0.95 | false | The pipeline can run mixed scalar/object plans after fixes. | Generic schema/anchor-object gap remains: requested `language_name` was not materialized, counts appear mismatched, and verifier accepted an incorrectly shaped result. |
| PF-G2 | python.org success-story cards whose title contains Arm: title + description. | catalog_or_card_extraction | python.org | content_failure_despite_verifier_accept | `artifacts/results/execution_20260605_094213.json` | `cards[*].title`, `description`, and `href` were `null`; `raw_text` contained aggregated sections including `Python on Arm: 2025 Update` | `page_snapshot`, aggregate `raw_text` | accept / 0.95 | false | Card extraction reaches execution and sees the relevant page region. | Generic condition/filter + card-field projection gap: the requested structured fields were not populated, and matching content remained inside broad container text. |
| PF-J2 | SPbU second wording: postal address, phone, email. | single_entity_metadata | spbu.ru | content_failure | `artifacts/results/execution_20260605_094406.json` | none | none | reject / 0.00 | false | Second wording still exercises live navigation resilience. | Execution JSON reports generic browser navigation failure: `Page.goto` timed out while navigating to `https://spbu.ru/`. No content artifact was produced. No 429/session-limit occurred. |

Post-fix Batch B summary:

- No clean second-wording success was proven in this batch.
- Two runs show verifier false positives for structured content shape: PF-A2 and PF-G2.
- PF-J2 failed before extraction due to navigation timeout recorded in `execution_20260605_094406.json`.
- No Ollama 429/session-limit occurred; live runs were stopped voluntarily after Batch B to avoid burning backend capacity without new signal.

## Post-Fix Generic Diagnosis

| area | evidence | generic diagnosis | next generic fix direction |
|---|---|---|---|
| Verifier normalization / required field shape | PF-A2, PF-G2 | Verifier can accept results where requested fields are missing, null, or filled with values from the wrong semantic slot. | Normalize verifier checks against requested schema: required field present, non-null, semantically typed, and not only aggregate `raw_text`. |
| Mixed anchor-value object extraction | PF-A1 vs PF-A2 | Simple scalar extraction works, but `language_name + article_count` object shape collapses or mismatches fields. | Generic anchor-object extractor: represent nearby label/value pairs as one row/object, then map requested fields by semantic role. |
| Description / paragraph extraction | PF-B1 | Description goals still route/execute through fragile locator or scalar paths. | Treat description/first-paragraph as a generic text-block extraction intent with fallback to readable article/section summaries. |
| Search/navigation waits | PF-C1 | Planner can produce bare-text `wait_for` that validator rejects. | Normalize weak waits into URL/title/selector/exact conditions or controlled skip before validation. |
| Semantic navigation + section summary | PF-D1 | Scenario reaches execution but extracts sidebar/list links instead of the target page summary. | After navigation, prefer page/section heading + leading paragraph extraction over repeated item extraction unless the requested output is a list. |
| Table rows and field splitting | PF-E1 | Row condition not applied; whole row text copied into each requested field. | Generic table-row extraction must bind header cells, apply row predicates, and split cells by column schema. |
| Repeated list extraction | PF-F1 | Improved payload, still mixed categories/navigation with rows. | Rank repeated candidates by requested item semantics and suppress global nav/category containers. |
| Card filtering / projection | PF-G1, PF-G2 | Cards are found, but filtered card fields can become aggregate raw text or null projection. | Apply conditions after card segmentation and require projected title/description fields per card. |
| Row/list action planning | PF-H1 | Action validation can fail because condition is omitted. | Normalize action targets into a generic condition object before validation; otherwise controlled-skip with reason. |
| Visual controlled skip | PF-I1 | Visual task now fails validation before controlled runtime classification. | Infer generic visual target from goal or classify as controlled skip before emitting invalid visual action. |
| Backend/navigation resilience | PF-J2 | Live navigation timeout can block otherwise passing field-schema tasks. | Add bounded retry/backoff and explicit external/navigation failure classification in execution artifacts. |

## Second-Fix Live Smoke

After adding generic weak-wait normalization, visual target normalization, and verifier projection guards, three targeted live smokes were run. No 429/session-limit occurred.

| ID | user_goal | task_type | site | status | artifact | main_content_fields | auxiliary_fields | verifier | full_vocabulary_was_used | what it proves | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SF-C1 | MDN search first result title + description. | search_results_extraction | developer.mozilla.org | runtime_failure_after_validation_fix | `artifacts/results/execution_20260605_095752.json` | none | `search_snapshot` | reject / 0.00 | false | Weak bare-text `wait_for` no longer stops execution before artifact creation. | Generic search navigation gap remains: execution failed with `target_link_not_found` for `first result`. |
| SF-I1 | Wikipedia visual count of central language blocks. | visual_or_spatial_task | wikipedia.org | controlled_failure | `artifacts/results/execution_20260605_095852.json` | none | none | reject / 0.00 | false | Visual task returned to controlled runtime classification instead of pre-execution validator failure. | Generic visual controller limitation remains: `visual_spatial_no_decision`, object count not exposed by reliable SVG/DOM geometry. |
| SF-G2 | python.org Arm story cards: title + description. | catalog_or_card_extraction | python.org | content_failure_correctly_rejected | `artifacts/results/execution_20260605_100000.json` | `arm_cards` list contained `raw_text`, but `title`, `description`, and `href` were `null` | `page_snapshot`, aggregate `raw_text` | reject / 0.00 | false | Verifier guard now prevents the previous false accept for raw-only card collections. | Generic card segmentation/projection and condition filtering remain unresolved. |

Second-fix code impact:

- `wait_for(text=...)` without selector/url/scope is normalized to a generic heading selector before validation.
- `visual_extract_object_count` without target gets a neutral target from output shape so runtime can produce controlled failure.
- Verifier rejects raw-only collection payloads and count-like values in name/title fields before LLM accept can mask them.

## Generic Skills Fix Pass, 2026-06-05

Scope:

- Runtime code audit after this pass found no `wikipedia`, `python.org`, `w3schools`, `developer.mozilla`, `mdn`, `spbu`, `spbstu`, or `spbgu` strings under `app/`.
- `task_router.py` audit found no Cyrillic text and no Russian structural cue block.
- Profiles stayed restricted; every profile still reports `full_vocabulary_was_used=false`.
- Full pytest, MiniWoB, BrowserGym, and `scripts/run_real_web_skill_suite.py` were not run.
- Screenshots, traces, and videos were not opened. Live inspection used only `artifacts/results/execution_*.json`.

Generic changes covered by unit tests:

- `text_block` / page-summary extraction now has a source-text fallback for main readable prose when DOM paragraph selection is empty.
- `field_schema` can project table-like rows from page text when the user goal supplies a generic `FIELD contains VALUE` row condition.
- Search-result navigation uses generic `search_results` and does not rely on literal link text such as "first result".
- Row/list action repair normalizes missing conditions to `{field:null, operator:"contains", value:"..."}` and rejects broad ancestor-only row matches instead of clicking them.
- Anchor-object extraction maps label fields and count fields separately.
- Verifier rejects URL/title/page_snapshot-only content, raw-only card collections, and semantically swapped name/count values.

Targeted validation:

- `python -m py_compile` passed for changed app modules.
- Targeted pytest passed: `99 passed, 1 warning` for semantic region, cards, router profiles, two-stage workflow, table rows, replanner compaction, verifier payload/shape, text-block, search-result, row-action, anchor-object, and runtime fallback tests.
- The warning was pytest cache write denial under `.pytest_cache`; it did not affect test outcomes.

## Third-Fix Live Smoke

First full smoke after this fix pass:

| ID | user_goal | site | status | artifact | main_content_fields | auxiliary_fields | verifier | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|
| LF-1 | MDN WebSocket short description. | developer.mozilla.org | controlled_failure | `artifacts/results/execution_20260605_182100.json` | none | none | reject / 0.00 | `failure_stage=description_not_found`; generic text-block extraction still failed before source-text fallback was added. |
| LF-2 | W3Schools table row where Company contains Alfreds. | w3schools.com | content_failure | `artifacts/results/execution_20260605_182445.json` | wrong row: `Centro comercial Moctezuma / Francisco Chang / Mexico`; whole row copied into multiple fields | `page_snapshot` | reject / 0.00 | Field-schema path did not apply the table row condition or header projection. |
| LF-3 | MDN search first relevant result, title + first description sentence. | developer.mozilla.org | content_failure | `artifacts/results/execution_20260605_182726.json` | opened `/WebSockets_API`; extracted only title field `название` | `result_page` snapshot | reject / 0.00 | Search-result navigation improved, but description extraction failed with `description_not_found`. |
| LF-4 | Python success-story cards whose title contains Arm. | python.org | content_failure | `artifacts/results/execution_20260605_183212.json` | scalar-ish `название="2025 Update"`, `описание="2025 Update"` | `page_snapshot` | uncertain / 0.00 | Plan used field-schema style extraction, not a card list; no structured card projection. |
| LF-5 | W3Schools todo delete Pay bills, then remaining items. | w3schools.com | content_failure | `artifacts/results/execution_20260605_183539.json` | `row_ref` was a broad tutorial container; `remaining_items=[]` | none | uncertain / 0.00 | Row condition/action now executes, but candidate narrowing still selected a broad ancestor before the broad-match guard was added. |
| LF-6 | Wikipedia Francais language name + article count. | wikipedia.org | content_failure | `artifacts/results/execution_20260605_183914.json` | `article_count="2 761 000+"`; no valid `language_name` field | text-block object with full language list | uncertain / 0.00 | Mixed anchor-object shape still not selected live; verifier did not mark it a clean success. |
| LF-7 | Wikipedia visual count of central language blocks. | wikipedia.org | runtime_failure_after_validation_fix | `artifacts/results/execution_20260605_184554.json` | none | none | reject / 0.00 | Navigation timed out at `open_url`; `failure_stage=browser_operation_failed`, not a validation crash and not a 429/session limit. |

Batch C reruns after the source-text, table-text, and broad-row generic fixes:

| ID | user_goal | site | status | artifact | main_content_fields | auxiliary_fields | verifier | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|
| LF-1b | MDN WebSocket short description. | developer.mozilla.org | content_success_with_verifier_issue | `artifacts/results/execution_20260605_185519.json` | `full_text.title="WebSocket"`; `full_text.description="The WebSocket object provides the API for creating and managing a WebSocket connection..."`; description sentence also saved under requested Cyrillic output key | `page_snapshot` | reject / 0.00 | Content extraction succeeded; verifier/schema alignment still rejected because produced keys did not exactly match normalized expected fields. |
| LF-2b | W3Schools table row where Company contains Alfreds. | w3schools.com | pre_execution_failure | no new execution artifact | none | none | n/a | Validator stopped before execution: required field `Company` was not produced by any top-level step. Generic plan/output normalization remains open. |
| LF-3b | MDN search first relevant result, title + first description sentence. | developer.mozilla.org | controlled_failure | `artifacts/results/execution_20260605_185709.json` | none | `search_snapshot` only | reject / 0.00 | Search page snapshot did not expose result-like links; execution stopped with `failure_stage=target_link_not_found`. |
| LF-5b | W3Schools todo delete Pay bills, then remaining items. | w3schools.com | controlled_failure | `artifacts/results/execution_20260605_185853.json` | none | none | reject / 0.00 | Broad-container false success was removed. Execution now reports `failure_stage=row_not_found` when only broad ancestor/source-code text matches. |

No Ollama 429/session-limit occurred in the full smoke or Batch C reruns. Some runs recorded empty-content retries, connection resets, or timeouts; these were treated as backend instability but not as the configured 429/session-limit stop condition.

## Continuation After API Key Refresh, 2026-06-06

Scope:

- Continued controlled live evaluation through `python -m app.main --backend ollama_cloud --two-stage-planning --export-format json`.
- `scripts/run_real_web_skill_suite.py`, full pytest, MiniWoB, and BrowserGym were not run.
- Screenshots, traces, videos, raw LLM artifacts, plans, verdict JSON, and logs were not opened for diagnosis. Live inspection used only `artifacts/results/execution_*.json` plus CLI summary lines.
- No Ollama 429/session-limit occurred.
- Runtime code audit after this continuation found no `wikipedia`, `python.org`, `w3schools`, `developer.mozilla`, `mdn`, `spbu`, `spbstu`, or `spbgu` strings under `app/`.
- `app/planner/task_router.py` audit found no Cyrillic text and no Russian structural cue block.

Generic fixes added in this continuation:

- Planner/Replanner now map nested structural fields from `extract_by_intent(fields/columns/headers)` to their parent top-level artifact for validator `required_fields`.
- Replanner re-normalizes `required_fields` after `coalesce_field_schema_steps`, fixing the `Company`/`Contact`/`Country` parent-artifact mismatch.
- Malformed `extract_items(save_as=remaining_items/list_items)` is normalized to generic `extract_by_intent(intent=card_items)` instead of failing validation.
- Row action condition normalization now handles phrases such as `close button for list item containing text Pay bills` and `close button for list item Pay bills`.
- Row candidate discovery now includes accessible frames and preserves `frame_index`/`frame_url`; row action can click inside the referenced frame.
- Row action click uses a guarded actionability fallback: normal click, then force click, then DOM click only for viewport/actionability/timeout failures.
- Non-unique row selectors are refined with `row_ref.text` before searching for the action control.

Targeted validation:

- `python -m py_compile` passed for changed app modules.
- Targeted pytest passed: `124 passed, 1 warning` across table rows, row actions, two-stage workflow, search result navigation, text-block extraction, verifier shape/payload checks, router profiles, semantic region, cards, replanner prompt compaction, anchor-object extraction, and action-handler fallbacks.
- The warning was pytest cache write denial under `.pytest_cache`; it did not affect test outcomes.

Live reruns:

| ID | user_goal | site | status | artifact | main_content_fields | auxiliary_fields | verifier | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|
| LF-2c | W3Schools table row where Company contains Alfreds. | w3schools.com | content_success | `artifacts/results/execution_20260605_220202.json` | `extracted_fields.company="Alfreds Futterkiste"`, `contact="Maria Anders"`, `country="Germany"` plus original header keys | `page_snapshot` text excerpt only | accept / 0.95 | Table row now reaches execution and returns the correct row. Corrective retry was used once. |
| LF-5c | W3Schools todo delete Pay bills, then remaining items. | w3schools.com | controlled_failure | `artifacts/results/execution_20260605_220305.json` | none | none | reject / 0.00 | Condition normalized to `Pay bills`, but candidate collection did not see the target row yet: `failure_stage=row_not_found`, `rows_checked=14`. |
| LF-5d | Same row-action task after frame candidate support. | w3schools.com | pre_execution_failure | no execution artifact | none | none | n/a | Validator rejected malformed `extract_items` without fields. Fixed generically with `remaining_items/list_items` output-key normalization. |
| LF-5e | Same row-action task after malformed-list normalization. | w3schools.com | runtime_failure_after_row_found | `artifacts/results/execution_20260605_221100.json` | only `page_snapshot` | row/control were found before click failed | reject / 0.00 | Generic actionability issue: close control resolved but Playwright reported it outside viewport. Fixed with guarded force/DOM click fallback. |
| LF-5f | Same row-action task after click fallback. | w3schools.com | controlled_failure | `artifacts/results/execution_20260605_221341.json` | none | none | reject / 0.00 | Row was found as `ul > li.checked:nth-of-type(2)`, but action control was not found, likely because selector was non-unique/ambiguous. Added row-text locator refinement. Corrective plan was also rejected by validator for malformed `wait_for`. |
| LF-5g | Same row-action task after row-text refinement. | w3schools.com | controlled_failure | `artifacts/results/execution_20260605_221557.json` | none | none | reject / 0.00 | LLM emitted a new condition phrase `close button for list item Pay bills`; runtime treated it as a literal row term. Added generic normalization for this variant, but stopped this task type without another live retry to avoid burning backend capacity. |
| LF-1c | MDN WebSocket short description. | developer.mozilla.org | content_success | `artifacts/results/execution_20260605_221844.json` | title `WebSocket`; description `The WebSocket object provides the API for creating and managing a WebSocket connection...` | none | accept / 0.95 | Description/text-block path now succeeds cleanly; previous verifier/schema mismatch is gone. |
| LF-3c | MDN search first relevant result, title + first description sentence. | developer.mozilla.org | content_success_with_limitation | `artifacts/results/execution_20260605_221955.json` | `result` fields populated; opened `WebSocket()` constructor page | `tech_page` snapshot includes full page context | accept / 0.95 | Search/navigation path now succeeds, but specificity is imperfect: it opened the constructor result rather than the broader WebSocket overview, and the extracted description field is thin. |

Continuation conclusion:

- Passed live after fixes: `text_block/description`, `table row field extraction`, and `search-result navigation with page extraction`.
- Still unresolved live: `row_or_item_action` on embedded todo-style examples. The generic pipeline progressed from broad false success to precise controlled failures and added frame/actionability/selector refinements, but the task type still needs another pass before it can be marked generalized.

## Fourth-Fix Continuation, 2026-06-06

Scope:

- Continued from the live-evaluation/fix loop using the same `app.main` command shape.
- Screenshots, traces, videos, raw LLM artifacts, plans, verdict JSON, and standalone log JSON were not opened. Diagnosis used only `artifacts/results/execution_*.json` plus CLI summaries.
- Live runs were stopped when Ollama Cloud returned `status_code=429` weekly usage limit.

Generic fixes added:

- Collection/card/list repair now canonicalizes user goals with real Cyrillic and mojibake variants for structural cues, conditions, and requested item fields.
- Broad semantic clicks such as bare `link` are removed from collection plans and guarded at runtime with a controlled `target_too_broad` failure.
- Card extraction filters matching candidates before projecting requested fields, and result-like content links are merged into generic card candidates.
- Search-result fallback ranking now prefers ranked content/result links and can recover query context from the current URL.
- Anchor-object planning now detects generic `label + adjacent count` goals such as `language_name + article_count`, converting mistaken `card_items/table_rows/text_block/field_schema` plans to `extract_by_intent(intent=anchor_object)`.
- Anchor-object runtime extraction can use explicit anchors inside navigation regions and strips numeric count text out of label fields.
- Validator now treats `anchor_object.fields` as produced nested fields.
- Verifier now deterministically accepts well-shaped mixed label/count anchor objects and still rejects swapped name/count values.

Targeted validation:

- `python -m py_compile` passed for changed planner, executor, validator, and verifier modules.
- Focused pytest passed: `102 passed, 1 warning` across search-result navigation, row actions, action-handler fallbacks, runtime fallbacks, two-stage workflow, card runtime, anchor-object extraction, verifier shape, and validator nested-field checks.
- The warning was pytest cache write denial under `.pytest_cache`; it did not affect test outcomes.

Live reruns before backend limit:

| ID | user_goal | site | status | artifact | main_content_fields | auxiliary_fields | verifier | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|
| LF-1d | MDN WebSocket short description. | developer.mozilla.org | content_success | `artifacts/results/execution_20260605_232917.json` | WebSocket title/description content | page metadata | accept / 0.95 | Direct text-block/description path remained passing. |
| LF-2d | W3Schools table row where Company contains Alfreds. | w3schools.com | content_success | `artifacts/results/execution_20260605_235026.json` | Alfreds Futterkiste / Maria Anders / Germany | page snapshot | accept / 0.99 | Table-row condition/header extraction remained passing. |
| LF-3d | MDN search first relevant result, title + description. | developer.mozilla.org | content_success | `artifacts/results/execution_20260605_232757.json` | opened `https://developer.mozilla.org/en-US/docs/Web/API/WebSocket`; title and description populated | search/result provenance | accept / 0.95 | Search-result navigation generalized beyond previous constructor-page miss. |
| LF-5h | W3Schools todo delete Pay bills, then remaining items. | w3schools.com | content_success | `artifacts/results/execution_20260605_230830.json` | row action completed; remaining items extracted | row_action diagnostics | accept / 0.95 | Row/list action scenario moved from controlled failure to passing live. |
| LF-4d | Python success-story cards with title containing Arm. | python.org | content_success | `artifacts/results/execution_20260606_082007.json` | `arm_cards[0].название="Python on Arm: 2025 Update"` and description populated | selector provenance | accept / 0.95 | Card filtering/projection now works for a second-wording filtered-card goal without site-specific branching. |
| LF-6d | Wikipedia Français language + article count. | wikipedia.org | data_correct_backend_blocked | `artifacts/results/execution_20260606_084404.json` | `metadata.language_name="Français"`, `metadata.article_count="2 761 000+ articles"` | `www_wikipedia_org`, `page_snapshot` | uncertain / 0.00 | Execution data is correct, but live verdict remained uncertain during Ollama backend empty-content/429 pressure. Local verifier/validator fixes were added after this artifact; no further live retry was made because 429 weekly limit appeared. |

Stopped due to backend limit:

- Ollama Cloud returned `status_code=429` weekly usage limit during the LF-6 retry/corrective flow.
- Remaining scenarios were not run after that point:
  - LF-7 visual count of central Wikipedia language blocks.
  - PF-F1 / docs.python.org standard-library modules second pass after the latest repeated-list/card fixes.
  - Any additional Batch B/C second-site or problematic-type checks.

## API Refresh Completion, 2026-06-06

After the API key was refreshed, live runs resumed from the stopped point. The same artifact-inspection rule was followed: only `artifacts/results/execution_*.json` was opened for diagnostics; no screenshots, traces, videos, raw LLM, plan, verdict, or standalone log artifacts were opened.

Additional generic fixes:

- `field_schema` now recognizes visible object/list summary requests (`visible/visual/central/large blocks + count + names`) and converts visible link/object candidates into `{count, names, items}` instead of extracting the first scalar number from page text.
- `extract_visible_links` now preserves generic bbox/viewport metadata so central/large visible blocks can be ranked without reading screenshots.
- `card_items` internally collects a larger candidate window before projection and has a generic module-like ranker for goals requesting modules: `module.name — description` rows are split into title/name plus description and ranked ahead of category/container TOC nodes.
- Collection repair preserves a complete `field_schema` table-row extraction when it already contains the requested header fields, preventing table-row plans from being rewritten to a weaker parent key.
- Empty/advisory `wait_for` steps without selector/url/text are dropped before validator, preventing corrective plans from failing on no-op waits.
- `field_schema` now has generic address extraction for labeled address lines and postal/street-like address candidates.
- Phone extraction accepts Unicode dash characters so formatted phone numbers are not truncated.

Targeted validation:

- `python -m py_compile` passed for changed planner, executor, validator, and verifier modules.
- Focused pytest passed: `150 passed, 1 warning` across search-result navigation, row actions, action-handler fallbacks, runtime fallbacks, two-stage workflow, card runtime, anchor-object extraction, table rows, text-block, semantic region, verifier payload/shape, and router-profile tests.
- The warning was pytest cache write denial under `.pytest_cache`; it did not affect test outcomes.
- Runtime code audit found no `wikipedia`, `python.org`, `w3schools`, `developer.mozilla`, `mdn`, `spbu`, `spbstu`, or `spbgu` strings under `app/`.

Live reruns:

| ID | user_goal | site | status | artifact | main_content_fields | auxiliary_fields | verifier | limitation/failure_reason |
|---|---|---|---|---|---|---|---|---|
| LF-6e | Wikipedia Français language + article count. | wikipedia.org | content_success | `artifacts/results/execution_20260606_085621.json` | `metadata.language_name="Français"`, `metadata.article_count="2 761 000+ articles"` | none beyond `metadata.raw_item` provenance | accept / 0.93 | Clean retry after API refresh; no corrective retry. |
| LF-7e | Wikipedia visual count of central language blocks, before visible-summary fix. | wikipedia.org | content_failure | `artifacts/results/execution_20260606_085905.json` | `language_links` contained the central language links, but `extracted_result.верни_количество="2 103 000+"` was an article count, not block count | `page_snapshot`, URL provenance | reject / 0.30 | Generic shape issue: visible link list was available but final extraction used scalar field schema over page text. Fixed by visible object/list summary. |
| LF-7f | Same visual/count task after visible-summary fix. | wikipedia.org | content_success | `artifacts/results/execution_20260606_090423.json` | count/names derived from central visible language-link blocks | `language_links`, `page_snapshot` | accept / 0.93 | Visual/count structural type now passes without reading screenshot artifacts. |
| PF-F1e | Python docs modules before module-like rank split. | docs.python.org | content_success_with_limitation | `artifacts/results/execution_20260606_090547.json` | `modules` list populated, but first rows included categories/topics such as `Introduction`, `Built-in Types` | `page_snapshot` | accept / 0.95 | Verifier accepted, but execution JSON showed the repeated-list extraction still mixed category/container TOC nodes. Fixed by module-like ranking and `name — description` splitting. |
| PF-F1f | Python docs modules after module-like ranking. | docs.python.org | content_success | `artifacts/results/execution_20260606_091141.json` | `modules` starts with real module-like rows: `string`, `string.templatelib`, `re`, `difflib`, `textwrap`, `unicodedata`, `struct`, `datetime`, each with visible descriptions | `page_snapshot` | accept / 0.90 | Repeated/list extraction for documentation modules now passes cleanly. |
| J2a | SPbU contacts, second wording, no explicit URL. | inferred official site | timeout_no_artifact | no execution artifact | none | none | n/a | Initial no-URL run exceeded task timeout before saving execution JSON. |
| J2b | Same SPbU contact wording with explicit `https://spbu.ru`. | spbu.ru | content_success | `artifacts/results/execution_20260606_092103.json` | postal address, phone, email populated | `contacts_snapshot` | accept / 0.95 | Proved contact extraction path works when URL is explicit. |
| J2c | Same no-URL SPbU contact wording after empty-wait repair. | inferred official site | content_failure | `artifacts/results/execution_20260606_093541.json` | phone/email populated; postal address missing | none | reject / 0.90 | URL inference/navigation reached execution; generic address extraction was missing. |
| J2d | Same no-URL SPbU contact wording after address/phone fixes. | inferred official site | content_success | `artifacts/results/execution_20260606_094600.json` | `почтовый_адрес="Россия, 199034, Санкт-Петербург, Университетская наб., д. 7–9"`, `телефон="7-812-328-96-44"`, `электронную_почту="spbu@spbu.ru"` | `page_snapshot` | accept / 0.95 | Second wording and official-site inference now pass cleanly. |

Current conclusion after API refresh:

- The previously unresolved live scenarios now have clean passing runs: LF-6 mixed anchor object, LF-7 visible/visual count+names, PF-F1 docs module list, and J2 second-wording contact extraction.
- No new Ollama 429/session limit occurred during the resumed runs.
