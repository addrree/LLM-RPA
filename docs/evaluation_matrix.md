# Structural Task Type Evaluation Matrix

Manual live checks were run through `python -m app.main`. No
`scripts/run_real_web_skill_suite.py` run was used, and no `example.org` live
target was used in this pass. No new live checks were run during the final
generic-runtime cleanup; the live rows below describe existing artifacts.

## Confirmed live through app.main

| task_type | scenario | artifact path | status / verifier | extracted_data key | used_skills | full_vocabulary_was_used | limitations |
|---|---|---:|---|---|---|---|---|
| direct_value_extraction | Extract article count next to a visible anchor. | `artifacts/results/execution_20260602_144520.json` | success / accept | `english_article_count` | `observe_page`, `extract_value_near_anchor` | false | Earlier live confirmation; not re-run in the final pass. |
| single_entity_metadata | Extract page/object metadata from a public documentation page. | `artifacts/results/execution_20260602_152028.json` | success / historical reject; verifier fixed by targeted preprocessing test | `package_metadata`, `name`, `description`, `source_url` | `observe_page`, `extract_by_intent`, `package_metadata_extraction` | false | Historical artifact predates generic `field_schema` normalization and verifier nested-field normalization. Current runtime has no package-specific extractor. |
| semantic_navigation | Click a visible navigation target, then extract opened URL/title/links. | `artifacts/results/execution_20260602_152452.json` | success / accept | `final_url`, `page_title`, `top_links` | `observe_page`, `semantic_click`, `extract_by_intent`, `extract_visible_links` | false | Earlier live confirmation; not re-run in the final pass. |
| structured_table_extraction | Extract rows/columns from a visible HTML table. | `artifacts/results/execution_20260602_152558.json` | success / accept | `extracted_table` | `observe_page`, `extract_by_intent` | false | Earlier live confirmation; not re-run in the final pass. |
| catalog_or_card_extraction | Extract generic non-product cards with title/description/link. | `artifacts/results/execution_20260602_153105.json` | success / historical reject | `extracted_cards` | `observe_page`, `extract_by_intent`, `row_list_extraction` | false | Generic `card_items` extracted cards, but this page had weak/repeated descriptions. |
| catalog_or_card_extraction + condition_filtering | Extract cards whose title contains a requested term. | `artifacts/results/execution_20260602_154131.json` | success / accept | `extracted_cards` | `observe_page`, `extract_by_intent`, `row_list_extraction` | false | Confirms confident condition propagation into generic item filtering. |
| search_results_extraction | Multi-step search result: open search page, open first relevant result, extract title/description/current URL. | `artifacts/results/execution_20260602_170455.json` | success / accept | `search_results`, `clicked_text`, `name`, `description`, `final_url` | `observe_page`, `extract_by_intent`, `row_list_extraction`, `extract_visible_links`, `semantic_click`, `package_metadata_extraction` | false | Historical public-search confirmation. Current runtime uses only structural extraction/navigation paths and moves `final_url` extraction after metadata navigation. |
| row_or_item_action | Find a row/list-like item by text and perform an action. | `artifacts/results/execution_20260602_201231.json` | success / accept | `row_ref`, `row_action` | `observe_page`, `row_list_email_action` | false | Historical artifact predates the generic skill rename. Current runtime accepts arbitrary action labels/selectors and reports `row_or_item_action`; no new live confirmation was run. |
| visual_or_spatial_task | Count visible link-like objects in the page geometry. | `artifacts/results/execution_20260602_202554.json` | success / accept | `screenshot`, `count` | `observe_page`, `visual_svg_recognition`, `visual_dom_geometry` | false | Confirmed with DOM geometry fallback; latest plan counted visible links on Wikipedia and deterministic verifier accepted the populated count. |

## Controlled skipped

| scenario | task_type | artifact path | status | failure_stage | extracted_data key | full_vocabulary_was_used | limitations |
|---|---|---:|---|---|---|---|---|
| arXiv search results | `single_entity_metadata` in this skipped run; expected structural target is search/list extraction | `artifacts/results/execution_20260602_162913.json` | skipped / reject | `skipped_rate_limited` | none | false | arXiv was run exactly once. `open_url` detected rate/blocked page and stopped; not re-run. The artifact also shows a pre-fix routing weakness for search/list goals. |
| Anti-bot/CAPTCHA catalog pages | catalog_or_card_extraction | prior live context | controlled skipped | `skipped_captcha_or_antibot` | none | false | External anti-bot remains a controlled skip, not a pipeline failure. |

## Implemented not live-confirmed

| scenario | code status | artifact path | extracted_data key | task_type | full_vocabulary_was_used | limitations |
|---|---|---:|---|---|---|---|
| row_or_item_action on a clean public row-action app | implemented | n/a | n/a | row_or_item_action | false | W3Schools live confirmed runtime behavior, but a cleaner stable public row-action site would provide better selector diagnostics. |
| arXiv successful result extraction | implemented generically but blocked by live preflight | `artifacts/results/execution_20260602_162913.json` | none | skipped artifact routed as single_entity_metadata before later search-routing fixes | false | Not re-run by instruction/limit; needs one fresh live pass when arXiv is not rate-limited. |

## Generic fixes confirmed by targeted tests

| area | change | validation |
|---|---|---|
| Verifier nested metadata | Flatten/nested lookup plus aliases for `title/name/page_title`, `url/final_url/current_url`, and `description/summary/snippet`; partial metadata reports `uncertain` instead of hard reject. Deterministic accept is limited to reliable metadata aliases and count fields, so arbitrary/visual answers still reach verifier LLM. | `python -m pytest tests/test_verifier_payload_normalization.py tests/test_llm_verifier_vision_policy.py` |
| Verifier count fields | Populated `count` / `*_count` required fields are accepted deterministically without relying on verifier LLM JSON. | `python -m pytest tests/test_verifier_payload_normalization.py` |
| Structural routing | `search + result/list + navigation + extract` beats object-field signals. Domain words no longer affect routing or emitted `item_type`; only structural item types are emitted. | `python -m pytest tests/test_task_router_profiles.py` |
| Multi-step search navigation | Dynamic result href/open-url plans are normalized to generic first-result click; brittle guessed `wait_for url_contains` is dropped when not present in the goal; `final_url` extraction is moved after metadata navigation. | Targeted `tests/test_two_stage_workflow.py` cases plus live `execution_20260602_170455.json`. |
| Row action normalization | Bare semantic row clicks and incomplete `click_row_action` args are rewritten to `click_row_action` with generic action/condition extraction; weak text waits are dropped. | Targeted `tests/test_two_stage_workflow.py` cases plus live `execution_20260602_201231.json`. |
| Row runtime selection | Matching row candidates prefer specific `tr/li/[role=row/listitem]` over large parent containers; delete matching no longer treats `Next` as a delete target because of a bare `x`. | `python -m pytest tests/test_runtime_generic_fallbacks.py` |
| Visual local fallback | Replanner local fallback builds `visual_observe -> visual_extract_object_count` for visible/count/center/link-like goals. Runtime can count DOM geometry for link/button/input/item targets when SVG counts are absent. | `python -m pytest tests/test_runtime_generic_fallbacks.py` and live `execution_20260602_202554.json`. |
| Initial planner without explicit URL | Planner requests disable model thinking so structured JSON is returned in `message.content`; the model can infer a canonical public URL without a new URL fallback. | Targeted `tests/test_llm_client_retry.py` and `tests/test_initial_plan_normalization.py`. |
| Generic object description | `field_schema` uses meta description, then the first meaningful visible main-content paragraph. Missing values remain missing. | Targeted `tests/test_semantic_region_fields.py`. |
| Arbitrary requested fields | Router/schema preprocessing derives explicitly listed field names from the goal instead of a closed contact/product/content vocabulary. | Targeted `tests/test_semantic_region_fields.py` and `tests/test_task_router_profiles.py`. |
| Generic repeated blocks | `card_items` discovers repeated sibling-shaped visible blocks and removes identical descriptions shared by distinct items instead of fabricating per-item descriptions. | Targeted `tests/test_semantic_region_fields.py` and `tests/test_card_items_runtime.py`. |
| Generic table projection | `table_rows` projects `headers`, `cells`, and `fields_by_header` to the requested columns and generates aliases directly from header text. | Targeted `tests/test_table_rows_header_filter.py`. |
| Generic row/item action | Runtime accepts arbitrary action labels, target candidates, or an explicit selector; fixed star/delete/reply profiles and the email-specific skill name were removed. | Targeted `tests/test_action_handler_fallbacks.py`, `tests/test_card_items_runtime.py`, and `tests/test_two_stage_workflow.py`. |
| Generic typed scalar extraction | `value_near_anchor` no longer infers a contact profile or unknown values as numbers. It uses a supported generic value shape or an explicit observed `value_pattern`. | Targeted `tests/test_action_handler_fallbacks.py` and validator tests. |
| No task-profile runtime aliases | Real-web normalizer/executor no longer silently maps `product_cards`, `paper_results`, `repository_results`, `article_results`, `news_items`, or package-specific intents to hidden runtime behavior. Plans must use structural intents and requested `fields`. | Targeted `tests/test_semantic_region_fields.py` and `tests/test_card_items_runtime.py`. |
| Scope boundary | MiniWoB/BrowserGym implementation was restored unchanged; this cleanup affects only the real-web planner/executor/verifier path. | Static `git diff --name-only` scope audit. |

## Latest focused commands

- `python -m pytest tests/test_article_news_text_fallback.py tests/test_search_results_direct_hit.py tests/test_runtime_generic_fallbacks.py tests/test_verifier_payload_normalization.py tests/test_semantic_region_fields.py tests/test_card_items_runtime.py tests/test_table_rows_header_filter.py tests/test_action_handler_fallbacks.py tests/test_task_router_profiles.py tests/test_initial_plan_normalization.py tests/test_llm_client_retry.py tests/test_two_stage_workflow.py tests/test_plan_validator_group_references.py tests/test_interaction_grounding.py tests/test_numeric_token_normalization.py -q --basetemp artifacts/pytest_tmp_realweb_final_2` (`118 passed`)
- `python -m pytest tests/test_benchmark_contract_normalizer.py tests/test_benchmark_runtime_diagnostics.py tests/test_compare_pipeline.py tests/test_corrective_retry_and_json_parsing.py tests/test_extract_pattern_from_page_text.py tests/test_extract_text_fallback.py tests/test_llm_verifier_vision_policy.py tests/test_page_candidates.py tests/test_pipeline_smoke.py tests/test_replanner_prompt_compaction.py tests/test_stability_smoke_suite.py tests/test_wikipedia_planner_repair.py -q --basetemp artifacts/pytest_tmp_realweb_regression_2` (`92 passed`)
- `python -m py_compile app/executor/action_handlers.py app/executor/playwright_executor.py app/interaction/action_grounder.py app/orchestrator/workflow_manager.py app/planner/action_vocab.py app/planner/planner.py app/planner/prompts.py app/planner/replanner.py app/planner/task_router.py app/utils/llm_client.py app/validator/plan_validator.py app/verifier/llm_verifier.py`

All focused pytest commands completed with a non-fatal cache warning because `.pytest_cache` is not writable in this workspace.
