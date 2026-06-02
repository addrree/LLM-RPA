# Structural Task Type Evaluation Matrix

Manual live checks were run through `python -m app.main`. No
`scripts/run_real_web_skill_suite.py` run was used, and no `example.org` live
target was used in this pass.

## Confirmed live through app.main

| task_type | scenario | artifact path | status / verifier | extracted_data key | used_skills | full_vocabulary_was_used | limitations |
|---|---|---:|---|---|---|---|---|
| direct_value_extraction | Extract article count next to a visible anchor. | `artifacts/results/execution_20260602_144520.json` | success / accept | `english_article_count` | `observe_page`, `extract_value_near_anchor` | false | Earlier live confirmation; not re-run in the final pass. |
| single_entity_metadata | Extract page/object metadata from a public documentation page. | `artifacts/results/execution_20260602_152028.json` | success / historical reject; verifier fixed by targeted preprocessing test | `package_metadata`, `name`, `description`, `source_url` | `observe_page`, `extract_by_intent`, `package_metadata_extraction` | false | Artifact was produced before verifier nested-field normalization; post-fix targeted verifier accepts nested metadata and aliases. |
| semantic_navigation | Click a visible navigation target, then extract opened URL/title/links. | `artifacts/results/execution_20260602_152452.json` | success / accept | `final_url`, `page_title`, `top_links` | `observe_page`, `semantic_click`, `extract_by_intent`, `extract_visible_links` | false | Earlier live confirmation; not re-run in the final pass. |
| structured_table_extraction | Extract rows/columns from a visible HTML table. | `artifacts/results/execution_20260602_152558.json` | success / accept | `extracted_table` | `observe_page`, `extract_by_intent` | false | Earlier live confirmation; not re-run in the final pass. |
| catalog_or_card_extraction | Extract generic non-product cards with title/description/link. | `artifacts/results/execution_20260602_153105.json` | success / historical reject | `extracted_cards` | `observe_page`, `extract_by_intent`, `row_list_extraction` | false | Generic `card_items` extracted cards, but this page had weak/repeated descriptions. |
| catalog_or_card_extraction + condition_filtering | Extract cards whose title contains a requested term. | `artifacts/results/execution_20260602_154131.json` | success / accept | `extracted_cards` | `observe_page`, `extract_by_intent`, `row_list_extraction` | false | Confirms confident condition propagation into generic item filtering. |
| search_results_extraction | Multi-step search result: open search page, open first relevant result, extract title/description/current URL. | `artifacts/results/execution_20260602_170455.json` | success / accept | `search_results`, `clicked_text`, `name`, `description`, `final_url` | `observe_page`, `extract_by_intent`, `row_list_extraction`, `extract_visible_links`, `semantic_click`, `package_metadata_extraction` | false | Confirmed on Python.org search. Generic fixes prevent false success when the first result is not opened and move `final_url` extraction after metadata navigation. |
| row_or_item_action | Find a row/list-like item by text and perform a delete action. | `artifacts/results/execution_20260602_201231.json` | success / accept | `row_ref`, `row_action` | `observe_page`, `row_list_email_action` | false | Confirmed on W3Schools tutorial page. The page markup is tutorial/code-heavy, so the diagnostic `row_ref.selector` can still be broad, but action completed and verifier accepted. |
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
| Verifier nested metadata | Flatten/nested lookup plus aliases for `title/name/page_title`, `url/final_url/current_url`, and `description/summary/snippet`; partial metadata reports `uncertain` instead of hard reject. | `python -m pytest tests/test_verifier_payload_normalization.py` |
| Verifier count fields | Populated `count` / `*_count` required fields are accepted deterministically without relying on verifier LLM JSON. | `python -m pytest tests/test_verifier_payload_normalization.py` |
| Search-result routing | `search + result/list + navigation + extract` beats object metadata words. Domain words remain weak hints. | `python -m pytest tests/test_task_router_profiles.py` |
| Multi-step search navigation | Dynamic result href/open-url plans are normalized to generic first-result click; brittle guessed `wait_for url_contains` is dropped when not present in the goal; `final_url` extraction is moved after metadata navigation. | Targeted `tests/test_two_stage_workflow.py` cases plus live `execution_20260602_170455.json`. |
| Row action normalization | Bare semantic row clicks and incomplete `click_row_action` args are rewritten to `click_row_action` with generic action/condition extraction; weak text waits are dropped. | Targeted `tests/test_two_stage_workflow.py` cases plus live `execution_20260602_201231.json`. |
| Row runtime selection | Matching row candidates prefer specific `tr/li/[role=row/listitem]` over large parent containers; delete matching no longer treats `Next` as a delete target because of a bare `x`. | `python -m pytest tests/test_runtime_generic_fallbacks.py` |
| Visual local fallback | Replanner local fallback builds `visual_observe -> visual_extract_object_count` for visible/count/center/link-like goals. Runtime can count DOM geometry for link/button/input/item targets when SVG counts are absent. | `python -m pytest tests/test_runtime_generic_fallbacks.py` and live `execution_20260602_202554.json`. |

## Latest focused commands

- `python -m pytest tests/test_runtime_generic_fallbacks.py`
- `python -m pytest tests/test_verifier_payload_normalization.py`
- Targeted `tests/test_two_stage_workflow.py` cases for multi-step, row action, and visual fallback.
- `python -m py_compile app/executor/action_handlers.py app/planner/replanner.py app/planner/task_router.py app/verifier/llm_verifier.py`
