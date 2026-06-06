import asyncio

from app.executor.action_handlers import ActionHandlers


def test_click_by_semantic_target_opens_first_result_like_link():
    handler = ActionHandlers()

    async def _search_results(*, page, args, runtime_state=None):
        return [{"title": "Project Alpha", "href": "https://projects.sample.test/alpha"}]

    async def _wait(_page):
        return None

    async def _raise(_page, *, runtime_state=None, stage=""):
        return None

    class _Page:
        def __init__(self):
            self.opened = []

        async def goto(self, url, wait_until="domcontentloaded", timeout=20000):
            self.opened.append((url, wait_until, timeout))

    handler._collect_search_results_generic = _search_results  # type: ignore[method-assign]
    handler._wait_after_possible_navigation = _wait  # type: ignore[method-assign]
    handler._raise_if_page_blocked_or_limited = _raise  # type: ignore[method-assign]

    page = _Page()
    runtime_state = {}
    result = asyncio.run(
        handler.click_by_semantic_target(
            page,
            {"target_text": "first relevant repository result", "role": "link"},
            runtime_state,
        )
    )

    assert result == "Project Alpha"
    assert page.opened[0][0] == "https://projects.sample.test/alpha"
    assert runtime_state["last_opened_result"]["title"] == "Project Alpha"


def test_first_result_navigation_request_is_structural_not_literal_text():
    assert ActionHandlers._looks_like_first_result_navigation_request(
        {"target_text": "open the first relevant result", "role": "link"}
    )
    assert ActionHandlers._looks_like_first_result_navigation_request(
        {"target_text": "открой первый релевантный результат", "role": "link"}
    )
    assert not ActionHandlers._looks_like_first_result_navigation_request(
        {"target_text": "first paragraph", "role": "link"}
    )


def test_visual_extract_object_count_uses_dom_geometry_for_countable_targets():
    handler = ActionHandlers()

    class _Page:
        async def evaluate(self, script, payload=None):
            if "shapeTags" in str(script):
                return {"shape_counts": {}, "shapes": []}
            if "targetKind" in str(script):
                return {"count": 10, "items": []}
            return {}

    runtime_state = {}
    result = asyncio.run(
        handler.visual_extract_object_count(
            _Page(),
            {"target": "link", "region": {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}},
            runtime_state,
        )
    )

    assert result == 10
    assert "visual_dom_geometry" in runtime_state["used_skills"]


def test_visible_object_summary_counts_language_like_blocks_not_footer_links():
    links = [
        {"text": "Русский 2 103 000+ статей", "href": "https://ru.example.test/", "selector": "#ru"},
        {"text": "English 7,189,000+ articles", "href": "https://en.example.test/", "selector": "#en"},
        {"text": "日本語 1,503,000+ 記事", "href": "https://ja.example.test/", "selector": "#ja"},
        {"text": "Deutsch 3.125.000+ Artikel", "href": "https://de.example.test/", "selector": "#de"},
        {"text": "Français 2 761 000+ articles", "href": "https://fr.example.test/", "selector": "#fr"},
        {"text": "Español 2.116.000+ artículos", "href": "https://es.example.test/", "selector": "#es"},
        {"text": "中文 1,537,000+ 条目 / 條目", "href": "https://zh.example.test/", "selector": "#zh"},
        {"text": "Italiano 1.971.000+ voci", "href": "https://it.example.test/", "selector": "#it"},
        {"text": "Polski 1 696 000+ haseł", "href": "https://pl.example.test/", "selector": "#pl"},
        {"text": "Português 1.173.000+ artigos", "href": "https://pt.example.test/", "selector": "#pt"},
        {"text": "Google Play Store", "href": "https://play.google.com/store/apps/details", "selector": "footer .app-badge"},
        {"text": "Apple App Store", "href": "https://itunes.apple.com/app", "selector": "footer .app-badge"},
    ]

    result = ActionHandlers._visible_object_summary_from_links(
        goal="visually count the large language blocks and return count and visible language names",
        fields={
            "верни_количество": {"type": "number"},
            "названия_видимых_языков": {"type": "text"},
        },
        links=links,
    )

    assert result["верни_количество"] == 10
    assert result["названия_видимых_языков"] == [
        "Русский",
        "English",
        "日本語",
        "Deutsch",
        "Français",
        "Español",
        "中文",
        "Italiano",
        "Polski",
        "Português",
    ]
    assert result["status"] == "success"


def test_address_candidate_extracts_next_line_after_address_label():
    source = """
    Contacts
    Legal address
    Russia, 199034, Saint Petersburg, University Embankment, 7-9
    Email
    office@example.test
    """

    assert ActionHandlers._extract_address_candidate(region_text=source, anchors=[]) == (
        "Russia, 199034, Saint Petersburg, University Embankment, 7-9"
    )


def test_phone_candidate_accepts_unicode_dashes():
    value = ActionHandlers._extract_typed_scalar_candidate(
        value_type="phone",
        text="Телефоны\n+7 (812) 328–96–44\n+7 (812) 328–95–39",
        href_values=[],
    )

    assert value == "+7 (812) 328–96–44"


def test_row_condition_prefers_specific_row_over_large_matching_container():
    rows = [
        {
            "tag": "div",
            "role": "",
            "text": "Tutorial page Hit the gym Remove Next",
            "selector": "main > div",
            "cells": [],
        },
        {
            "tag": "li",
            "role": "",
            "text": "Hit the gym",
            "selector": "#myUL > li:nth-of-type(1)",
            "cells": [],
        },
    ]

    selected = ActionHandlers._best_matching_row_by_terms(rows=rows, terms=["Hit the gym"])

    assert selected["selector"] == "#myUL > li:nth-of-type(1)"


def test_row_condition_rejects_only_broad_container_match():
    row = {
        "tag": "div",
        "role": "",
        "text": "Pay bills " + ("tutorial source code " * 120),
        "selector": "main > div",
        "cells": [],
    }

    assert ActionHandlers._row_candidate_is_too_broad(row=row, terms=["Pay bills"])


def test_row_action_control_match_does_not_treat_next_as_delete_x():
    target_words = ["trash", "delete", "remove", "close", "×"]

    assert not ActionHandlers._row_action_control_matches(haystack="Next ws-btn", target_words=target_words)
    assert ActionHandlers._row_action_control_matches(haystack="× close", target_words=target_words)
