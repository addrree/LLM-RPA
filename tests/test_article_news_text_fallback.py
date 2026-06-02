from __future__ import annotations

from app.executor.action_handlers import ActionHandlers


def test_article_text_fallback_extracts_title_link_author_and_date():
    source_text = """
    LATEST
    Python 3.14.5 is out!
    Hugo van Kemenade
    ·
    May 10, 2026
    A special release with a new garbage collector.
    RECENT
    Python 3.15.0 beta 1 is here!
    Hugo van Kemenade
    ·
    May 7, 2026
    The propreantepenultimate 3.15 beta is out!
    CPython: 36 Years of Source Code
    Stan Ulbrych
    ·
    March 8, 2026
    An analysis of the growth of CPython's codebase.
    """
    links = [
        {"text": "Python 3.14.5 is out!", "href": "https://blog.python.local/python-3145-is-out.html"},
        {"text": "Python 3.15.0 beta 1 is here!", "href": "https://blog.python.local/python-3150-beta-1.html"},
        {"text": "CPython: 36 Years of Source Code", "href": "https://blog.python.local/cpython-36-years.html"},
    ]

    items = ActionHandlers._collect_article_like_items_from_text(source_text=source_text, links=links, limit=5)

    assert items[0]["title"] == "Python 3.14.5 is out!"
    assert items[0]["href"] == "https://blog.python.local/python-3145-is-out.html"
    assert items[0]["author"] == "Hugo van Kemenade"
    assert items[0]["publication_time"] == "May 10, 2026"
    assert items[1]["title"] == "Python 3.15.0 beta 1 is here!"
    assert items[2]["title"] == "CPython: 36 Years of Source Code"
    assert items[2]["publication_time"] == "March 8, 2026"


def test_article_metadata_requested_understands_russian_date_goal():
    assert ActionHandlers._article_metadata_requested(
        args={"intent": "news_items"},
        runtime_state={"user_goal": "Выгрузи последние статьи: заголовок, дату и ссылку."},
    )


def test_article_link_filter_accepts_date_based_blog_paths():
    links = [
        {"text": "Python 3.14.5 is out!", "href": "https://blog.python.org/2026/05/python-3145-is-out"},
        {"text": "RSS", "href": "https://blog.python.org/rss.xml"},
    ]

    filtered = ActionHandlers._filter_links_to_article_like_paths(links, current_url="https://blog.python.org/")

    assert [item["href"] for item in filtered] == ["https://blog.python.org/2026/05/python-3145-is-out"]

