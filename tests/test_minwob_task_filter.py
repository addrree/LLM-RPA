from app.browsergym_integration.miniwob_tasks import (
    env_id_for_task_name,
    select_minwob_subset,
    task_name_from_env_id,
)


def test_task_name_parsing_and_env_id_building():
    assert task_name_from_env_id("browsergym/miniwob.click-button") == "click-button"
    assert env_id_for_task_name("click-button") == "browsergym/miniwob.click-button"


def test_task_filter_accepts_env_ids_and_task_names():
    envs = [
        "browsergym/miniwob.click-button",
        "browsergym/miniwob.enter-text",
        "browsergym/miniwob.book-flight",
    ]
    assert select_minwob_subset(envs, task_ids="enter-text,browsergym/miniwob.book-flight") == [
        "browsergym/miniwob.enter-text",
        "browsergym/miniwob.book-flight",
    ]


def test_include_exclude_and_limit():
    envs = [
        "browsergym/miniwob.click-button",
        "browsergym/miniwob.click-link",
        "browsergym/miniwob.enter-text",
    ]
    assert select_minwob_subset(envs, include_patterns="click", exclude_patterns="link", limit=5) == [
        "browsergym/miniwob.click-button"
    ]


def test_recommended_subset_ignores_missing_envs():
    envs = ["browsergym/miniwob.click-button", "browsergym/miniwob.some-new-task"]
    assert select_minwob_subset(envs, limit=10) == ["browsergym/miniwob.click-button"]


def test_exclude_book_flight():
    envs = ["browsergym/miniwob.click-button", "browsergym/miniwob.book-flight"]
    assert select_minwob_subset(envs, exclude_patterns="book-flight") == ["browsergym/miniwob.click-button"]
