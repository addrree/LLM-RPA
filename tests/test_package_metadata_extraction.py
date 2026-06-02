from __future__ import annotations

import asyncio

from app.executor.action_handlers import ActionHandlers


def test_package_metadata_follows_matching_result_before_extracting_detail():
    class FakePackagePage:
        def __init__(self):
            self.url = "https://packages.local/search?q=requests"
            self.visited = []

        async def evaluate(self, *_args, **_kwargs):
            if "/project/requests" in self.url:
                return {
                    "url": self.url,
                    "title": "requests 2.32.5",
                    "heading": "requests 2.32.5",
                    "meta_description": "Python HTTP for Humans.",
                    "paragraphs": ["Python HTTP for Humans."],
                    "body_lines": ["requests 2.32.5", "Latest version", "2.32.5", "Python HTTP for Humans."],
                    "source_title": "requests 2.32.5",
                    "candidates": [],
                }
            return {
                "url": self.url,
                "title": "Search results",
                "heading": "Search results",
                "meta_description": "Generic package index description.",
                "paragraphs": [],
                "body_lines": ['10 projects for "requests"', "Filter by classifier", "requests", "Python HTTP for Humans."],
                "source_title": "Search results",
                "candidates": [
                    {"title": "Filter by classifier", "href": "", "description": ""},
                    {
                        "title": "requests",
                        "href": "https://packages.local/project/requests/",
                        "description": "Python HTTP for Humans.",
                        "text": "requests Python HTTP for Humans.",
                    },
                ],
            }

        async def goto(self, url, **_kwargs):
            self.url = url
            self.visited.append(url)

        async def wait_for_timeout(self, *_args, **_kwargs):
            return None

        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

    page = FakePackagePage()
    metadata = asyncio.run(
        ActionHandlers()._extract_package_metadata_generic(
            page=page,
            args={"intent": "package_metadata"},
            runtime_state={"user_goal": "find package requests"},
        )
    )

    assert page.visited == ["https://packages.local/project/requests/"]
    assert metadata["name"] == "requests"
    assert metadata["latest_version"] == "2.32.5"
    assert metadata["description"] == "Python HTTP for Humans."
