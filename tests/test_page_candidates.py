from __future__ import annotations

import asyncio
import pytest
from playwright.async_api import async_playwright

from app.interaction.page_candidates import PageCandidateExtractor


async def _extract_controls():
    html = """
    <button id='b'>Save</button><a id='l' href='/next'>Next</a>
    <input id='t' placeholder='Name'><textarea id='ta'></textarea>
    <input id='c' type='checkbox'><input id='r' type='radio'>
    <select id='s'><option>Paris</option></select><input id='d' type='date'>
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)
        candidates = await PageCandidateExtractor().extract(page)
        await browser.close()
    return candidates


def test_page_candidate_extractor_extracts_controls():
    try:
        candidates = asyncio.run(_extract_controls())
    except Exception as exc:  # noqa: BLE001
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
            pytest.skip(f"Playwright browser binaries are unavailable: {exc}")
        raise
    kinds = {candidate["kind"] for candidate in candidates}
    assert {"button", "link", "textbox", "checkbox", "radio", "select", "option", "date"}.issubset(kinds)
    assert all("selector" in candidate and "candidate_id" in candidate for candidate in candidates)
