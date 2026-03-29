from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from app.config import SCREENSHOTS_DIR
from app.schemas.page_snapshot import PageSnapshot

UTC = timezone.utc


class PageObserver:
    async def observe_page(self, page, screenshot_path: str | None = None, text_limit: int = 5000) -> PageSnapshot:
        screenshot_target = screenshot_path or str(SCREENSHOTS_DIR / f"observe_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png")
        await page.screenshot(path=screenshot_target, full_page=True)

        body_text = (await page.locator("body").inner_text()).strip()
        title = await page.title()

        headings = await self._collect_texts(page, "h1, h2, h3", limit=20)
        labels = await self._collect_texts(page, "label", limit=30)
        buttons = await self._collect_texts(page, "button, [role='button']", limit=30)
        inputs = await self._collect_inputs(page, limit=30)

        return PageSnapshot(
            url=page.url,
            title=title,
            screenshot_path=screenshot_target,
            page_text_excerpt=body_text[:text_limit],
            visible_headings=headings,
            visible_labels=labels,
            visible_buttons=buttons,
            visible_inputs=inputs,
            timestamp=datetime.now(UTC),
            page_text=body_text,
        )

    async def _collect_texts(self, page, selector: str, limit: int) -> List[str]:
        locator = page.locator(selector)
        count = min(await locator.count(), limit)
        values: List[str] = []
        for i in range(count):
            text = (await locator.nth(i).inner_text()).strip()
            if text:
                values.append(text)
        return values

    async def _collect_inputs(self, page, limit: int) -> List[str]:
        locator = page.locator("input, textarea, select")
        count = min(await locator.count(), limit)
        values: List[str] = []
        for i in range(count):
            item = locator.nth(i)
            label = await item.get_attribute("aria-label")
            placeholder = await item.get_attribute("placeholder")
            name = await item.get_attribute("name")
            desc = " | ".join([v for v in [label, placeholder, name] if v])
            if desc:
                values.append(desc)
        return values
