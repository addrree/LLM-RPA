from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from app.config import SCREENSHOTS_DIR
from app.schemas.page_snapshot import HeadingSnapshot, PageSnapshot

UTC = timezone.utc


class PageObserver:
    async def observe_page(self, page, screenshot_path: str | None = None, text_limit: int = 5000) -> PageSnapshot:
        screenshot_target = screenshot_path or str(SCREENSHOTS_DIR / f"observe_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png")
        await page.screenshot(path=screenshot_target, full_page=True)

        body_text = (await page.locator("body").inner_text()).strip()
        title = await page.title()

        headings_meta = await self._collect_headings_with_context(page=page, body_text=body_text, limit=30)
        headings = [item.text for item in headings_meta if item.visible]
        labels = await self._collect_texts(page, "label", limit=30)
        buttons = await self._collect_texts(page, "button, [role='button']", limit=30)
        inputs = await self._collect_inputs(page, limit=30)

        return PageSnapshot(
            url=page.url,
            title=title,
            screenshot_path=screenshot_target,
            page_text_excerpt=body_text[:text_limit],
            visible_headings=headings,
            headings=headings_meta,
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

    async def _collect_headings_with_context(self, *, page, body_text: str, limit: int) -> List[HeadingSnapshot]:
        payload = await page.evaluate(
            """
            ({ limit }) => {
              const normalize = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const isVisible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.visibility === "hidden" || style.display === "none") return false;
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              const buildDomPath = (node) => {
                const parts = [];
                let current = node;
                let depth = 0;
                while (current && current.nodeType === 1 && depth < 8) {
                  const tag = (current.tagName || "").toLowerCase();
                  if (!tag) break;
                  parts.unshift(tag);
                  if (tag === "body" || tag === "html") break;
                  current = current.parentElement;
                  depth += 1;
                }
                return parts.join(">");
              };
              const inferRegion = (node) => {
                if (!node) return "unknown";
                const explicit = node.closest("main, article, nav, header, footer, aside");
                if (explicit) return explicit.tagName.toLowerCase();
                const contentHint = node.closest("[role='main'], [id*='content'], [class*='content']");
                if (contentHint) return "content";
                return "unknown";
              };
              const nodes = Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6")).slice(0, limit);
              return nodes.map((node, index) => ({
                text: normalize(node.innerText || node.textContent || ""),
                level: (node.tagName || "h2").toLowerCase(),
                index,
                visible: isVisible(node),
                dom_path: buildDomPath(node),
                region: inferRegion(node),
              }));
            }
            """,
            {"limit": max(limit, 1)},
        )
        return self._attach_heading_context(body_text=body_text, headings_payload=payload or [])

    @staticmethod
    def _attach_heading_context(*, body_text: str, headings_payload: list[dict]) -> List[HeadingSnapshot]:
        lines = [line.strip() for line in str(body_text or "").splitlines()]
        normalized_lines = [line for line in lines if line]
        normalized_payload: list[dict] = []
        for item in headings_payload:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            normalized_payload.append(
                {
                    "text": text,
                    "level": str(item.get("level", "h2")).lower(),
                    "index": int(item.get("index", len(normalized_payload))),
                    "visible": bool(item.get("visible", True)),
                    "dom_path": str(item.get("dom_path", "") or ""),
                    "region": str(item.get("region", "unknown") or "unknown").lower(),
                }
            )

        heading_to_line_idx: list[int | None] = []
        cursor = 0
        for heading in normalized_payload:
            found = None
            target = heading["text"].lower()
            for idx in range(cursor, len(normalized_lines)):
                if normalized_lines[idx].lower() == target:
                    found = idx
                    cursor = idx + 1
                    break
            heading_to_line_idx.append(found)

        snapshots: List[HeadingSnapshot] = []
        for idx, heading in enumerate(normalized_payload):
            start_line = heading_to_line_idx[idx]
            end_line = len(normalized_lines)
            for next_idx in range(idx + 1, len(heading_to_line_idx)):
                if heading_to_line_idx[next_idx] is not None:
                    end_line = heading_to_line_idx[next_idx]
                    break
            section_lines: list[str] = []
            if start_line is not None:
                section_lines = [
                    line.strip()
                    for line in normalized_lines[start_line + 1 : end_line]
                    if line.strip() and line.strip().lower() != heading["text"].lower()
                ]
            region = heading.get("region", "unknown")
            is_content_heading = bool(
                heading.get("visible", True)
                and len(section_lines) > 0
                and region in {"main", "article", "content", "unknown"}
                and region not in {"nav", "header", "footer", "aside"}
            )
            snapshots.append(
                HeadingSnapshot(
                    text=heading["text"],
                    level=heading["level"],
                    index=heading["index"],
                    visible=heading["visible"],
                    dom_path=heading.get("dom_path", ""),
                    region=region,
                    preview_after=section_lines[:3],
                    line_count_after=len(section_lines),
                    is_content_heading=is_content_heading,
                )
            )
        return snapshots
