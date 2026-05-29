from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from app.config import SCREENSHOTS_DIR
from app.interaction.candidate_adapter import compact_text_lines, normalize_candidates_for_extraction, split_candidate_groups
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
        rich_context = await self._collect_rich_context(page=page, limit=250)
        candidate_groups = split_candidate_groups(rich_context.get("candidates", []))

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
            visible_links=rich_context.get("links", [])[:80],
            text_lines=compact_text_lines(body_text),
            candidates=candidate_groups["candidates"],
            buttons=candidate_groups["buttons"],
            links=rich_context.get("links", candidate_groups["links"]),
            inputs=candidate_groups["inputs"],
            rows=rich_context.get("rows", []),
            tables=rich_context.get("tables", []),
            timestamp=datetime.now(UTC),
            page_text=body_text[:text_limit],
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

    async def _collect_rich_context(self, *, page, limit: int) -> dict:
        try:
            payload = await page.evaluate(
                """
                ({ limit }) => {
                  const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
                  const cssEscape = (value) => {
                    if (window.CSS && CSS.escape) return CSS.escape(value);
                    return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
                  };
                  const isVisible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity || 1) !== 0 && rect.width > 0 && rect.height > 0;
                  };
                  const cssPath = (el) => {
                    if (!el || !el.tagName) return "";
                    if (el.id) return `#${cssEscape(el.id)}`;
                    const parts = [];
                    let cur = el;
                    while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 6) {
                      const tag = cur.tagName.toLowerCase();
                      let part = tag;
                      const cls = Array.from(cur.classList || []).slice(0, 2).map(cssEscape);
                      if (cls.length) part += "." + cls.join(".");
                      const parent = cur.parentElement;
                      if (parent) {
                        const siblings = Array.from(parent.children).filter((sib) => sib.tagName === cur.tagName);
                        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
                      }
                      parts.unshift(part);
                      cur = parent;
                    }
                    return parts.join(" > ");
                  };
                  const bbox = (el) => {
                    const rect = el.getBoundingClientRect();
                    return { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) };
                  };
                  const candidateFor = (el, index) => ({
                    candidate_id: `obs_${index + 1}`,
                    tag: (el.tagName || "").toLowerCase(),
                    role: el.getAttribute("role") || "",
                    id: el.getAttribute("id") || "",
                    className: el.getAttribute("class") || "",
                    text: norm(el.innerText || el.textContent || ""),
                    innerText: norm(el.innerText || ""),
                    textContent: norm(el.textContent || ""),
                    ariaLabel: el.getAttribute("aria-label") || "",
                    name: el.getAttribute("name") || "",
                    title: el.getAttribute("title") || "",
                    href: el.href || el.getAttribute("href") || "",
                    selector: cssPath(el),
                    bbox: bbox(el),
                    visible: true,
                  });
                  const interactiveSelector = [
                    "button", "a[href]", "input", "textarea", "select", "option",
                    "[role='button']", "[role='link']", "[role='textbox']", "[role='checkbox']",
                    "[role='radio']", "[role='combobox']", "[role='option']", "[onclick]", "[tabindex]"
                  ].join(",");
                  const candidates = Array.from(document.querySelectorAll(interactiveSelector))
                    .filter(isVisible)
                    .slice(0, limit)
                    .map(candidateFor);
                  const links = Array.from(document.querySelectorAll("a[href]"))
                    .filter(isVisible)
                    .slice(0, limit)
                    .map((el) => ({ text: norm(el.innerText || el.textContent || el.getAttribute("aria-label") || ""), href: el.href || el.getAttribute("href") || "", selector: cssPath(el), bbox: bbox(el) }))
                    .filter((item) => item.text && item.href);
                  const rows = Array.from(document.querySelectorAll("tr,[role='row'],li,article,[class*='row'],[class*='item'],[class*='card']"))
                    .filter(isVisible)
                    .slice(0, limit)
                    .map((el, index) => ({
                      row_id: `row_${index + 1}`,
                      tag: (el.tagName || "").toLowerCase(),
                      role: el.getAttribute("role") || "",
                      className: el.getAttribute("class") || "",
                      text: norm(el.innerText || el.textContent || ""),
                      selector: cssPath(el),
                      bbox: bbox(el),
                      cells: Array.from(el.querySelectorAll("th,td,[role='cell'],[role='gridcell']")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean),
                      links: Array.from(el.querySelectorAll("a[href]")).map((a) => ({ text: norm(a.innerText || a.textContent || ""), href: a.href || a.getAttribute("href") || "" })).filter((a) => a.text && a.href),
                    }))
                    .filter((row) => row.text);
                  const tables = Array.from(document.querySelectorAll("table,[role='table'],[role='grid']"))
                    .filter(isVisible)
                    .slice(0, 20)
                    .map((table, index) => ({
                      table_id: `table_${index + 1}`,
                      selector: cssPath(table),
                      headers: Array.from(table.querySelectorAll("thead th,thead [role='columnheader'],tr:first-child th,tr:first-child [role='columnheader']")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean),
                      rows: Array.from(table.querySelectorAll("tbody tr,tr,[role='row']")).slice(0, limit).map((row) => Array.from(row.querySelectorAll("th,td,[role='cell'],[role='gridcell']")).map((cell) => norm(cell.innerText || cell.textContent || "")).filter(Boolean)).filter((cells) => cells.length > 0),
                    }));
                  return { candidates, links, rows, tables };
                }
                """,
                {"limit": max(limit, 1)},
            )
        except Exception:
            return {"candidates": [], "links": [], "rows": [], "tables": []}
        if not isinstance(payload, dict):
            return {"candidates": [], "links": [], "rows": [], "tables": []}
        candidates = [
            self._compact_candidate(candidate)
            for candidate in normalize_candidates_for_extraction(payload.get("candidates") or [])[:120]
        ]
        return {
            "candidates": candidates,
            "links": [self._compact_link(item) for item in payload.get("links") or [] if isinstance(item, dict)][:80],
            "rows": [self._compact_row(item) for item in payload.get("rows") or [] if isinstance(item, dict)][:80],
            "tables": [self._compact_table(item) for item in payload.get("tables") or [] if isinstance(item, dict)][:8],
        }

    @staticmethod
    def _truncate(value: object, limit: int) -> str:
        text = " ".join(str(value or "").split())
        return text[:limit]

    @classmethod
    def _compact_candidate(cls, item: dict) -> dict:
        compact = dict(item)
        for key in ("text", "innerText", "inner_text", "textContent", "text_content", "name", "title", "ariaLabel", "aria_label"):
            if key in compact:
                compact[key] = cls._truncate(compact.get(key), 220)
        if "href" in compact:
            compact["href"] = cls._truncate(compact.get("href"), 260)
        return compact

    @classmethod
    def _compact_link(cls, item: dict) -> dict:
        return {
            "text": cls._truncate(item.get("text"), 180),
            "href": cls._truncate(item.get("href"), 300),
            "selector": cls._truncate(item.get("selector"), 220),
            "bbox": item.get("bbox") or {},
        }

    @classmethod
    def _compact_row(cls, item: dict) -> dict:
        return {
            "row_id": item.get("row_id"),
            "tag": item.get("tag"),
            "role": item.get("role"),
            "className": cls._truncate(item.get("className"), 160),
            "text": cls._truncate(item.get("text"), 420),
            "selector": cls._truncate(item.get("selector"), 220),
            "bbox": item.get("bbox") or {},
            "cells": [cls._truncate(cell, 120) for cell in (item.get("cells") or [])[:16]],
            "links": [cls._compact_link(link) for link in (item.get("links") or [])[:6] if isinstance(link, dict)],
        }

    @classmethod
    def _compact_table(cls, item: dict) -> dict:
        rows = []
        for row in (item.get("rows") or [])[:30]:
            if isinstance(row, list):
                rows.append([cls._truncate(cell, 120) for cell in row[:16]])
        return {
            "table_id": item.get("table_id"),
            "selector": cls._truncate(item.get("selector"), 220),
            "headers": [cls._truncate(header, 120) for header in (item.get("headers") or [])[:16]],
            "rows": rows,
        }

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
