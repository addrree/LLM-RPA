from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PageCandidateExtractor:
    """Extract a compact list of visible interactive candidates from a Playwright page.

    The extractor intentionally returns metadata only (selectors, labels, state,
    bounding boxes) and never persists raw DOM or screenshots.
    """

    max_candidates: int = 250

    async def extract(self, page) -> list[dict[str, Any]]:
        candidates = await page.evaluate(
            """
            ({ maxCandidates }) => {
              const cssEscape = (value) => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
              };
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (el.tagName && el.tagName.toLowerCase() === "option" && el.parentElement) {
                  return isVisible(el.parentElement);
                }
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
                  if (cur.id) {
                    part += `#${cssEscape(cur.id)}`;
                    parts.unshift(part);
                    break;
                  }
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
              const xPath = (el) => {
                if (!el || !el.tagName) return "";
                if (el.id) return `//*[@id="${el.id.replace(/"/g, '\\"')}"]`;
                const parts = [];
                let cur = el;
                while (cur && cur.nodeType === Node.ELEMENT_NODE) {
                  let index = 1;
                  let sib = cur.previousElementSibling;
                  while (sib) { if (sib.tagName === cur.tagName) index++; sib = sib.previousElementSibling; }
                  parts.unshift(`${cur.tagName.toLowerCase()}[${index}]`);
                  cur = cur.parentElement;
                }
                return "/" + parts.join("/");
              };
              const labelText = (el) => {
                if (!el) return "";
                if (el.labels && el.labels.length) return norm(Array.from(el.labels).map((label) => label.innerText || label.textContent).join(" "));
                const id = el.getAttribute("id");
                if (id) {
                  const label = document.querySelector(`label[for="${cssEscape(id)}"]`);
                  if (label) return norm(label.innerText || label.textContent);
                }
                const parentLabel = el.closest && el.closest("label");
                return parentLabel ? norm(parentLabel.innerText || parentLabel.textContent) : "";
              };
              const kindFor = (el) => {
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute("role") || "").toLowerCase();
                const type = (el.getAttribute("type") || "").toLowerCase();
                if (tag === "a") return "link";
                if (tag === "button" || role === "button" || type === "button" || type === "submit") return "button";
                if (tag === "textarea") return "textbox";
                if (tag === "select" || role === "combobox" || role === "listbox") return "select";
                if (tag === "option" || role === "option") return "option";
                if (type === "checkbox" || role === "checkbox") return "checkbox";
                if (type === "radio" || role === "radio") return "radio";
                if (type === "date") return "date";
                if (tag === "input" || role === "textbox" || el.isContentEditable) return "textbox";
                if (role === "menu" || role === "menuitem") return role;
                return "clickable";
              };
              const selector = [
                "button", "a[href]", "input", "textarea", "select", "option",
                "[role='button']", "[role='link']", "[role='textbox']", "[role='checkbox']",
                "[role='radio']", "[role='combobox']", "[role='listbox']", "[role='option']",
                "[role='menu']", "[role='menuitem']", "[onclick]", "[tabindex]", "[contenteditable='true']"
              ].join(",");
              const seen = new Set();
              const result = [];
              for (const el of Array.from(document.querySelectorAll(selector))) {
                if (seen.has(el) || !isVisible(el)) continue;
                seen.add(el);
                const rect = el.getBoundingClientRect();
                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute("role") || "";
                const inputType = (el.getAttribute("type") || (tag === "textarea" ? "textarea" : "")).toLowerCase();
                const text = norm(el.innerText || el.textContent || "");
                const item = {
                  candidate_id: `cand_${result.length + 1}`,
                  kind: kindFor(el),
                  role,
                  tag,
                  id: el.getAttribute("id") || "",
                  className: el.getAttribute("class") || "",
                  text,
                  innerText: norm(el.innerText || ""),
                  textContent: norm(el.textContent || ""),
                  inner_text: norm(el.innerText || ""),
                  text_content: norm(el.textContent || ""),
                  aria_label: el.getAttribute("aria-label") || "",
                  ariaLabel: el.getAttribute("aria-label") || "",
                  title: el.getAttribute("title") || "",
                  name: el.getAttribute("name") || labelText(el) || "",
                  value: el.value || el.getAttribute("value") || "",
                  placeholder: el.getAttribute("placeholder") || "",
                  href: el.href || el.getAttribute("href") || "",
                  selector: cssPath(el),
                  css_path: cssPath(el),
                  xpath: xPath(el),
                  bbox: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
                  visible: true,
                  enabled: !el.disabled && el.getAttribute("aria-disabled") !== "true",
                  input_type: inputType,
                  checked: typeof el.checked === "boolean" ? el.checked : null,
                  selected: typeof el.selected === "boolean" ? el.selected : null,
                  diagnostics: { source: "playwright_dom_candidate_extractor" },
                };
                result.push(item);
                if (result.length >= maxCandidates) break;
              }
              return result;
            }
            """,
            {"maxCandidates": self.max_candidates},
        )
        return [dict(item) for item in candidates]

    @staticmethod
    def compact(candidates: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
        keys = ["candidate_id", "kind", "role", "tag", "id", "className", "text", "innerText", "textContent", "aria_label", "ariaLabel", "name", "placeholder", "value", "href", "selector", "input_type", "bbox", "checked", "selected"]
        compacted = []
        for item in candidates[:limit]:
            compacted.append({key: item.get(key) for key in keys if item.get(key) not in (None, "", [])})
        return compacted
