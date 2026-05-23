from __future__ import annotations

from typing import Any


def _norm_text(v: Any) -> str:
    return " ".join(str(v or "").strip().split())


def _is_meaningful_text(v: Any) -> bool:
    t = _norm_text(v).lower()
    return bool(t and t not in {"generic", "listitem", "option", "menuitem"})


def extract_miniwob_dom_candidates(page) -> list[dict[str, Any]]:
    try:
        scale = float(getattr(page, "_bgym_scale_factor", 1.0) or 1.0)
    except Exception:
        scale = 1.0
    script = """
    () => {
      const selectors = [
        'a[href]','button','input','textarea','select','option','label','[role]','[onclick]',
        '.ui-menu-item','.ui-autocomplete li','.ui-menu li','.ui-datepicker','.ui-datepicker-title',
        '.ui-datepicker-month','.ui-datepicker-year','.ui-datepicker-prev','.ui-datepicker-next',
        '.ui-datepicker-calendar td a','.ui-datepicker-calendar td','.ui-state-default','.ui-state-active','.ui-priority-secondary'
      ];
      const seen = new Set();
      const out = [];
      const els = Array.from(document.querySelectorAll(selectors.join(',')));
      for (const el of els) {
        if (seen.has(el)) continue;
        seen.add(el);
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const visible = !!(rect.width || rect.height) && style.display !== 'none' && style.visibility !== 'hidden';
        const parent = el.parentElement;
        out.push({
          source: 'dom',
          tag: (el.tagName || '').toLowerCase(), role: el.getAttribute('role') || '', type: el.getAttribute('type') || '',
          id: el.id || '', name: el.getAttribute('name') || '', value: el.value || '',
          text: (el.innerText || el.textContent || '').trim(), innerText: (el.innerText || '').trim(), textContent: (el.textContent || '').trim(),
          href: el.getAttribute('href') || '', title: el.getAttribute('title') || '', ariaLabel: el.getAttribute('aria-label') || '',
          className: typeof el.className === 'string' ? el.className : '', placeholder: el.getAttribute('placeholder') || '',
          disabled: !!el.disabled, checked: !!el.checked, selected: !!el.selected, visible,
          bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height, left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom},
          page_center_x: rect.x + rect.width/2, page_center_y: rect.y + rect.height/2,
          bid: el.getAttribute('bid') || el.getAttribute('data-testid') || el.getAttribute('browsergym_id') || el.getAttribute('data-bid') || el.getAttribute('ref') || '',
          bid_source: el.getAttribute('bid') ? 'bid' : (el.getAttribute('data-testid') ? 'data-testid' : (el.getAttribute('browsergym_id') ? 'browsergym_id' : (el.getAttribute('data-bid') ? 'data-bid' : (el.getAttribute('ref') ? 'ref' : '')))),
          center_x: rect.x + rect.width/2, center_y: rect.y + rect.height/2,
          backendDOMNodeId: el.getAttribute('backendDOMNodeId') || el.dataset.backendDomNodeId || '',
          nodeId: el.getAttribute('nodeId') || '',
          parent_text: parent ? ((parent.innerText || parent.textContent || '').trim().slice(0,200)) : '',
          parent_class: parent ? (typeof parent.className === 'string' ? parent.className : '') : '',
          parent_tag: parent ? (parent.tagName || '').toLowerCase() : ''
        });
      }
      return out;
    }
    """
    try:
        cands = page.evaluate(script)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for c in cands or []:
        if not isinstance(c, dict):
            continue
        item = dict(c)
        x, y = item.get("page_center_x"), item.get("page_center_y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            item["browsergym_center_x"] = x * scale
            item["browsergym_center_y"] = y * scale
        out.append(item)
    return out


def merge_dom_candidates_with_ax(ax_candidates: list[dict[str, Any]], dom_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(c) for c in (ax_candidates or []) if isinstance(c, dict)]
    used_dom: set[int] = set()
    for i, a in enumerate(merged):
        ax_node = str(a.get("backendDOMNodeId") or a.get("nodeId") or "")
        hit = None
        for j, d in enumerate(dom_candidates or []):
            if not isinstance(d, dict):
                continue
            dom_node = str(d.get("backendDOMNodeId") or d.get("nodeId") or "")
            if ax_node and dom_node and ax_node == dom_node:
                hit = (j, d)
                break
        if hit:
            j, d = hit
            used_dom.add(j)
            for k in ("text", "innerText", "textContent", "href", "className", "bbox", "page_center_x", "page_center_y", "browsergym_center_x", "browsergym_center_y", "tag", "source"):
                if (not a.get(k)) and d.get(k):
                    a[k] = d.get(k)
            merged[i] = a
    for j, d in enumerate(dom_candidates or []):
        if j in used_dom:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("href") or _is_meaningful_text(d.get("text") or d.get("innerText") or d.get("textContent")):
            merged.append(dict(d))

    def score(c: dict[str, Any]) -> tuple[int, int, int]:
        bid = 1 if str(c.get("bid") or "").strip() else 0
        meaningful = 1 if _is_meaningful_text(c.get("text") or c.get("innerText") or c.get("textContent") or c.get("name")) else 0
        dom = 1 if c.get("source") == "dom" else 0
        return (bid + meaningful, bid, meaningful + dom)

    return sorted(merged, key=score, reverse=True)
