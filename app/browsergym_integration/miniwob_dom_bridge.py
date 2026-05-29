from __future__ import annotations

from typing import Any
import math


def _norm_text(v: Any) -> str:
    return " ".join(str(v or "").strip().split())


def _is_meaningful_text(v: Any) -> bool:
    t = _norm_text(v).lower()
    return bool(t and t not in {"generic", "listitem", "option", "menuitem"})


def _is_actionish_candidate(c: dict[str, Any]) -> bool:
    tag = _norm_text(c.get("tag")).lower()
    role = _norm_text(c.get("role")).lower()
    hay = " ".join(
        _norm_text(c.get(field)).lower()
        for field in ("id", "className", "parent_class", "title", "ariaLabel", "name")
    )
    return (
        tag in {"input", "textarea", "button", "select"}
        or role in {"button", "link", "textbox", "menuitem"}
        or any(
            key in hay
            for key in (
                "button",
                "action",
                "icon",
                "toolbar",
                "star",
                "important",
                "trash",
                "delete",
                "reply",
                "send",
                "forward",
                "email-actions",
                "email-reply",
                "email-forward",
            )
        )
    )


def _bbox_parts(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, dict):
        return None
    x = raw.get("x", raw.get("left"))
    y = raw.get("y", raw.get("top"))
    w = raw.get("width")
    h = raw.get("height")
    if w is None and raw.get("right") is not None and x is not None:
        w = float(raw.get("right")) - float(x)
    if h is None and raw.get("bottom") is not None and y is not None:
        h = float(raw.get("bottom")) - float(y)
    if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        return None
    if float(w) <= 0 or float(h) <= 0:
        return None
    return float(x), float(y), float(w), float(h)


def bbox_iou_or_center_distance(ax_bbox: Any, dom_bbox: Any) -> bool:
    a = _bbox_parts(ax_bbox)
    d = _bbox_parts(dom_bbox)
    if not a or not d:
        return False
    ax, ay, aw, ah = a
    dx, dy, dw, dh = d
    ax2, ay2 = ax + aw, ay + ah
    dx2, dy2 = dx + dw, dy + dh
    inter_w = max(0.0, min(ax2, dx2) - max(ax, dx))
    inter_h = max(0.0, min(ay2, dy2) - max(ay, dy))
    inter = inter_w * inter_h
    union = aw * ah + dw * dh - inter
    iou = (inter / union) if union > 0 else 0.0
    if iou > 0.5:
        return True
    acx, acy = ax + aw / 2, ay + ah / 2
    dcx, dcy = dx + dw / 2, dy + dh / 2
    dist = math.hypot(acx - dcx, acy - dcy)
    sim = (min(aw, dw) / max(aw, dw) >= 0.5) and (min(ah, dh) / max(ah, dh) >= 0.5)
    if dist < 5.0 and sim:
        return True
    a_contains_d_center = ax <= dcx <= ax2 and ay <= dcy <= ay2
    d_contains_a_center = dx <= acx <= dx2 and dy <= acy <= dy2
    return a_contains_d_center or d_contains_a_center


def extract_miniwob_dom_candidates(page) -> list[dict[str, Any]]:
    try:
        scale = float(getattr(page, "_bgym_scale_factor", 1.0) or 1.0)
    except Exception:
        scale = 1.0
    script = """
    () => {
      const selectors = [
        'a[href]','a','button','input','textarea','select','option','label','[role]','[onclick]','[role="link"]','span','div','p','td','li',
        'svg','svg circle','svg rect','svg polygon','svg path','svg text','.plot-point',
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
          className: typeof el.className === 'string' ? el.className : (el.getAttribute('class') || ''), placeholder: el.getAttribute('placeholder') || '',
          fill: el.getAttribute('fill') || style.fill || '', stroke: el.getAttribute('stroke') || style.stroke || '',
          points: el.getAttribute('points') || '', d: el.getAttribute('d') || '',
          cx: el.getAttribute('cx') || '', cy: el.getAttribute('cy') || '', r: el.getAttribute('r') || '',
          x_attr: el.getAttribute('x') || '', y_attr: el.getAttribute('y') || '',
          width_attr: el.getAttribute('width') || '', height_attr: el.getAttribute('height') || '',
          fontSize: el.getAttribute('font-size') || style.fontSize || '',
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

      const isActionish = (c) => {
        const hay = [
          c.id || '', c.className || '', c.parent_class || '', c.title || '',
          c.ariaLabel || '', c.name || '', c.role || ''
        ].join(' ').toLowerCase();
        return ['input','textarea','button','select'].includes(c.tag)
          || ['button','link','textbox','menuitem'].includes((c.role || '').toLowerCase())
          || (c.tag === 'circle' && /\\(-?\\d+\\s*,\\s*-?\\d+\\)/.test(c.id || ''))
          || (['circle','rect','polygon','path','text'].includes(c.tag) && c.parent_tag === 'svg')
          || ['button','action','icon','toolbar','star','important','trash','delete','reply','send','forward','email-actions','email-reply','email-forward','plot-point'].some(k => hay.includes(k));
      };
      let outFiltered = out.filter(c => c.visible && ((c.text||c.innerText||c.textContent||'').trim() || c.href || (c.className||'').toLowerCase().includes('ui-datepicker') || (c.className||'').toLowerCase().includes('ui-autocomplete') || c.role || isActionish(c)));
      if (!outFiltered.length) {
        const all = Array.from(document.querySelectorAll('*'));
        for (const el of all) {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          const txt = (el.innerText || el.textContent || '').trim();
          const parent = el.parentElement;
          const visible = !!(rect.width || rect.height) && style.display !== 'none' && style.visibility !== 'hidden';
          const clickable = !!el.getAttribute('href') || !!el.getAttribute('onclick') || style.cursor === 'pointer';
          const hay = [
            el.id || '',
            typeof el.className === 'string' ? el.className : '',
            parent ? (typeof parent.className === 'string' ? parent.className : '') : '',
            el.getAttribute('title') || '',
            el.getAttribute('aria-label') || '',
            el.getAttribute('name') || '',
            el.getAttribute('role') || ''
          ].join(' ').toLowerCase();
          const actionish = ['input','textarea','button','select'].includes((el.tagName || '').toLowerCase())
            || ['button','link','textbox','menuitem'].includes((el.getAttribute('role') || '').toLowerCase())
            || ['button','action','icon','toolbar','star','important','trash','delete','reply','send','forward','email-actions','email-reply','email-forward'].some(k => hay.includes(k));
          if (!visible || (!(txt && txt.length) && !clickable && !actionish)) continue;
          outFiltered.push({
            source: 'dom', tag: (el.tagName||'').toLowerCase(), role: el.getAttribute('role') || '', type: el.getAttribute('type') || '',
            id: el.id || '', name: el.getAttribute('name') || '', value: el.value || '',
            text: txt, innerText: (el.innerText||'').trim(), textContent: (el.textContent||'').trim(),
            href: el.getAttribute('href') || '', title: el.getAttribute('title') || '', ariaLabel: el.getAttribute('aria-label') || '',
            className: typeof el.className === 'string' ? el.className : (el.getAttribute('class') || ''), placeholder: el.getAttribute('placeholder') || '',
            fill: el.getAttribute('fill') || style.fill || '', stroke: el.getAttribute('stroke') || style.stroke || '',
            points: el.getAttribute('points') || '', d: el.getAttribute('d') || '',
            cx: el.getAttribute('cx') || '', cy: el.getAttribute('cy') || '', r: el.getAttribute('r') || '',
            x_attr: el.getAttribute('x') || '', y_attr: el.getAttribute('y') || '',
            width_attr: el.getAttribute('width') || '', height_attr: el.getAttribute('height') || '',
            fontSize: el.getAttribute('font-size') || style.fontSize || '',
            visible, bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height, left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom},
            page_center_x: rect.x + rect.width/2, page_center_y: rect.y + rect.height/2, center_x: rect.x + rect.width/2, center_y: rect.y + rect.height/2,
            parent_text: parent ? ((parent.innerText || parent.textContent || '').trim().slice(0,200)) : '',
            parent_class: parent ? (typeof parent.className === 'string' ? parent.className : '') : '',
            parent_tag: parent ? (parent.tagName || '').toLowerCase() : ''
          });
          if (outFiltered.length >= 300) break;
        }
      }
      return outFiltered.slice(0,300);

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
    merged_by_node_count = 0
    merged_by_bbox_count = 0
    kept_dom_only_count = 0
    for i, a in enumerate(merged):
        ax_text = a.get("text") or a.get("innerText") or a.get("textContent") or a.get("name")
        needs_enrichment = bool(str(a.get("bid") or "").strip()) and not _is_meaningful_text(ax_text)
        ax_node = str(a.get("backendDOMNodeId") or a.get("nodeId") or "")
        hit = None
        for j, d in enumerate(dom_candidates or []):
            if not isinstance(d, dict):
                continue
            dom_node = str(d.get("backendDOMNodeId") or d.get("nodeId") or "")
            if ax_node and dom_node and ax_node == dom_node:
                hit = (j, d)
                merged_by_node_count += 1
                break
        if hit is None and needs_enrichment:
            ax_bbox = a.get("bbox") or a.get("browsergym_bbox")
            for j, d in enumerate(dom_candidates or []):
                if j in used_dom or not isinstance(d, dict):
                    continue
                dom_bbox = d.get("bbox") or d.get("browsergym_bbox")
                if bbox_iou_or_center_distance(ax_bbox, dom_bbox):
                    hit = (j, d)
                    merged_by_bbox_count += 1
                    break
        if hit:
            j, d = hit
            used_dom.add(j)
            for k in ("tag", "role", "type", "id", "name", "value", "text", "innerText", "textContent", "href", "title", "ariaLabel", "className", "placeholder", "parent_text", "parent_class", "parent_tag", "bbox", "browsergym_center_x", "browsergym_center_y", "page_center_x", "page_center_y", "source"):
                if (not a.get(k)) and d.get(k):
                    a[k] = d.get(k)
            merged[i] = a
    for j, d in enumerate(dom_candidates or []):
        if j in used_dom:
            continue
        if not isinstance(d, dict):
            continue
        cls = _norm_text(d.get("className")).lower()
        keep_dom = bool(
            d.get("href")
            or _is_meaningful_text(d.get("text") or d.get("innerText") or d.get("textContent"))
            or "ui-datepicker" in cls
            or "ui-autocomplete" in cls
            or _is_actionish_candidate(d)
        )
        if keep_dom:
            merged.append(dict(d))
            kept_dom_only_count += 1

    diagnostics = {
        "dom_candidates_count": len([c for c in (dom_candidates or []) if isinstance(c, dict)]),
        "merged_by_node_count": merged_by_node_count,
        "merged_by_bbox_count": merged_by_bbox_count,
        "kept_dom_only_count": kept_dom_only_count,
    }
    for c in merged:
        if isinstance(c, dict):
            c.setdefault("merge_diagnostics", diagnostics)

    def score(c: dict[str, Any]) -> tuple[int, int, int]:
        bid = 1 if str(c.get("bid") or "").strip() else 0
        meaningful = 1 if _is_meaningful_text(c.get("text") or c.get("innerText") or c.get("textContent") or c.get("name")) else 0
        dom = 1 if c.get("source") == "dom" else 0
        return (bid + meaningful, bid, meaningful + dom)

    return sorted(merged, key=score, reverse=True)
