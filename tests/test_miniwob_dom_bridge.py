from app.browsergym_integration.miniwob_dom_bridge import merge_dom_candidates_with_ax
from app.browsergym_integration.miniwob_policy import make_click_action


def test_merge_dom_ax_backend_node_enriches_text_and_keeps_bid():
    ax = [{"bid": "15", "role": "generic", "backendDOMNodeId": "43", "text": ""}]
    dom = [{"source": "dom", "tag": "a", "backendDOMNodeId": "43", "text": "eu.", "href": "#"}]
    out = merge_dom_candidates_with_ax(ax, dom)
    c = next(x for x in out if x.get("bid") == "15")
    assert c["text"] == "eu."


def test_dom_link_without_bid_remains():
    out = merge_dom_candidates_with_ax([], [{"source": "dom", "tag": "a", "href": "/x", "text": "vitae", "bbox": {"x": 1}}])
    assert any(c.get("tag") == "a" for c in out)


def test_dom_empty_email_controls_remain():
    dom = [
        {"source": "dom", "tag": "textarea", "id": "reply-text", "text": "", "visible": True},
        {"source": "dom", "tag": "span", "id": "send-reply", "text": "", "visible": True},
        {"source": "dom", "tag": "span", "className": "star", "text": "", "visible": True},
    ]
    out = merge_dom_candidates_with_ax([], dom)
    assert any(c.get("tag") == "textarea" and c.get("id") == "reply-text" for c in out)
    assert any(c.get("id") == "send-reply" for c in out)
    assert any(c.get("className") == "star" for c in out)


def test_merge_dom_ax_by_bbox_overlap_enriches():
    ax = [{"bid": "15", "role": "generic", "text": "", "bbox": {"x": 10, "y": 10, "width": 100, "height": 20}}]
    dom = [{"source": "dom", "tag": "a", "text": "sed.", "className": "ui-state-default", "bbox": {"x": 12, "y": 10, "width": 100, "height": 20}}]
    out = merge_dom_candidates_with_ax(ax, dom)
    c = next(x for x in out if x.get("bid") == "15")
    assert c["text"] == "sed."
    assert c["className"] == "ui-state-default"


def test_make_click_action_bid_and_dom_center():
    a, s = make_click_action({"bid": "x"}, [])
    assert a.startswith("click(") and s == "bid_click"
    a2, s2 = make_click_action({"visible": True, "bbox": {"x": 1}, "browsergym_center_x": 10, "browsergym_center_y": 20}, [])
    assert a2.startswith("mouse_click(") and s2 == "dom_center_mouse_click"
    a3, _ = make_click_action({"visible": False}, [])
    assert a3 is None
