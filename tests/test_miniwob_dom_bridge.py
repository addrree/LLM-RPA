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


def test_make_click_action_bid_and_dom_center():
    a, s = make_click_action({"bid": "x"}, [])
    assert a.startswith("click(") and s == "bid_click"
    a2, s2 = make_click_action({"visible": True, "bbox": {"x": 1}, "browsergym_center_x": 10, "browsergym_center_y": 20}, [])
    assert a2.startswith("mouse_click(") and s2 == "dom_center_mouse_click"
    a3, _ = make_click_action({"visible": False}, [])
    assert a3 is None
