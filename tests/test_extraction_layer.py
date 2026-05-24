from app.extraction.intent_parser import parse_extraction_intent
from app.extraction.page_extractor import build_extraction_context
from app.extraction.extraction_controller import solve_extraction_task


def test_numbers_extracted():
    ctx = build_extraction_context({"visible_text": "price 10 and 25"}, {"axtree_excerpt": ""}, [])
    assert any(n["value"] == 10 for n in ctx["numeric_values"])


def test_list_email_calendar_tree_extractors():
    candidates = [
        {"text": "item 1", "role": "listitem", "bid": "l1"},
        {"text": "From: Alice Subject: Hi", "role": "row", "bid": "e1"},
        {"text": "10:30 AM Meeting", "className": "calendar-event", "bid": "cal1"},
        {"text": "Node A", "role": "treeitem", "bid": "t1"},
    ]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    assert ctx["list_like_items"]
    assert ctx["email_like_items"]
    assert ctx["calendar_like_items"]
    assert ctx["tree_like_items"]


def test_intent_parser_examples():
    assert parse_extraction_intent("Find the product with the highest rating")["intent"] == "find_max_numeric"
    assert parse_extraction_intent("Count visible product cards")["intent"] == "count_objects"
    assert parse_extraction_intent("Find the email marked important")["intent"] == "find_important_email"


def test_solver_max_count_parity_email_grid_find_text():
    candidates = [
        {"text": "12", "role": "button", "bid": "b12"},
        {"text": "87", "role": "button", "bid": "b87"},
        {"text": "odd", "role": "button", "bid": "bo"},
        {"text": "even", "role": "button", "bid": "be"},
        {"text": "From: Bob Subject: Important", "role": "row", "bid": "mail1"},
        {"text": "R1C1", "role": "gridcell", "bid": "g1"},
    ]
    ctx = build_extraction_context({"visible_text": "12 87"}, {"axtree_excerpt": "Find word apple\napple"}, candidates)
    assert solve_extraction_task({"intent": "find_max_numeric"}, ctx, candidates, []).answer == "87"
    assert solve_extraction_task({"intent": "count_objects"}, ctx, candidates, []).answer is not None
    assert solve_extraction_task({"intent": "parity_check"}, ctx, candidates, []).answer in {"odd", "even"}
    assert solve_extraction_task({"intent": "find_email"}, ctx, candidates, []) is not None
    assert solve_extraction_task({"intent": "grid_lookup"}, ctx, candidates, []) is not None
    assert solve_extraction_task({"intent": "find_text"}, ctx, candidates, []) is not None


def test_max_numeric_ignores_wrapper_candidate():
    candidates = [
        {"text": "4610\nSubmit", "bid": "wrapbid", "id": "wrap", "tag": "div", "bbox": {"width": 400, "height": 400}},
        {"text": "4610", "bid": "n1", "tag": "button", "bbox": {"width": 30, "height": 20}},
    ]
    ctx = build_extraction_context({"visible_text": "4610"}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_max_numeric", "constraints": {}}, ctx, candidates, [])
    assert d is not None and d.selected_candidate is not None
    assert d.selected_candidate.get("bid") == "n1"


def test_submit_selector_ignores_wrapper_and_uses_real_button():
    candidates = [
        {"text": "368\nSubmit", "bid": "wrap", "id": "wrap", "tag": "div", "bbox": {"width": 300, "height": 220}},
        {"text": "Submit", "bid": "s1", "tag": "button", "role": "button"},
    ]
    ctx = build_extraction_context({"visible_text": "find greatest"}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_max_numeric", "constraints": {"history": [{"action": "click(\"n1\", \"left\")", "selected_candidate_bid": "n1"}]}}, {**ctx, "numeric_values": [{"value": 368}]}, [{"text": "368", "bid": "n1"}, *candidates], [])
    assert d is not None and d.strategy == "max_numeric_submit"
    assert d.selected_candidate.get("bid") == "s1"
