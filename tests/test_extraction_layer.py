from app.extraction.intent_parser import parse_extraction_intent
from app.extraction.page_extractor import build_extraction_context
from app.extraction.extraction_controller import solve_extraction_task


def test_numbers_extracted():
    ctx = build_extraction_context({"visible_text": "price 10 and 25"}, {"axtree_excerpt": ""}, [])
    assert any(n["value"] == 10 for n in ctx["numeric_values"])


def test_intent_parser_examples():
    assert parse_extraction_intent("Find the product with the highest rating")["intent"] == "find_max_numeric"
    assert parse_extraction_intent("Find the email marked important")["intent"] == "find_important_email"
    p = parse_extraction_intent('Find the email by Marlie and click the star icon to mark it as important.')
    assert p["intent"] == "find_email"
    assert p["constraints"]["requested_email_action"] == "star"


def test_max_numeric_ignores_wrapper_candidate_and_submits():
    candidates = [
        {"text": "4610\nSubmit", "bid": "wrapbid", "id": "wrap", "tag": "div", "bbox": {"width": 400, "height": 400}},
        {"text": "4610", "bid": "n1", "tag": "button", "bbox": {"width": 30, "height": 20}},
        {"text": "Submit", "bid": "s1", "tag": "button"},
    ]
    ctx = build_extraction_context({"visible_text": "4610"}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_max_numeric", "constraints": {}}, ctx, candidates, [])
    assert d and d.selected_candidate and d.selected_candidate.get("bid") == "n1"
    d2 = solve_extraction_task({"intent": "find_max_numeric", "constraints": {"history": [{"selected_candidate_bid": "n1"}]}}, ctx, candidates, [])
    assert d2 and d2.strategy == "max_numeric_submit" and d2.selected_candidate.get("bid") == "s1"


def test_parity_rows_sequence_and_submit():
    candidates = [
        {"text": "Odd", "bid": "o1", "className": "row", "parent_text": "Odd\n2\nEven", "bbox": {"y": 10, "height": 10}},
        {"text": "Even", "bid": "e1", "className": "row", "parent_text": "Odd\n2\nEven", "bbox": {"y": 10, "height": 10}},
        {"text": "Odd", "bid": "o2", "className": "row", "parent_text": "Odd\n5\nEven", "bbox": {"y": 30, "height": 10}},
        {"text": "Even", "bid": "e2", "className": "row", "parent_text": "Odd\n5\nEven", "bbox": {"y": 30, "height": 10}},
        {"text": "Odd", "bid": "o3", "className": "row", "parent_text": "Odd\n8\nEven", "bbox": {"y": 50, "height": 10}},
        {"text": "Even", "bid": "e3", "className": "row", "parent_text": "Odd\n8\nEven", "bbox": {"y": 50, "height": 10}},
        {"text": "Submit", "bid": "s1", "tag": "button"},
    ]
    ctx = build_extraction_context({"visible_text": "2 5 8"}, {"axtree_excerpt": ""}, candidates)
    d1 = solve_extraction_task({"intent": "parity_check", "constraints": {}}, ctx, candidates, [])
    assert d1.selected_candidate.get("bid") == "e1" and d1.diagnostics["rows_total"] == 3
    d2 = solve_extraction_task({"intent": "parity_check", "constraints": {"history": [{"selected_candidate_bid": "e1"}]}}, ctx, candidates, [])
    assert d2.selected_candidate.get("bid") == "o2"
    d3 = solve_extraction_task({"intent": "parity_check", "constraints": {"history": [{"selected_candidate_bid": "e1"}, {"selected_candidate_bid": "o2"}]}}, ctx, candidates, [])
    assert d3.selected_candidate.get("bid") == "e3"
    d4 = solve_extraction_task({"intent": "parity_check", "constraints": {"history": [{"selected_candidate_bid": "e1"}, {"selected_candidate_bid": "o2"}, {"selected_candidate_bid": "e3"}]}}, ctx, candidates, [])
    assert d4.strategy == "parity_submit" and d4.selected_candidate.get("bid") == "s1"


def test_ordinal_word_fill_then_submit():
    candidates = [
        {"text": "one two three four five six seven eight nine ten eleven twelve thirteen Fourteen", "tag": "p", "bid": "p1"},
        {"tag": "input", "bid": "t1", "role": "textbox"},
        {"text": "Submit", "tag": "button", "bid": "s1"},
    ]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    d1 = solve_extraction_task({"intent": "ordinal_word_extraction", "constraints": {"ordinal_index": 14}}, ctx, candidates, [])
    assert d1.strategy == "ordinal_word_fill" and "Fourteen" in d1.action
    d2 = solve_extraction_task({"intent": "ordinal_word_extraction", "constraints": {"ordinal_index": 14, "history": [{"action": d1.action}]}}, ctx, candidates, [])
    assert d2.strategy == "ordinal_word_submit" and d2.selected_candidate.get("bid") == "s1"


def test_email_reply_workflow_and_star():
    row = {"text": "From: Winona Subject: Hi", "role": "row", "bid": "r1"}
    candidates1 = [row]
    ctx1 = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates1)
    intent = parse_extraction_intent('Find the email by Winona and reply to them with the text "Sapien sit.".')
    d1 = solve_extraction_task(intent, ctx1, candidates1, [])
    assert d1.strategy == "email_open_row"

    candidates2 = [row, {"text": "Reply", "bid": "rep1", "tag": "button", "className": "email-header"}]
    d2 = solve_extraction_task({"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}]}}, ctx1, candidates2, [])
    assert d2.strategy == "email_reply_click"

    candidates3 = [row, {"tag": "textarea", "role": "textbox", "bid": "tb1", "className": "email-body"}]
    d3 = solve_extraction_task({"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}, {"selected_candidate_bid": "rep1"}]}}, ctx1, candidates3, [])
    assert d3.strategy == "email_reply_fill"

    candidates4 = [row, {"text": "Send", "bid": "snd1", "tag": "button", "className": "email-left"}]
    d4 = solve_extraction_task({"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}, {"action": d3.action}]}}, ctx1, candidates4, [])
    assert d4.strategy == "email_reply_send"

    important_intent = parse_extraction_intent("Find the email by Marlie and click the star icon to mark it as important.")
    row2 = {"text": "From: Marlie Subject: X", "role": "row", "bid": "r2"}
    candidates_star = [row2, {"text": "star", "bid": "st1", "className": "star", "parent_bid": "r2"}]
    ctx2 = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates_star)
    ds = solve_extraction_task(important_intent, ctx2, candidates_star, [])
    assert ds.strategy == "email_star_click"


def test_navigate_tree_exact_node_click():
    candidates = [{"text": "Node A", "role": "treeitem", "bid": "t1"}, {"text": "Node B", "role": "treeitem", "bid": "t2"}]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_tree_node", "constraints": {"quoted_targets": ["Node B"]}}, ctx, candidates, [])
    assert d and d.selected_candidate and d.selected_candidate.get("bid") == "t2"
