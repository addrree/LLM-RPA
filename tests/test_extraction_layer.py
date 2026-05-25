from app.extraction.intent_parser import parse_extraction_intent
from app.extraction.page_extractor import build_extraction_context
from app.extraction.extraction_controller import _find_click_candidate_by_text, solve_extraction_task


def test_numbers_extracted():
    ctx = build_extraction_context({"visible_text": "price 10 and 25"}, {"axtree_excerpt": ""}, [])
    assert any(n["value"] == 10 for n in ctx["numeric_values"])


def test_intent_parser_examples():
    assert parse_extraction_intent("Find the product with the highest rating")["intent"] == "find_max_numeric"
    assert parse_extraction_intent("Find the email marked important")["intent"] == "find_important_email"
    p = parse_extraction_intent('Find the email by Marlie and click the star icon to mark it as important.')
    assert p["intent"] == "find_email"
    assert p["constraints"]["requested_email_action"] == "star"
    f = parse_extraction_intent("Find the email by Daffy and forward that email to Desdemona.")
    assert f["constraints"]["requested_email_action"] == "forward"
    assert f["constraints"]["forward_to"] == "Desdemona"


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


def test_ordinal_word_ignores_ax_metadata_and_splits_wrapper_submit():
    paragraph = "Mi. Risus cras fermentum. Mauris nunc. Enim elit. Augue ipsum. Eu id"
    candidates = [
        {
            "bid": "meta1",
            "tag": "div",
            "name": {"type": "computedString", "value": "", "sources": [{"type": "attribute", "attribute": "title"}]},
            "parent_text": f"{paragraph}\n\n Submit",
        },
        {"tag": "input", "type": "text", "role": "textbox", "bid": "t1", "parent_text": f"{paragraph}\n\n Submit"},
        {"text": "Submit", "innerText": "Submit", "tag": "button", "bid": "s1"},
    ]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    d1 = solve_extraction_task({"intent": "ordinal_word_extraction", "constraints": {"ordinal_index": 5}}, ctx, candidates, [])
    assert d1.strategy == "ordinal_word_fill"
    assert d1.answer == "Mauris"
    assert "computedString" not in d1.diagnostics["paragraph_text"]
    assert d1.selected_candidate.get("bid") == "t1"

    d2 = solve_extraction_task({"intent": "ordinal_word_extraction", "constraints": {"ordinal_index": 5, "history": [{"action": d1.action}]}}, ctx, candidates, [])
    assert d2.strategy == "ordinal_word_submit"
    assert d2.selected_candidate.get("bid") == "s1"


def test_safe_submit_selector_rejects_wrapper():
    candidates = [
        {"text": "4610\nSubmit", "bid": "wrapbid", "id": "wrap", "tag": "div", "bbox": {"width": 400, "height": 400}},
        {"text": "Submit", "bid": "s1", "tag": "button"},
    ]
    assert _find_click_candidate_by_text(candidates, "Submit").get("bid") == "s1"


def test_email_reply_workflow_and_star():
    row = {"text": "From: Winona Subject: Hi", "role": "row", "bid": "r1"}
    candidates1 = [row]
    ctx1 = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates1)
    intent = parse_extraction_intent('Find the email by Winona and reply to them with the text "Sapien sit.".')
    d1 = solve_extraction_task(intent, ctx1, candidates1, [])
    assert d1.strategy == "email_open_row"

    opened_sender = {"text": "Winona", "bid": "sender1", "tag": "div", "className": "email-sender", "parent_class": "email-left"}
    candidates2 = [row, opened_sender, {"text": "Reply", "bid": "rep1", "tag": "button", "className": "email-header"}]
    d2 = solve_extraction_task({"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}]}}, ctx1, candidates2, [])
    assert d2.strategy == "email_reply_click"
    assert d2.selected_candidate.get("bid") == "rep1"

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


def test_email_reply_ignores_compose_container_and_sends_icon():
    intent = parse_extraction_intent('Find the email by Winona and reply to them with the text "Sapien sit.".')
    row = {"text": "Winona\nSubject\nSnippet", "className": "email-thread", "bid": "r1"}
    reply_container = {
        "text": "to: Winona\nsubject: Re: Hi",
        "id": "reply",
        "tag": "div",
        "bid": "reply-container",
        "bbox": {"width": 160, "height": 120},
    }
    textbox = {"tag": "textarea", "role": "textbox", "id": "reply-text", "bid": "tb1", "visible": True}
    reply_sender = {"text": "Winona", "className": "reply-sender", "tag": "span", "bid": "sender-not-send", "visible": True}
    send_icon = {"text": "", "id": "send-reply", "tag": "span", "bid": "send1", "visible": True}
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, [row, reply_container, textbox, reply_sender, send_icon])

    d1 = solve_extraction_task(
        {"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}]}},
        ctx,
        [row, reply_container, textbox, reply_sender, send_icon],
        [],
    )
    assert d1.strategy == "email_reply_fill"
    assert d1.selected_candidate.get("bid") == "tb1"

    d2 = solve_extraction_task(
        {"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}, {"action": d1.action}]}},
        ctx,
        [row, reply_container, textbox, reply_sender, send_icon],
        [],
    )
    assert d2.strategy == "email_reply_send"
    assert d2.selected_candidate.get("bid") == "send1"


def test_email_forward_sequence_fills_recipient_and_sends():
    intent = parse_extraction_intent("Find the email by Daffy and forward that email to Desdemona.")
    row = {"text": "Daffy\nSubject\nSnippet", "className": "email-thread", "bid": "r1"}
    opened_sender = {"text": "Daffy", "bid": "sender1", "tag": "div", "className": "email-sender", "parent_class": "email-left"}
    forward_button = {"text": "Forward", "tag": "span", "className": "email-forward", "bid": "fw1", "visible": True}
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, [row, opened_sender, forward_button])

    d1 = solve_extraction_task(
        {"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}]}},
        ctx,
        [row, opened_sender, forward_button],
        [],
    )
    assert d1.strategy == "email_forward_click"
    assert d1.selected_candidate.get("bid") == "fw1"

    forward_container = {"text": "to:\nsubject: Fwd: Hi", "id": "forward", "tag": "div", "bid": "forward-container"}
    recipient = {"tag": "input", "type": "text", "className": "forward-sender", "bid": "to1", "visible": True}
    body = {"tag": "textarea", "id": "forward-text", "value": "original body", "bid": "body1", "visible": True}
    send_icon = {"id": "send-forward", "tag": "span", "bid": "send1", "visible": True}
    d2 = solve_extraction_task(
        {"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}, {"selected_candidate_bid": "fw1"}]}},
        ctx,
        [row, forward_container, recipient, body, send_icon],
        [],
    )
    assert d2.strategy == "email_forward_fill"
    assert d2.selected_candidate.get("bid") == "to1"
    assert "Desdemona" in d2.action

    d3 = solve_extraction_task(
        {"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r1"}, {"action": d2.action}]}},
        ctx,
        [row, forward_container, recipient, body, send_icon],
        [],
    )
    assert d3.strategy == "email_forward_send"
    assert d3.selected_candidate.get("bid") == "send1"


def test_mark_important_opens_row_then_uses_opened_star_without_refiltering_important():
    intent = parse_extraction_intent("Find the email by Marlie and click the star icon to mark it as important.")
    row = {"text": "Marlie\nSubject\nSnippet", "className": "email-thread", "bid": "r2"}
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, [row])

    d1 = solve_extraction_task(intent, ctx, [row], [])
    assert d1.strategy == "email_open_row"
    assert d1.selected_candidate.get("bid") == "r2"

    opened_sender = {"text": "Marlie", "bid": "sender2", "tag": "div", "className": "email-sender", "parent_class": "email-left"}
    opened_star = {"text": "", "ariaLabel": "Star", "bid": "star2", "tag": "button", "className": "star-icon"}
    d2 = solve_extraction_task({"intent": intent["intent"], "constraints": {**intent["constraints"], "history": [{"selected_candidate_bid": "r2"}]}}, ctx, [opened_sender, opened_star], [])
    assert d2.strategy == "email_star_click"
    assert d2.selected_candidate.get("bid") == "star2"


def test_navigate_tree_exact_node_click():
    candidates = [{"text": "Node A", "role": "treeitem", "bid": "t1"}, {"text": "Node B", "role": "treeitem", "bid": "t2"}]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_tree_node", "constraints": {"quoted_targets": ["Node B"]}}, ctx, candidates, [])
    assert d and d.selected_candidate and d.selected_candidate.get("bid") == "t2"


def test_navigate_tree_expands_parent_when_target_hidden_in_textcontent():
    candidates = [
        {"text": "Nieves", "textContent": "NievesCristinBrianaJess", "tag": "li", "className": "expandable", "bid": "folder1"},
        {"text": "Nieves", "tag": "span", "className": "folder", "bid": "folder-label"},
    ]
    ctx = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, candidates)
    d = solve_extraction_task({"intent": "find_tree_node", "constraints": {"quoted_targets": ["Jess"]}}, ctx, candidates, [])
    assert d.strategy == "tree_expand_toward_target"
    assert d.selected_candidate.get("bid") == "folder1"

    expanded = candidates + [{"text": "Jess", "tag": "li", "className": "file", "bid": "target1"}]
    ctx2 = build_extraction_context({"visible_text": ""}, {"axtree_excerpt": ""}, expanded)
    d2 = solve_extraction_task({"intent": "find_tree_node", "constraints": {"quoted_targets": ["Jess"], "history": [{"selected_candidate_bid": "folder1"}]}}, ctx2, expanded, [])
    assert d2.strategy == "tree_node_click"
    assert d2.selected_candidate.get("bid") == "target1"
