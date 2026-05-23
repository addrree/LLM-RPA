from app.browsergym_integration.miniwob_policy import MiniWoBDeterministicPolicy, unwrap_ax_value
from app.browsergym_integration.miniwob_grounding import ground_miniwob_action


def test_click_button_sequence_next_label():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "1", "role": "button", "text": "ONE"}, {"bid": "2", "role": "button", "text": "TWO"}]
    h = [{"action": 'click("1", "left")', "reward": 0, "selected_candidate_text": "one"}]
    r = p.try_act(env_id="browsergym/miniwob.click-button-sequence", task_name="click-button-sequence", instruction="Click button ONE, then click button TWO.", candidates=cands, history=h, action_syntax=[])
    assert r and '"2"' in r.action


def test_click_dialog_prefers_close_button():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "1", "role": "generic", "text": "x"}, {"bid": "2", "role": "button", "text": "Close"}]
    r = p.try_act(env_id="browsergym/miniwob.click-dialog", task_name="click-dialog", instruction='Close the dialog box by clicking the "x".', candidates=cands, history=[], action_syntax=[])
    assert r and '"2"' in r.action


def test_click_link_policy():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "9", "role": "generic", "tag": "a", "text": "Nulla.", "href": "/elit"}]
    r = p.try_act(env_id="browsergym/miniwob.click-link", task_name="click-link", instruction='Click on the link "Nulla.".', candidates=cands, history=[], action_syntax=[])
    assert r and '"9"' in r.action


def test_click_link_policy_dom_only_uses_mouse_click():
    p = MiniWoBDeterministicPolicy()
    cands = [{"source": "dom", "role": "generic", "tag": "a", "text": "sed.", "href": "/elit", "visible": True, "bbox": {"x": 1}, "browsergym_center_x": 20, "browsergym_center_y": 30}]
    r = p.try_act(env_id="browsergym/miniwob.click-link", task_name="click-link", instruction='Click on the link "sed.".', candidates=cands, history=[], action_syntax=[])
    assert r and r.action.startswith("mouse_click(")

def test_click_link_rejects_empty_generic_candidates():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "11", "role": "generic", "name": "", "text": ""}]
    r = p.try_act(env_id="browsergym/miniwob.click-link", task_name="click-link", instruction='Click on the link "fames".', candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_error == "link_target_not_found"


def test_link_grounding_allows_explicit_bid_when_no_link_candidates():
    res = ground_miniwob_action(action='click("5", "left")', parsed_response={"miniwob_instruction": 'Click on the link "foo".'}, candidates=[{"bid": "5", "role": "button", "text": "foo"}], history=[])
    assert res.action.startswith("click(")


def test_choose_date_policy_steps():
    p = MiniWoBDeterministicPolicy()
    r1 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "d", "role": "textbox", "text": "Date"}], history=[], action_syntax=[])
    assert r1 and '"d"' in r1.action
    r2 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "22", "role": "button", "text": "22", "className": "ui-state-default"}, {"bid": "h", "text": "October 2016", "className": "ui-datepicker-title"}], history=[], action_syntax=[])
    assert r2 and '"22"' in r2.action
    r3 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "s", "role": "button", "text": "Submit"}, {"bid": "d", "role": "textbox", "text": "Date", "value": "10/22/2016"}], history=[{"selected_candidate_text": "22", "mapping_strategy": "policy_choose_date_day"}], action_syntax=[])
    assert r3 and '"s"' in r3.action


def test_choose_date_day_dom_only_uses_mouse_click():
    p = MiniWoBDeterministicPolicy()
    cands = [{"source": "dom", "role": "generic", "tag": "a", "text": "22", "className": "ui-state-default", "visible": True, "bbox": {"x": 1}, "browsergym_center_x": 21, "browsergym_center_y": 22}, {"text": "October 2016", "className": "ui-datepicker-title"}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_choose_date_day" and r.action.startswith("mouse_click(")


def test_use_autocomplete_policy_flow():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "Ch" and ends with "hile".'
    r1 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "t", "role": "textbox", "text": "Tags"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[], action_syntax=[])
    assert r1 and r1.action == 'fill("t", "Ch")'
    r2 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "l1", "role": "option", "text": "Chile"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[{"action": r1.action}], action_syntax=[])
    assert r2 and '"l1"' in r2.action
    r3 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "s", "role": "button", "text": "Submit"}], history=[{"selected_candidate_role": "option"}], action_syntax=[])
    assert r3 and '"s"' in r3.action


def test_use_autocomplete_blocks_repeated_fill_and_fails_early():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "Fi".'
    cands = [{"bid": "t", "role": "textbox", "text": "Tags", "value": "Fi"}]
    r1 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r1 and r1.mapping_strategy == "policy_use_autocomplete_wait_suggestions" and r1.action == "noop()"
    r2 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=cands, history=[{"mapping_strategy": r1.mapping_strategy}], action_syntax=[])
    assert r2 and r2.mapping_error == "autocomplete_suggestions_not_found"
    assert r2.mapping_strategy == "policy_use_autocomplete_not_found"


def test_use_autocomplete_uses_real_text_not_role_value():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "De" and ends with "ark".'
    cands = [{"bid": "l1", "role": {"value": "listitem"}, "text": {"value": "listitem"}, "innerText": "Denmark", "className": "ui-menu-item"}]
    r = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=cands, history=[{"mapping_strategy": "policy_use_autocomplete_fill_prefix"}], action_syntax=[])
    assert r and r.mapping_strategy == "policy_use_autocomplete_pick" and '"l1"' in r.action


def test_choose_date_month_navigation():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/10/2016 as the date and hit submit."
    cands = [{"bid": "n", "text": "Next", "className": "ui-datepicker-next"}, {"bid": "h", "text": "May 2016", "className": "ui-datepicker-title"}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r and '"n"' in r.action and r.mapping_strategy == "policy_choose_date_next_month"


def test_choose_date_ignores_other_month_days():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 05/27/2016 as the date and hit submit."
    cands = [{"bid": "x", "role": "button", "text": "27", "className": "ui-state-default other-month"}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r is None


def test_choose_date_fill_stops_after_one_attempt_without_progress():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/10/2016 as the date and hit submit."
    cands = [{"bid": "d", "role": "textbox", "text": "Date"}]
    r1 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=["fill(bid, value)"])
    assert r1 and r1.mapping_strategy == "policy_choose_date_open"
    r2 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[{"mapping_strategy": "policy_choose_date_fill"}], action_syntax=["fill(bid, value)"])
    assert r2 and r2.mapping_strategy == "policy_choose_date_open"


def test_choose_date_prefers_hasdatepicker_input():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/19/2016 as the date and hit submit."
    cands = [{"bid": "17", "role": "textbox", "className": "hasDatepicker", "text": "Date field"}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_choose_date_open" and '"17"' in r.action


def test_choose_date_stops_if_datepicker_not_opened_after_two_clicks():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/19/2016 as the date and hit submit."
    cands = [{"bid": "17", "role": "textbox", "className": "hasDatepicker", "text": "Date field"}]
    history = [{"mapping_strategy": "policy_choose_date_open"}, {"mapping_strategy": "policy_choose_date_open"}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=history, action_syntax=[])
    assert r and r.mapping_strategy == "policy_choose_date_no_datepicker" and r.action == "noop()"


def test_book_flight_repeated_search_no_progress():
    p = MiniWoBDeterministicPolicy()
    instruction = "Book the cheapest one-way flight from: A to: B on 11/29/2016."
    cands = [{"bid": "f", "role": "textbox", "text": "From"}, {"bid": "t", "role": "textbox", "text": "To"}, {"bid": "d", "role": "textbox", "text": "Date"}, {"bid": "s", "role": "button", "text": "Search"}]
    h = [
        {"action": 'fill("f", "A")'}, {"action": 'fill("t", "B")'}, {"action": 'fill("d", "11/29/2016")'},
        {"action": 'click("s")', "mapping_strategy": "policy_book_flight_search"},
        {"action": 'click("s")', "mapping_strategy": "policy_book_flight_search"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.book-flight", task_name="book-flight", instruction=instruction, candidates=cands, history=h, action_syntax=[])
    assert r and r.mapping_strategy == "policy_book_flight_search_no_progress"


def test_book_flight_policy_produces_date_after_from_to():
    p = MiniWoBDeterministicPolicy()
    instruction = "Book the cheapest one-way flight from: Napaskiak, AK to: SWD on 11/29/2016."
    cands = [
        {"bid": "f", "role": "textbox", "text": "From:"},
        {"bid": "t", "role": "textbox", "text": "To:"},
        {"bid": "d", "role": "textbox", "text": "Departure Date"},
    ]
    h = [{"action": 'fill("f", "Napaskiak, AK")'}, {"action": 'fill("t", "SWD")'}]
    r = p.try_act(env_id="browsergym/miniwob.book-flight", task_name="book-flight", instruction=instruction, candidates=cands, history=h, action_syntax=[])
    assert r and 'fill("d", "11/29/2016")' == r.action


def test_book_flight_policy_does_not_search_without_fields():
    p = MiniWoBDeterministicPolicy()
    instruction = "Book the cheapest one-way flight from: A to: B on 11/29/2016."
    cands = [{"bid": "s", "role": "button", "text": "Search"}]
    r = p.try_act(env_id="browsergym/miniwob.book-flight", task_name="book-flight", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r is None


def test_unwrap_ax_value_and_norm():
    p = MiniWoBDeterministicPolicy()
    assert unwrap_ax_value({"type": "role", "value": "textbox"}) == "textbox"
    assert unwrap_ax_value({"type": "computedString", "value": "Submit"}) == "Submit"
    assert p._norm({"type": "role", "value": "listitem"}) == "listitem"


def test_candidate_texts_extract_nested_ax_fields():
    p = MiniWoBDeterministicPolicy()
    c = {"name": {"type": "computedString", "value": "Tags:"}, "placeholder": {"type": "string", "value": "From:"}}
    texts = p._candidate_texts(c)
    assert "Tags:" in texts
    assert "From:" in texts
