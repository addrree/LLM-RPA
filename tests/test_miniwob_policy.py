from app.browsergym_integration.miniwob_policy import MiniWoBDeterministicPolicy
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
    cands = [{"bid": "9", "role": "link", "tag": "a", "text": "elit", "href": "/elit"}]
    r = p.try_act(env_id="browsergym/miniwob.click-link", task_name="click-link", instruction='Click on the link "elit".', candidates=cands, history=[], action_syntax=[])
    assert r and '"9"' in r.action


def test_link_grounding_allows_explicit_bid_when_no_link_candidates():
    res = ground_miniwob_action(action='click("5", "left")', parsed_response={"miniwob_instruction": 'Click on the link "foo".'}, candidates=[{"bid": "5", "role": "button", "text": "foo"}], history=[])
    assert res.action.startswith("click(")


def test_choose_date_policy_steps():
    p = MiniWoBDeterministicPolicy()
    r1 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "d", "role": "textbox", "text": "Date"}], history=[], action_syntax=[])
    assert r1 and '"d"' in r1.action
    r2 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "22", "role": "button", "text": "22"}], history=[], action_syntax=[])
    assert r2 and '"22"' in r2.action
    r3 = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction="Select 10/22/2016 as the date and hit submit.", candidates=[{"bid": "s", "role": "button", "text": "Submit"}], history=[{"selected_candidate_text": "22"}], action_syntax=[])
    assert r3 and '"s"' in r3.action


def test_use_autocomplete_policy_flow():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "Ch" and ends with "hile".'
    r1 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "t", "role": "textbox", "text": "Tags"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[], action_syntax=[])
    assert r1 and r1.action == 'fill("t", "Ch")'
    r2 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "l1", "role": "option", "text": "Chile"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[{"action": r1.action}], action_syntax=[])
    assert r2 and '"l1"' in r2.action
    r3 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "s", "role": "button", "text": "Submit"}], history=[{"selected_candidate_role": "option"}], action_syntax=[])
    assert r3 and '"s"' in r3.action


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
