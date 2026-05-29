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


def test_generic_click_button_selects_real_button_not_wrapper():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "wrap", "bid_source": "bid", "id": "wrap", "tag": "div", "text": "Click me\nSubmit", "parent_tag": "body"},
        {"bid": "22", "bid_source": "bid", "tag": "button", "text": "Submit"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.click-button", task_name="click-button", instruction='Click on the "Submit" button.', candidates=cands, history=[], action_syntax=[])
    assert r and r.action == 'click("22", "left")'
    assert r.mapping_strategy == "policy_basic_button_click"


def test_generic_enter_text_fills_then_submits():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "t", "tag": "input", "type": "text"}, {"bid": "s", "tag": "button", "text": "Submit"}]
    r1 = p.try_act(env_id="browsergym/miniwob.enter-text", task_name="enter-text", instruction='Enter "Cristin" into the text field and press Submit.', candidates=cands, history=[], action_syntax=[])
    assert r1 and r1.action == 'fill("t", "Cristin")'
    r2 = p.try_act(env_id="browsergym/miniwob.enter-text", task_name="enter-text", instruction='Enter "Cristin" into the text field and press Submit.', candidates=cands, history=[{"action": r1.action}], action_syntax=[])
    assert r2 and r2.action == 'click("s", "left")'


def test_generic_checkbox_flow_selects_each_target_then_submit():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "a", "tag": "label", "text": "Alpha"},
        {"bid": "b", "tag": "label", "text": "Beta"},
        {"bid": "s", "tag": "button", "text": "Submit"},
    ]
    instr = "Select Alpha, Beta and click Submit."
    r1 = p.try_act(env_id="browsergym/miniwob.click-checkboxes", task_name="click-checkboxes", instruction=instr, candidates=cands, history=[], action_syntax=[])
    assert r1 and r1.action == 'click("a", "left")'
    r2 = p.try_act(env_id="browsergym/miniwob.click-checkboxes", task_name="click-checkboxes", instruction=instr, candidates=cands, history=[{"selected_candidate_text": "Alpha"}], action_syntax=[])
    assert r2 and r2.action == 'click("b", "left")'
    r3 = p.try_act(env_id="browsergym/miniwob.click-checkboxes", task_name="click-checkboxes", instruction=instr, candidates=cands, history=[{"selected_candidate_text": "Alpha"}, {"selected_candidate_text": "Beta"}], action_syntax=[])
    assert r3 and r3.action == 'click("s", "left")'


def test_generic_checkbox_ignores_parent_text_for_wrong_label():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "first", "tag": "label", "text": "rNbYEh9", "parent_text": "rNbYEh9\nYC\na1\nWgY\nsvK"},
        {"bid": "target", "tag": "label", "text": "a1", "parent_text": "rNbYEh9\nYC\na1\nWgY\nsvK"},
        {"bid": "s", "tag": "button", "text": "Submit"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.click-checkboxes", task_name="click-checkboxes", instruction="Select a1, WgY, svK and click Submit.", candidates=cands, history=[], action_syntax=[])
    assert r and r.action == 'click("target", "left")'


def test_generic_scroll_list_uses_select_option_then_submit():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "sel", "tag": "select", "text": "Mollee\nVittoria\nMarlo", "value": "Mollee"},
        {"bid": "s", "tag": "button", "text": "Submit"},
    ]
    instruction = "Select Vittoria from the scroll list and click Submit."
    r1 = p.try_act(env_id="browsergym/miniwob.click-scroll-list", task_name="click-scroll-list", instruction=instruction, candidates=cands, history=[], action_syntax=['select_option("bid", ["option_text"])'])
    assert r1 and r1.action == 'select_option("sel", ["Vittoria"])'
    r2 = p.try_act(env_id="browsergym/miniwob.click-scroll-list", task_name="click-scroll-list", instruction=instruction, candidates=[{**cands[0], "value": "Vittoria"}, cands[1]], history=[{"action": r1.action}], action_syntax=['select_option("bid", ["option_text"])'])
    assert r2 and r2.action == 'click("s", "left")'


def test_generic_scroll_list_selects_multiple_options_together():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "sel", "tag": "select", "text": "Somalia\nMonaco\nNauru", "value": ""},
        {"bid": "s", "tag": "button", "text": "Submit"},
    ]
    instruction = "Select Somalia, Monaco from the scroll list and click Submit."
    r1 = p.try_act(env_id="browsergym/miniwob.click-scroll-list", task_name="click-scroll-list", instruction=instruction, candidates=cands, history=[], action_syntax=['select_option("bid", ["option_text"])'])
    assert r1 and r1.action == 'select_option("sel", ["Somalia", "Monaco"])'
    r2 = p.try_act(env_id="browsergym/miniwob.click-scroll-list", task_name="click-scroll-list", instruction=instruction, candidates=cands, history=[{"action": r1.action}], action_syntax=['select_option("bid", ["option_text"])'])
    assert r2 and r2.action == 'click("s", "left")'


def test_generic_login_flow_fills_username_password_then_login():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "u", "tag": "input", "type": "text"},
        {"bid": "p", "tag": "input", "type": "password"},
        {"bid": "l", "tag": "button", "text": "Login"},
    ]
    instr = 'Enter the username "jess" and the password "S2" into the text fields and press login.'
    r1 = p.try_act(env_id="browsergym/miniwob.login-user", task_name="login-user", instruction=instr, candidates=cands, history=[], action_syntax=[])
    assert r1 and r1.action == 'fill("u", "jess")'
    r2 = p.try_act(env_id="browsergym/miniwob.login-user", task_name="login-user", instruction=instr, candidates=cands, history=[{"action": r1.action}], action_syntax=[])
    assert r2 and r2.action == 'fill("p", "S2")'
    r3 = p.try_act(env_id="browsergym/miniwob.login-user", task_name="login-user", instruction=instr, candidates=cands, history=[{"action": r1.action}, {"action": r2.action}], action_syntax=[])
    assert r3 and r3.action == 'click("l", "left")'


def test_generic_tabs_and_collapsible_probe_before_link_click():
    p = MiniWoBDeterministicPolicy()
    tab_cands = [{"bid": "t1", "tag": "a", "text": "Tab #1"}, {"bid": "t2", "tag": "a", "text": "Tab #2"}]
    r = p.try_act(env_id="browsergym/miniwob.click-tab-2", task_name="click-tab-2", instruction='Switch between the tabs to find and click on the link "target".', candidates=tab_cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_basic_tab_probe"
    link_r = p.try_act(env_id="browsergym/miniwob.click-tab-2", task_name="click-tab-2", instruction='Switch between the tabs to find and click on the link "target".', candidates=[{"bid": "lnk", "tag": "span", "className": "alink", "text": "target"}], history=[], action_syntax=[])
    assert link_r and link_r.action == 'click("lnk", "left")'


def test_tab_link_match_rejects_panel_substring_wrapper():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "panel", "tag": "div", "role": "tabpanel", "text": "Large paragraph with Magna somewhere inside."},
        {"bid": "tab2", "tag": "a", "text": "Tab #2"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.click-tab-2", task_name="click-tab-2", instruction='Switch between the tabs to find and click on the link "Magna".', candidates=cands, history=[{"selected_candidate_text": "Tab #1"}], action_syntax=[])
    assert r and r.mapping_strategy == "policy_basic_tab_probe"
    assert r.action == 'click("tab2", "left")'


def test_short_link_target_requires_exact_match():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "wrong", "tag": "span", "className": "alink", "text": "consectetur"},
        {"bid": "next", "tag": "h3", "role": "tab", "text": "Section #2"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.click-collapsible-2", task_name="click-collapsible-2", instruction='Expand the sections below, to find and click on the link "et".', candidates=cands, history=[{"selected_candidate_text": "Section #1"}], action_syntax=[])
    assert r and r.mapping_strategy == "policy_basic_collapsible_probe"
    assert r.action == 'click("next", "left")'


def test_click_button_sequence_uses_safe_point_for_overlapping_buttons():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "one", "tag": "button", "text": "ONE", "visible": True, "bbox": {"left": 21, "top": 121, "width": 40, "height": 40}, "page_center_x": 41, "page_center_y": 141, "browsergym_center_x": 61.5, "browsergym_center_y": 211.5},
        {"bid": "two", "tag": "button", "text": "TWO", "visible": True, "bbox": {"left": 31, "top": 115, "width": 40, "height": 40}, "page_center_x": 51, "page_center_y": 135, "browsergym_center_x": 76.5, "browsergym_center_y": 202.5},
    ]
    r = p.try_act(env_id="browsergym/miniwob.click-button-sequence", task_name="click-button-sequence", instruction="Click button ONE, then click button TWO.", candidates=cands, history=[], action_syntax=['mouse_click(x, y, "left")'])
    assert r and r.mapping_strategy == "policy_basic_button_sequence"
    assert r.action.startswith("mouse_click(")
    assert (r.mapping_diagnostics or {}).get("click_strategy") == "safe_mouse_click"
    assert (r.mapping_diagnostics or {}).get("page_x") != 41


def test_click_menu_stops_after_hover_without_submenu():
    p = MiniWoBDeterministicPolicy()
    cands = [{"bid": "m1", "tag": "div", "role": "menuitem", "text": "Berna", "className": "ui-menu-item-wrapper", "visible": True}]
    instruction = "Select Berna>Fernanda>Layne"
    r = p.try_act(env_id="browsergym/miniwob.click-menu", task_name="click-menu", instruction=instruction, candidates=cands, history=[{"mapping_strategy": "policy_click_menu_hover_parent"}], action_syntax=["mouse_move(x, y)", 'click("bid", "left")'])
    assert r and r.action == "noop()"
    assert r.mapping_strategy == "policy_click_menu_hover_required"
    assert r.mapping_error == "menu_requires_hover_no_supported_action"


def test_click_menu_hovers_next_visible_submenu_level():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"bid": "m1", "tag": "div", "role": "menuitem", "text": "Berna", "className": "ui-menu-item-wrapper ui-state-active", "visible": True, "browsergym_center_x": 20, "browsergym_center_y": 30},
        {"bid": "m2", "tag": "div", "role": "menuitem", "text": "Fernanda", "className": "ui-menu-item-wrapper", "visible": True, "browsergym_center_x": 80, "browsergym_center_y": 30},
    ]
    instruction = "Select Berna>Fernanda>Layne"
    r = p.try_act(
        env_id="browsergym/miniwob.click-menu",
        task_name="click-menu",
        instruction=instruction,
        candidates=cands,
        history=[{"mapping_strategy": "policy_click_menu_hover_parent", "selected_candidate_text": "Berna"}],
        action_syntax=["mouse_move(x, y)", 'click("bid", "left")'],
    )
    assert r and r.mapping_strategy == "policy_click_menu_hover_parent"
    assert r.selected_candidate and r.selected_candidate["text"] == "Fernanda"
    assert r.action == "mouse_move(80, 30)"


def test_grid_coordinate_clicks_exact_svg_point_candidate():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"tag": "circle", "id": "(1,0)", "className": "plot-point", "visible": True, "browsergym_center_x": 160.5, "browsergym_center_y": 190.5, "page_center_x": 107, "page_center_y": 127},
        {"tag": "circle", "id": "(2,0)", "className": "plot-point", "visible": True, "browsergym_center_x": 205.5, "browsergym_center_y": 190.5},
    ]
    r = p.try_act(env_id="browsergym/miniwob.grid-coordinate", task_name="grid-coordinate", instruction="Click on the grid coordinate (1,0).", candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_grid_coordinate_point"
    assert r.action == 'mouse_click(160, 190, "left")'
    assert (r.mapping_diagnostics or {}).get("target_id") == "(1,0)"


def test_grid_coordinate_uses_svg_geometry_fallback():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"tag": "svg", "visible": True, "bbox": {"x": 2, "y": 52, "width": 150, "height": 150}, "page_center_x": 77, "page_center_y": 127, "browsergym_center_x": 115.5, "browsergym_center_y": 190.5},
    ]
    r = p.try_act(env_id="browsergym/miniwob.grid-coordinate", task_name="grid-coordinate", instruction="Click on the grid coordinate (-2,2).", candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_grid_coordinate_geometry"
    assert r.action == 'mouse_click(26, 100, "left")'
    assert (r.mapping_diagnostics or {}).get("page_x") == 17
    assert (r.mapping_diagnostics or {}).get("page_y") == 67


def test_count_shape_counts_svg_items_and_clicks_answer_button():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"tag": "polygon", "parent_tag": "svg", "fill": "red", "visible": True, "bbox": {"left": 10, "top": 10, "width": 20, "height": 20}},
        {"tag": "polygon", "parent_tag": "svg", "fill": "red", "visible": True, "bbox": {"left": 40, "top": 10, "width": 20, "height": 20}},
        {"tag": "polygon", "parent_tag": "svg", "fill": "blue", "visible": True, "bbox": {"left": 70, "top": 10, "width": 20, "height": 20}},
        {"tag": "rect", "parent_tag": "svg", "fill": "red", "visible": True, "bbox": {"left": 10, "top": 40, "width": 20, "height": 20}},
        {"bid": "b1", "tag": "button", "role": "button", "text": "1", "visible": True},
        {"bid": "b2", "tag": "button", "role": "button", "text": "2", "visible": True},
    ]
    r = p.try_act(env_id="browsergym/miniwob.count-shape", task_name="count-shape", instruction="How many large red triangles are there?", candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_count_shape_answer"
    assert r.action == 'click("b2", "left")'
    assert (r.mapping_diagnostics or {}).get("matched_count") == 2


def test_count_shape_handles_exact_letter_plural():
    p = MiniWoBDeterministicPolicy()
    cands = [
        {"tag": "text", "parent_tag": "svg", "textContent": "Q", "fill": "green", "fontSize": "20px", "visible": True, "bbox": {"left": 10, "top": 10, "width": 12, "height": 20}},
        {"tag": "text", "parent_tag": "svg", "textContent": "q", "fill": "green", "fontSize": "10px", "visible": True, "bbox": {"left": 30, "top": 10, "width": 7, "height": 10}},
        {"tag": "text", "parent_tag": "svg", "textContent": "5", "fill": "green", "fontSize": "20px", "visible": True, "bbox": {"left": 50, "top": 10, "width": 10, "height": 20}},
        {"bid": "b2", "tag": "button", "role": "button", "text": "2", "visible": True},
    ]
    r = p.try_act(env_id="browsergym/miniwob.count-shape", task_name="count-shape", instruction="How many green Qs are there?", candidates=cands, history=[], action_syntax=[])
    assert r and r.action == 'click("b2", "left")'


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

def test_choose_date_day_works_when_header_year_missing_if_day_present():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/28/2016 as the date and hit submit."
    cands = [
        {"bid": "h", "text": "June", "className": "ui-datepicker-title"},
        {"bid": "d28", "role": "button", "name": "28", "className": "ui-state-default"},
    ]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_choose_date_day"
    assert '"d28"' in r.action


def test_use_autocomplete_policy_flow():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "Ch" and ends with "hile".'
    r1 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "t", "role": "textbox", "text": "Tags"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[], action_syntax=[])
    assert r1 and r1.action == 'fill("t", "Ch")'
    r2 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "l1", "role": "option", "text": "Chile"}, {"bid": "s", "role": "button", "text": "Submit"}], history=[{"action": r1.action}], action_syntax=[])
    assert r2 and '"l1"' in r2.action
    r3 = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=[{"bid": "s", "role": "button", "text": "Submit"}], history=[{"selected_candidate_role": "option"}], action_syntax=[])
    assert r3 and '"s"' in r3.action


def test_use_autocomplete_detects_plain_input_tag():
    p = MiniWoBDeterministicPolicy()
    instruction = 'Enter an item that starts with "Alan".'
    cands = [{"bid": "tags", "tag": "input", "id": "tags", "className": "ui-autocomplete-input"}, {"bid": "s", "tag": "button", "text": "Submit"}]
    r = p.try_act(env_id="browsergym/miniwob.use-autocomplete", task_name="use-autocomplete", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r and r.action == 'fill("tags", "Alan")'


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


def test_choose_date_detects_date_input_without_role():
    p = MiniWoBDeterministicPolicy()
    instruction = "Select 06/28/2016 as the date and hit submit."
    cands = [{"bid": "17", "tag": "input", "type": "text", "id": "datepicker", "className": "hasDatepicker", "role": ""}]
    r = p.try_act(env_id="browsergym/miniwob.choose-date", task_name="choose-date", instruction=instruction, candidates=cands, history=[], action_syntax=[])
    assert r and r.mapping_strategy == "policy_choose_date_open"
    assert r.action == 'click("17", "left")'
    assert (r.mapping_diagnostics or {}).get("date_input_bid") == "17"


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
