from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord


def test_report_serialization():
    report = BrowserGymRunReport(
        env_id="browsergym/openended",
        goal="Find heading",
        status="partial",
        steps=[BrowserGymStepRecord(step_idx=0, action="noop()")],
    )
    payload = report.model_dump(mode="json")
    assert payload["env_id"] == "browsergym/openended"
    assert payload["steps"][0]["action"] == "noop()"
