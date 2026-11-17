from scripts.aggregator import aggregate_results_all


def ml(label):
    return {"sql_injection": {"label": label, "score": 0.9 if label == "vulnerable" else 0.1}}


def test_all_clear_is_secure():
    result = aggregate_results_all(ml("secure"), [])
    assert result["overall_status"] == "secure"


def test_ml_detection_is_vulnerable():
    result = aggregate_results_all(ml("vulnerable"), [])
    assert result["overall_status"] == "vulnerable"


def test_bandit_high_severity_is_vulnerable():
    result = aggregate_results_all(ml("secure"), [{"test_id": "B602", "issue_severity": "HIGH"}])
    assert result["overall_status"] == "vulnerable"


def test_bandit_low_severity_and_errors_do_not_flag():
    findings = [{"test_id": "B101", "issue_severity": "LOW"}, {"error": "Bandit failed"}]
    result = aggregate_results_all(ml("secure"), findings)
    assert result["overall_status"] == "secure"
    assert result["non_ml_results_summary"] == findings
