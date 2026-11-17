import json

# Bandit findings at these severities make the overall status "vulnerable"
BLOCKING_SEVERITIES = {"MEDIUM", "HIGH"}


def aggregate_results_all(ml_results, non_ml_results):
    """
    Aggregates ML model results and non-ML tool results into a structured final output.
    """
    final_output = {
        "ml_results_summary": [],
        "non_ml_results_summary": [],
    }

    # Process ML results
    for vuln_type, result in ml_results.items():
        final_output["ml_results_summary"].append({
            "vulnerability": vuln_type,
            "status": result["label"],
            "score": result.get("score"),
        })

    # Process Non-ML tool results (Bandit findings)
    if isinstance(non_ml_results, str):
        try:
            non_ml_results = json.loads(non_ml_results)
        except Exception as e:
            non_ml_results = [{"error": f"Failed to parse Bandit output: {str(e)}"}]

    final_output["non_ml_results_summary"] = non_ml_results

    # Overall status: vulnerable if any model flags it, or Bandit reports a MEDIUM/HIGH issue.
    # Bandit errors and LOW-severity findings are shown but don't change the verdict.
    ml_flagged = any(ml["status"] == "vulnerable" for ml in final_output["ml_results_summary"])
    bandit_flagged = any(
        f.get("issue_severity") in BLOCKING_SEVERITIES
        for f in non_ml_results
        if isinstance(f, dict)
    )
    final_output["overall_status"] = "vulnerable" if (ml_flagged or bandit_flagged) else "secure"

    return final_output
