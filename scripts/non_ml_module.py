import json
import subprocess
import sys


def run_bandit(file_path):
    """Run Bandit on a file and return its findings, or a list with one {'error': ...} entry."""
    try:
        # Run via the current interpreter so it uses the same virtualenv as the app
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-q", file_path],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return [{"error": "Bandit timed out."}]

    if not result.stdout:
        if result.stderr:
            return [{"error": f"Bandit failed: {result.stderr.strip()[:500]}"}]
        return []

    try:
        findings = json.loads(result.stdout).get("results", [])
    except json.JSONDecodeError as e:
        return [{"error": f"Failed to parse Bandit output: {e}"}]

    # Don't leak the server's temp path into the report
    for f in findings:
        f["filename"] = "submitted code"
    return findings
