"""
src/api/alert_webhook_proxy.py
Webhook proxy service that:
  1. Receives Alertmanager webhook payloads on POST /alert
  2. Translates them to GitHub Actions workflow_dispatch calls
  3. Triggers the intelligent-retrain CI/CD stage (Stage 4)
  4. Exposes Prometheus metrics on GET /metrics
  5. Runs in safe "local-dev" mode when GITHUB_TOKEN is not configured

Deploy as the `alert-webhook` Docker Compose service (port 5001).
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("alert-webhook")

# ── Configuration ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "Muhammad-Subhan034/fraud-detection-mlops")
GITHUB_REF = os.environ.get("GITHUB_REF", "main")
WORKFLOW_FILE = "fraud-detection-cicd.yml"
GITHUB_API = (
    f"https://api.github.com/repos/{GITHUB_REPO}"
    f"/actions/workflows/{WORKFLOW_FILE}/dispatches"
)

# Local-dev mode: log alerts but don't call GitHub (no token available)
LOCAL_DEV_MODE = (
    not GITHUB_TOKEN or GITHUB_TOKEN == "dummy-for-local-dev"
)
if LOCAL_DEV_MODE:
    logger.warning(
        "⚠️  GITHUB_TOKEN not set — running in LOCAL DEV MODE. "
        "Alerts will be logged but NOT forwarded to GitHub Actions."
    )

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
ALERTS_RECEIVED = Counter(
    "webhook_alerts_received_total",
    "Total Alertmanager webhooks received",
    ["alert_type"],
)
GITHUB_DISPATCHES = Counter(
    "webhook_github_dispatches_total",
    "GitHub Actions workflow_dispatch calls made",
    ["status"],   # success / error / skipped
)
DISPATCH_LATENCY = Histogram(
    "webhook_github_dispatch_latency_seconds",
    "Latency of GitHub API calls",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
LAST_ALERT_TIMESTAMP = Gauge(
    "webhook_last_alert_received_timestamp",
    "Unix timestamp of the most recent alert received",
)


# ── Alert parsing ─────────────────────────────────────────────────────────────
# Maps Prometheus alertname patterns → workflow_dispatch trigger_reason values
ALERT_REASON_MAP = {
    "FraudRecallCritical": "recall_drop",
    "FraudRecallWarning": "recall_drop",
    "DataDriftCritical": "drift_alert",
    "DataDriftDetected": "drift_alert",
    "LowConfidencePredictions": "performance_degradation",
    "APILatencyCritical": "performance_degradation",
    "APIHighErrorRate": "performance_degradation",
    "FraudAPIDown": "performance_degradation",
}


def parse_alert_payload(payload: dict) -> dict:
    """
    Extract trigger_reason, current_recall, and drift_score from
    an Alertmanager webhook payload.

    Alertmanager payload shape:
    {
      "alerts": [
        {
          "status": "firing",
          "labels":      { "alertname": "FraudRecallCritical", ... },
          "annotations": { "value": "0.65", ... }
        }
      ]
    }
    """
    result = {
        "trigger_reason": None,
        "current_recall": None,
        "drift_score": None,
        "alert_names": [],
    }

    for alert in payload.get("alerts", []):
        if alert.get("status") != "firing":
            continue

        name = alert.get("labels", {}).get("alertname", "")
        value = alert.get("annotations", {}).get("value", "")
        result["alert_names"].append(name)

        reason = ALERT_REASON_MAP.get(name)
        if reason and result["trigger_reason"] is None:
            result["trigger_reason"] = reason

        # Attempt to extract numeric value from annotations
        try:
            numeric_val = float(value)
        except (ValueError, TypeError):
            numeric_val = None

        if reason == "recall_drop" and numeric_val is not None:
            result["current_recall"] = numeric_val
        elif reason == "drift_alert" and numeric_val is not None:
            result["drift_score"] = numeric_val

    # Fallback reason if none matched
    if result["alert_names"] and result["trigger_reason"] is None:
        result["trigger_reason"] = "performance_degradation"

    return result


# ── GitHub dispatch ───────────────────────────────────────────────────────────
def dispatch_github_workflow(trigger_reason: str,
                             current_recall: float = None,
                             drift_score: float = None) -> bool:
    """Call GitHub Actions workflow_dispatch API. Returns True on success."""
    inputs = {"trigger_reason": trigger_reason}
    if current_recall is not None:
        inputs["current_recall"] = f"{current_recall:.4f}"
    if drift_score is not None:
        inputs["drift_score"] = f"{drift_score:.4f}"

    payload = {"ref": GITHUB_REF, "inputs": inputs}
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    start = time.time()
    try:
        resp = requests.post(
            GITHUB_API, json=payload, headers=headers, timeout=10
        )
        elapsed = time.time() - start
        DISPATCH_LATENCY.observe(elapsed)

        if resp.status_code == 204:
            logger.info(
                "✅ GitHub Actions triggered | reason=%s recall=%s drift=%s (%.2fs)",
                trigger_reason, current_recall, drift_score, elapsed,
            )
            GITHUB_DISPATCHES.labels(status="success").inc()
            return True
        else:
            logger.error(
                "❌ GitHub API returned %s: %s", resp.status_code, resp.text[:200]
            )
            GITHUB_DISPATCHES.labels(status="error").inc()
            return False

    except requests.Timeout:
        DISPATCH_LATENCY.observe(time.time() - start)
        logger.error("❌ GitHub API call timed out after 10s")
        GITHUB_DISPATCHES.labels(status="error").inc()
        return False
    except Exception as exc:
        DISPATCH_LATENCY.observe(time.time() - start)
        logger.error("❌ Unexpected error calling GitHub API: %s", exc)
        GITHUB_DISPATCHES.labels(status="error").inc()
        return False


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "local_dev_mode": LOCAL_DEV_MODE,
        "github_repo": GITHUB_REPO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/status")
def status():
    """Returns current configuration summary (no secrets exposed)."""
    return jsonify({
        "mode": "local-dev" if LOCAL_DEV_MODE else "production",
        "github_repo": GITHUB_REPO,
        "github_ref": GITHUB_REF,
        "workflow_file": WORKFLOW_FILE,
        "alert_map": list(ALERT_REASON_MAP.keys()),
    })


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.post("/alert")
def receive_alert():
    """
    Primary endpoint: called by Alertmanager when an alert fires.
    Parses the payload and dispatches to GitHub Actions.
    """
    payload = request.get_json(silent=True)
    if not payload:
        logger.warning("Received empty or non-JSON payload")
        return jsonify({"error": "Empty or invalid JSON payload"}), 400

    LAST_ALERT_TIMESTAMP.set(time.time())

    alert_info = parse_alert_payload(payload)
    alert_names = alert_info["alert_names"]
    trigger_reason = alert_info["trigger_reason"]

    logger.info(
        "Alert received | alerts=%s reason=%s recall=%s drift=%s",
        alert_names, trigger_reason,
        alert_info["current_recall"], alert_info["drift_score"],
    )

    # Track counters per alert type
    for name in alert_names:
        ALERTS_RECEIVED.labels(alert_type=name).inc()

    if trigger_reason is None:
        logger.info("No actionable trigger in this alert batch — ignoring")
        GITHUB_DISPATCHES.labels(status="skipped").inc()
        return jsonify({"dispatched": False, "reason": "no_actionable_alert"}), 200

    if LOCAL_DEV_MODE:
        logger.info(
            "LOCAL DEV: would dispatch reason=%s recall=%s drift=%s",
            trigger_reason, alert_info["current_recall"], alert_info["drift_score"],
        )
        GITHUB_DISPATCHES.labels(status="skipped").inc()
        return jsonify({
            "dispatched": False,
            "reason": "local_dev_mode",
            "would_send": {
                "trigger_reason": trigger_reason,
                "current_recall": alert_info["current_recall"],
                "drift_score": alert_info["drift_score"],
            },
        }), 200

    success = dispatch_github_workflow(
        trigger_reason=trigger_reason,
        current_recall=alert_info["current_recall"],
        drift_score=alert_info["drift_score"],
    )

    if success:
        return jsonify({
            "dispatched": True,
            "trigger_reason": trigger_reason,
            "alert_names": alert_names,
        }), 200
    else:
        return jsonify({
            "dispatched": False,
            "error": "GitHub API call failed — see logs",
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
