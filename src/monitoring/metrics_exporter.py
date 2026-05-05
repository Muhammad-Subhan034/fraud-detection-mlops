"""
Model Metrics Exporter
Runs alongside the inference API and pushes model-level metrics to Prometheus.
Tracks: recall, FPR, PSI scores, missing rates, transaction distributions.
"""

import os
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prometheus_client import (
    start_http_server, Gauge, Counter, Histogram,
    CollectorRegistry, push_to_gateway
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Metrics ----
FEATURE_PSI         = Gauge("fraud_feature_psi_score", "Overall feature PSI drift score")
DRIFTED_FEATURES    = Gauge("fraud_drifted_features_count", "Number of features with PSI > threshold")
MISSING_RATE        = Gauge("fraud_input_missing_rate", "Rate of missing values in incoming data")
TX_AMOUNT_MEAN      = Gauge("fraud_transaction_amount_mean", "Mean transaction amount (rolling window)")
TX_AMOUNT_PSI       = Gauge("fraud_amount_psi_score", "PSI for TransactionAmt feature")
CARD_FEATURES_PSI   = Gauge("fraud_card_features_psi", "Mean PSI for card-related features")
MODEL_RECALL        = Gauge("fraud_model_recall", "Current fraud model recall estimate")
MODEL_FPR           = Gauge("fraud_false_positive_rate", "Current false positive rate")
TX_AMOUNT_HIST      = Histogram(
    "fraud_transaction_amount",
    "Transaction amount distribution",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)


class ModelMetricsExporter:
    """
    Computes and exports model/data metrics to Prometheus.
    Uses a sliding window of recent predictions + ground truth (when available).
    """

    def __init__(self, reference_data_path: str = None, window_hours: int = 1):
        self.window_hours = window_hours
        self.predictions_buffer = []   # (timestamp, prob, ground_truth_if_known)
        self.features_buffer    = []   # recent feature rows

        if reference_data_path and os.path.exists(reference_data_path):
            self.reference = pd.read_parquet(reference_data_path)
            logger.info(f"Loaded reference data: {self.reference.shape}")
        else:
            self.reference = None
            logger.warning("No reference data — PSI metrics will be zero")

    def record_prediction(self, features: dict, prob: float, ground_truth: int = None):
        """Called after every inference."""
        self.predictions_buffer.append({
            "timestamp": datetime.utcnow(),
            "prob": prob,
            "ground_truth": ground_truth,
        })
        self.features_buffer.append(features)

        # Record to histogram
        if "TransactionAmt" in features:
            TX_AMOUNT_HIST.observe(features["TransactionAmt"])
            TX_AMOUNT_MEAN.set(features["TransactionAmt"])

        # Trim old entries
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        self.predictions_buffer = [p for p in self.predictions_buffer if p["timestamp"] > cutoff]
        self.features_buffer    = self.features_buffer[-10000:]  # Keep last 10k

    def compute_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        eps = 1e-8
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()) + eps,
            buckets + 1
        )
        exp_pct = np.histogram(expected, breakpoints)[0] / len(expected) + eps
        act_pct = np.histogram(actual, breakpoints)[0] / len(actual) + eps
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def update_drift_metrics(self):
        """Compute PSI for recent vs reference data."""
        if self.reference is None or len(self.features_buffer) < 100:
            return

        current_df = pd.DataFrame(self.features_buffer[-2000:])
        psi_scores = []
        card_psi   = []

        numeric_cols = self.reference.select_dtypes(include=np.number).columns[:30]
        for col in numeric_cols:
            if col in current_df.columns and current_df[col].notna().sum() > 50:
                try:
                    psi = self.compute_psi(
                        self.reference[col].dropna(),
                        current_df[col].dropna()
                    )
                    psi_scores.append(psi)
                    if col.startswith("card"):
                        card_psi.append(psi)
                except Exception:
                    pass

        if psi_scores:
            max_psi = max(psi_scores)
            FEATURE_PSI.set(max_psi)
            DRIFTED_FEATURES.set(sum(p > 0.2 for p in psi_scores))

        if card_psi:
            CARD_FEATURES_PSI.set(np.mean(card_psi))

        # Transaction amount PSI
        if "TransactionAmt" in current_df.columns and "TransactionAmt" in self.reference.columns:
            amt_psi = self.compute_psi(
                self.reference["TransactionAmt"].dropna(),
                current_df["TransactionAmt"].dropna()
            )
            TX_AMOUNT_PSI.set(amt_psi)

        # Missing rate
        if current_df.shape[0] > 0:
            missing_rate = current_df.isnull().mean().mean()
            MISSING_RATE.set(float(missing_rate))

    def update_performance_metrics(self):
        """
        Estimate recall/FPR from labeled predictions in window.
        Only possible when ground truth labels are available (feedback loop).
        """
        labeled = [p for p in self.predictions_buffer if p["ground_truth"] is not None]
        if len(labeled) < 50:
            # Use placeholder — real system would get labels via feedback loop
            return

        y_true = np.array([p["ground_truth"] for p in labeled])
        y_pred = np.array([1 if p["prob"] >= 0.5 else 0 for p in labeled])

        if y_true.sum() > 0:
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()

            recall = tp / (tp + fn + 1e-8)
            fpr    = fp / (fp + tn + 1e-8)

            MODEL_RECALL.set(float(recall))
            MODEL_FPR.set(float(fpr))

    def run_update_loop(self, interval_seconds: int = 60):
        """Background thread: update metrics every interval."""
        while True:
            try:
                self.update_drift_metrics()
                self.update_performance_metrics()
                logger.debug("Metrics updated")
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
            time.sleep(interval_seconds)

    def start_background_updates(self, interval_seconds: int = 60):
        t = threading.Thread(
            target=self.run_update_loop,
            args=(interval_seconds,),
            daemon=True
        )
        t.start()
        logger.info(f"Metrics exporter started (interval={interval_seconds}s)")


# Singleton for use from API
_exporter = None

def get_exporter(reference_data_path: str = None) -> ModelMetricsExporter:
    global _exporter
    if _exporter is None:
        _exporter = ModelMetricsExporter(reference_data_path)
        _exporter.start_background_updates(interval_seconds=60)
    return _exporter


if __name__ == "__main__":
    # Standalone metrics server (port 8001)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",           type=int, default=8001)
    parser.add_argument("--reference-data", type=str, default=None)
    args = parser.parse_args()

    exporter = get_exporter(args.reference_data)
    start_http_server(args.port)
    logger.info(f"Metrics exporter serving on :{args.port}")

    while True:
        time.sleep(60)
