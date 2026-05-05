"""
Task 7 & 8: Drift Simulation + Intelligent Retraining
- Time-based drift simulation (early vs late data)
- New fraud pattern injection
- Feature importance shift detection
- Threshold-based, periodic, and hybrid retraining strategies
- Comparison of retraining strategies
"""

import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# DRIFT SIMULATION (Task 7)
# ============================================================

class DriftSimulator:
    """
    Simulates realistic concept and data drift for fraud detection.
    - Train on earlier transactions, test on later ones
    - Inject new fraud patterns (different amount ranges, new cards)
    - Shift feature importance by modifying correlations
    """

    def __init__(self, df: pd.DataFrame, time_col: str = "TransactionDT"):
        self.df = df.sort_values(time_col).reset_index(drop=True)
        self.time_col = time_col
        n = len(df)
        # Split: first 60% = train era, last 40% = test era
        self.train_era = self.df.iloc[:int(n * 0.60)]
        self.test_era = self.df.iloc[int(n * 0.60):]
        logger.info(f"Train era: {len(self.train_era)} rows, "
                    f"Test era: {len(self.test_era)} rows")

    def inject_new_fraud_patterns(self, df: pd.DataFrame, fraud_rate_increase: float = 0.02) -> pd.DataFrame:
        """
        Simulate new fraud patterns:
        1. Large-amount fraud (new high-value fraud type)
        2. New card prefix fraud patterns
        """
        df = df.copy()
        n_new_fraud = int(len(df) * fraud_rate_increase)
        fraud_idx = np.random.choice(df[df["isFraud"] == 0].index, n_new_fraud, replace=False)

        # Pattern 1: High-amount fraud
        high_amount_idx = fraud_idx[:n_new_fraud // 2]
        df.loc[high_amount_idx, "isFraud"] = 1
        df.loc[high_amount_idx, "TransactionAmt"] = np.random.uniform(800, 3000, len(high_amount_idx))

        # Pattern 2: Micro-transaction fraud (new pattern)
        micro_idx = fraud_idx[n_new_fraud // 2:]
        df.loc[micro_idx, "isFraud"] = 1
        df.loc[micro_idx, "TransactionAmt"] = np.random.uniform(0.01, 1.0, len(micro_idx))

        logger.info(f"Injected {n_new_fraud} new fraud patterns. "
                    f"New fraud rate: {df['isFraud'].mean():.4f}")
        return df

    def shift_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate feature importance shift by adding noise to previously
        important features and making new features more predictive.
        """
        df = df.copy()
        # Add noise to V features (previously important)
        v_cols = [c for c in df.columns if c.startswith("V")][:20]
        for col in v_cols:
            if col in df.columns:
                df[col] = df[col] + np.random.normal(0, df[col].std() * 0.5, len(df))

        # Make TransactionAmt more predictive in test era
        if "TransactionAmt" in df.columns:
            fraud_mask = df["isFraud"] == 1
            df.loc[fraud_mask, "TransactionAmt"] *= 1.5

        return df

    def get_drifted_test_set(self) -> pd.DataFrame:
        """Return test era with injected drift."""
        drifted = self.inject_new_fraud_patterns(self.test_era)
        drifted = self.shift_feature_importance(drifted)
        return drifted

    def measure_distribution_drift(self, col: str, method: str = "ks") -> dict:
        """Measure statistical drift between train and test era for a feature."""
        train_vals = self.train_era[col].dropna()
        test_vals = self.test_era[col].dropna()

        if method == "ks":
            stat, p_val = stats.ks_2samp(train_vals, test_vals)
        elif method == "psi":
            stat = self._psi(train_vals, test_vals)
            p_val = None

        return {
            "feature": col,
            "method": method,
            "statistic": float(stat),
            "p_value": float(p_val) if p_val is not None else None,
            "drifted": stat > 0.1,  # KS threshold
        }

    def _psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Population Stability Index."""
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            buckets + 1
        )
        expected_pct = np.histogram(expected, breakpoints)[0] / len(expected) + 1e-8
        actual_pct = np.histogram(actual, breakpoints)[0] / len(actual) + 1e-8
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    def compute_all_drift(self, top_n: int = 30) -> pd.DataFrame:
        """Compute drift for all numeric features."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        numeric_cols = [c for c in numeric_cols if c != "isFraud"][:top_n]

        results = []
        for col in numeric_cols:
            results.append(self.measure_distribution_drift(col))

        drift_df = pd.DataFrame(results).sort_values("statistic", ascending=False)
        n_drifted = drift_df["drifted"].sum()
        logger.info(f"Drift detected in {n_drifted}/{len(drift_df)} features")
        return drift_df


# ============================================================
# RETRAINING STRATEGIES (Task 8)
# ============================================================

class RetrainingStrategy:
    """Base class for retraining strategies."""

    def __init__(self, name: str):
        self.name = name
        self.retrain_history = []

    def should_retrain(self, **kwargs) -> bool:
        raise NotImplementedError

    def log_retrain_event(self, reason: str, metrics: dict):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": self.name,
            "reason": reason,
            "metrics": metrics,
        }
        self.retrain_history.append(event)
        logger.info(f"[{self.name}] Retraining triggered: {reason}")
        return event


class ThresholdBasedRetraining(RetrainingStrategy):
    """
    Trigger retraining when monitored metrics drop below thresholds.
    - Recall drops below 0.75
    - AUC-ROC drops below 0.82
    - Data drift (PSI) exceeds 0.2 in any feature
    """

    def __init__(self, min_recall: float = 0.75, min_auc: float = 0.82,
                 max_drift_psi: float = 0.2):
        super().__init__("threshold_based")
        self.min_recall = min_recall
        self.min_auc = min_auc
        self.max_drift_psi = max_drift_psi

    def should_retrain(self, recall: float = None, auc: float = None,
                       max_psi: float = None, **kwargs) -> tuple[bool, str]:
        reasons = []
        if recall is not None and recall < self.min_recall:
            reasons.append(f"recall={recall:.4f} < {self.min_recall}")
        if auc is not None and auc < self.min_auc:
            reasons.append(f"auc={auc:.4f} < {self.min_auc}")
        if max_psi is not None and max_psi > self.max_drift_psi:
            reasons.append(f"PSI={max_psi:.4f} > {self.max_drift_psi}")

        should = bool(reasons)
        return should, "; ".join(reasons) if reasons else "no trigger"


class PeriodicRetraining(RetrainingStrategy):
    """
    Trigger retraining on a fixed schedule (e.g., weekly).
    Simple but ignores actual drift signals.
    """

    def __init__(self, interval_days: int = 7):
        super().__init__("periodic")
        self.interval_days = interval_days
        self.last_retrain = datetime.utcnow()

    def should_retrain(self, current_time: datetime = None, **kwargs) -> tuple[bool, str]:
        now = current_time or datetime.utcnow()
        delta_days = (now - self.last_retrain).days
        if delta_days >= self.interval_days:
            self.last_retrain = now
            return True, f"Periodic: {delta_days} days since last retrain"
        return False, f"Next retrain in {self.interval_days - delta_days} days"


class HybridRetraining(RetrainingStrategy):
    """
    Combines threshold-based triggers with periodic scheduling.
    Strategy: retrain if EITHER trigger fires, but at most once per min_interval.
    """

    def __init__(self, min_recall: float = 0.75, min_auc: float = 0.82,
                 max_drift_psi: float = 0.2, periodic_days: int = 14,
                 min_interval_hours: int = 12):
        super().__init__("hybrid")
        self.threshold = ThresholdBasedRetraining(min_recall, min_auc, max_drift_psi)
        self.periodic = PeriodicRetraining(periodic_days)
        self.min_interval_hours = min_interval_hours
        self.last_retrain = None

    def should_retrain(self, **kwargs) -> tuple[bool, str]:
        # Cooldown: don't retrain more than once per min_interval_hours
        if self.last_retrain:
            hours_since = (datetime.utcnow() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.min_interval_hours:
                return False, f"Cooldown: {hours_since:.1f}h < {self.min_interval_hours}h"

        trig_t, reason_t = self.threshold.should_retrain(**kwargs)
        trig_p, reason_p = self.periodic.should_retrain(**kwargs)

        if trig_t:
            self.last_retrain = datetime.utcnow()
            return True, f"Threshold trigger: {reason_t}"
        if trig_p:
            self.last_retrain = datetime.utcnow()
            return True, f"Periodic trigger: {reason_p}"
        return False, "No trigger"


# ============================================================
# STRATEGY COMPARISON
# ============================================================

def compare_retraining_strategies(simulation_data: list[dict]) -> pd.DataFrame:
    """
    Simulate monitoring events and compare strategies.
    simulation_data: list of dicts with keys: recall, auc, max_psi, days_elapsed
    Returns comparison DataFrame.
    """
    strategies = {
        "threshold_based": ThresholdBasedRetraining(),
        "periodic_weekly": PeriodicRetraining(interval_days=7),
        "periodic_biweekly": PeriodicRetraining(interval_days=14),
        "hybrid": HybridRetraining(),
    }

    results = {name: {"retrain_count": 0, "missed_alerts": 0, "events": []}
               for name in strategies}

    base_time = datetime.utcnow()

    for event in simulation_data:
        current_time = base_time
        # Advance time
        if "days_elapsed" in event:
            from datetime import timedelta
            current_time = base_time + timedelta(days=event["days_elapsed"])

        for name, strategy in strategies.items():
            should, reason = strategy.should_retrain(
                recall=event.get("recall"),
                auc=event.get("auc"),
                max_psi=event.get("max_psi"),
                current_time=current_time,
            )
            if should:
                results[name]["retrain_count"] += 1
                results[name]["events"].append({
                    "day": event.get("days_elapsed", 0),
                    "reason": reason,
                    "recall": event.get("recall"),
                })
            elif event.get("recall", 1.0) < 0.70:
                # Critical drop that was missed
                results[name]["missed_alerts"] += 1

    comparison = []
    for name, res in results.items():
        comparison.append({
            "strategy": name,
            "retrain_count": res["retrain_count"],
            "missed_alerts": res["missed_alerts"],
            "compute_cost": res["retrain_count"] * 45,  # ~45 min per retrain
            "responsiveness": "HIGH" if res["missed_alerts"] == 0 else "LOW",
        })

    return pd.DataFrame(comparison).sort_values("missed_alerts")


# ============================================================
# DRIFT MONITOR (for production)
# ============================================================

class DriftMonitor:
    """
    Production drift monitor that tracks feature distributions.
    Integrates with Prometheus via metrics file.
    """

    def __init__(self, reference_data: pd.DataFrame, psi_threshold: float = 0.2):
        self.reference = reference_data
        self.psi_threshold = psi_threshold
        self.drift_history = []

    def check_drift(self, current_data: pd.DataFrame) -> dict:
        simulator = DriftSimulator.__new__(DriftSimulator)
        simulator.train_era = self.reference
        simulator.test_era = current_data
        simulator.df = pd.concat([self.reference, current_data])
        simulator.time_col = None

        numeric_cols = self.reference.select_dtypes(include=np.number).columns[:30]
        drift_results = []
        for col in numeric_cols:
            psi = simulator._psi(self.reference[col].dropna(), current_data[col].dropna())
            drift_results.append({"feature": col, "psi": float(psi), "drifted": psi > self.psi_threshold})

        max_psi = max(r["psi"] for r in drift_results)
        n_drifted = sum(r["drifted"] for r in drift_results)
        alert = max_psi > self.psi_threshold

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "max_psi": max_psi,
            "n_features_drifted": n_drifted,
            "alert": alert,
            "feature_drift": sorted(drift_results, key=lambda x: x["psi"], reverse=True)[:10],
        }
        self.drift_history.append(result)
        return result


if __name__ == "__main__":
    # Demo: show retraining strategy comparison
    import pandas as pd

    # Simulate 90-day monitoring events
    events = []
    for day in range(0, 90, 3):
        recall = max(0.60, 0.85 - day * 0.003 + np.random.normal(0, 0.02))
        auc = max(0.70, 0.91 - day * 0.002 + np.random.normal(0, 0.01))
        psi = min(0.50, 0.05 + day * 0.004 + np.random.normal(0, 0.02))
        events.append({"days_elapsed": day, "recall": recall, "auc": auc, "max_psi": psi})

    comparison = compare_retraining_strategies(events)
    print("\n=== Retraining Strategy Comparison ===")
    print(comparison.to_string(index=False))
