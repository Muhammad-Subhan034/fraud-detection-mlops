from src.monitoring.drift_detection import ThresholdBasedRetraining
from src.data.quality_checks import run_quality_checks
from kubeflow.components.model_evaluation.component import business_impact
from kubeflow.components.data_preprocessing.component import basic_feature_engineering
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_basic_feature_engineering_adds_columns():
    df = pd.DataFrame({"TransactionDT": [3600, 7200], "TransactionAmt": [10.0, 15.0]})
    out = basic_feature_engineering(df)
    assert "transaction_hour" in out.columns
    assert "TransactionAmt_log" in out.columns


def test_business_impact_returns_required_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    impact = business_impact(y_true, y_pred, y_pred.astype(float))
    for key in ["missed_fraud_loss", "false_alarm_cost", "net_value"]:
        assert key in impact


def test_threshold_strategy_triggers_on_low_recall():
    strategy = ThresholdBasedRetraining(min_recall=0.8)
    should_retrain, _ = strategy.should_retrain(recall=0.6, auc=0.9, max_psi=0.05)
    assert should_retrain


def test_quality_checks_skip_missing_file():
    result = run_quality_checks("data/sample/does-not-exist.parquet")
    assert result["passed"] is True
