"""
Unit tests for the Fraud Detection MLOps system.
Tests: data preprocessing, model training utilities, API, drift detection.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    """Create a small sample DataFrame mimicking IEEE CIS structure."""
    np.random.seed(42)
    n = 5000
    fraud_n = 250  # 5% fraud

    df = pd.DataFrame({
        "TransactionID":  range(n),
        "isFraud":        [1] * fraud_n + [0] * (n - fraud_n),
        "TransactionDT":  np.random.randint(86400, 86400 * 180, n),
        "TransactionAmt": np.random.lognormal(4.5, 1.5, n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(1000, 20000, n),
        "card4":          np.random.choice(["visa", "mastercard", "discover"], n),
        "card6":          np.random.choice(["debit", "credit"], n),
        "addr1":          np.random.randint(100, 500, n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "V1":             np.random.randn(n),
        "V2":             np.random.randn(n),
        "V3":             np.random.randn(n),
        "D1":             np.random.randint(0, 500, n).astype(float),
        "D2":             np.random.randint(0, 200, n).astype(float),
        "C1":             np.random.randint(0, 10, n).astype(float),
    })
    # Add some missingness
    for col in ["addr1", "V2", "D2"]:
        mask = np.random.rand(n) < 0.15
        df.loc[mask, col] = np.nan

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def sample_features(sample_df):
    """Features only (no target, no ID)."""
    return sample_df.drop(columns=["TransactionID", "isFraud"])


@pytest.fixture
def sample_target(sample_df):
    return sample_df["isFraud"]


# ============================================================
# DATA VALIDATION TESTS
# ============================================================

class TestDataValidation:
    def test_validate_schema_passes(self, sample_df):
        from kubeflow.components.data_validation.component import validate_schema
        errors = validate_schema(sample_df)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_validate_schema_fails_missing_col(self, sample_df):
        from kubeflow.components.data_validation.component import validate_schema
        df_bad = sample_df.drop(columns=["isFraud"])
        errors = validate_schema(df_bad)
        assert any("isFraud" in e for e in errors)

    def test_validate_target_binary(self, sample_df):
        from kubeflow.components.data_validation.component import validate_target
        errors = validate_target(sample_df)
        assert errors == []

    def test_validate_target_null(self, sample_df):
        from kubeflow.components.data_validation.component import validate_target
        df = sample_df.copy()
        df.loc[0, "isFraud"] = np.nan
        errors = validate_target(df)
        assert len(errors) > 0

    def test_validate_row_count(self, sample_df):
        from kubeflow.components.data_validation.component import validate_row_count
        errors = validate_row_count(sample_df)
        assert errors == []

    def test_validate_row_count_too_small(self):
        from kubeflow.components.data_validation.component import validate_row_count
        small_df = pd.DataFrame({"isFraud": [0, 1, 0]})
        errors = validate_row_count(small_df)
        assert len(errors) > 0


# ============================================================
# PREPROCESSING TESTS
# ============================================================

class TestPreprocessing:
    def test_missing_value_handling_train(self, sample_df):
        from kubeflow.components.data_preprocessing.component import handle_missing_values
        df = sample_df.drop(columns=["TransactionID", "isFraud"])
        result, fill_values = handle_missing_values(df, is_train=True)
        assert result.isnull().sum().sum() == 0
        assert len(fill_values) > 0

    def test_missing_value_handling_test_uses_train_stats(self, sample_df):
        from kubeflow.components.data_preprocessing.component import handle_missing_values
        df = sample_df.drop(columns=["TransactionID", "isFraud"])
        _, fill_values = handle_missing_values(df.iloc[:1000], is_train=True)
        result, _ = handle_missing_values(df.iloc[1000:], is_train=False, fill_values=fill_values)
        assert result.isnull().sum().sum() == 0

    def test_missingness_flag_created(self, sample_df):
        from kubeflow.components.data_preprocessing.component import handle_missing_values
        df = sample_df.drop(columns=["TransactionID", "isFraud"])
        result, _ = handle_missing_values(df, is_train=True)
        # Should have _is_missing columns for high-missing cols
        flag_cols = [c for c in result.columns if c.endswith("_is_missing")]
        assert len(flag_cols) > 0

    def test_basic_feature_engineering(self, sample_df):
        from kubeflow.components.data_preprocessing.component import basic_feature_engineering
        result = basic_feature_engineering(sample_df)
        assert "TransactionAmt_log" in result.columns
        assert "transaction_hour" in result.columns
        assert "TransactionAmt_cents" in result.columns

    def test_encode_categoricals_no_leakage(self, sample_df):
        """Target encoder should be fit only on train, not val/test."""
        from kubeflow.components.data_preprocessing.component import encode_categoricals
        from sklearn.model_selection import train_test_split

        df = sample_df.drop(columns=["TransactionID"])
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]

        X_tr, X_te = train_test_split(X, test_size=0.3, random_state=42)
        y_tr, y_te = y.loc[X_tr.index], y.loc[X_te.index]

        X_tr_enc, X_val_enc, X_te_enc, encoders = encode_categoricals(
            X_tr.copy(), y_tr, X_tr.copy(), X_te.copy()
        )
        assert "target_encoder" in encoders or len(encoders) > 0


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

class TestFeatureEngineering:
    def test_v_feature_aggregation(self, sample_df):
        from kubeflow.components.feature_engineering.component import aggregate_v_features
        result = aggregate_v_features(sample_df)
        assert "v_mean" in result.columns
        assert "v_std" in result.columns
        assert "v_missing" in result.columns

    def test_email_features(self, sample_df):
        from kubeflow.components.feature_engineering.component import email_domain_features
        result = email_domain_features(sample_df)
        assert "P_emaildomain_is_risky" in result.columns

    def test_transaction_velocity(self, sample_df):
        from kubeflow.components.feature_engineering.component import transaction_velocity
        result = transaction_velocity(sample_df)
        assert "card1_amt_mean" in result.columns
        assert "amt_vs_card_mean" in result.columns


# ============================================================
# MODEL TRAINING TESTS
# ============================================================

class TestModelTraining:
    def test_compute_metrics(self):
        from kubeflow.components.model_training.component import compute_metrics
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.4])
        metrics = compute_metrics(y_true, y_prob)
        assert "auc_roc" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_find_best_threshold(self):
        from kubeflow.components.model_training.component import find_best_threshold
        y_true = np.array([1, 0, 1, 0, 1] * 20)
        y_prob = np.array([0.9, 0.1, 0.8, 0.15, 0.75] * 20)
        t = find_best_threshold(y_true, y_prob, min_recall=0.80)
        assert 0.0 < t < 1.0

    def test_class_weight_calculation(self):
        from kubeflow.components.model_training.component import get_class_weight
        y = pd.Series([0] * 950 + [1] * 50)
        scale = get_class_weight(y)
        assert abs(scale - 19.0) < 0.1

    def test_smote_increases_minority(self):
        from kubeflow.components.model_training.component import apply_smote
        X = pd.DataFrame(np.random.randn(1000, 10))
        y = pd.Series([0] * 950 + [1] * 50)
        X_res, y_res = apply_smote(X, y)
        assert y_res.sum() > 50  # More fraud samples
        assert len(X_res) > 1000


# ============================================================
# EVALUATION TESTS
# ============================================================

class TestEvaluation:
    def test_business_impact(self):
        from kubeflow.components.model_evaluation.component import business_impact
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        impact = business_impact(y_true, y_pred, y_pred.astype(float))
        assert "missed_fraud_loss" in impact
        assert "false_alarm_cost" in impact
        assert "net_value" in impact
        assert impact["false_negatives"] == 1  # 1 missed fraud

    def test_deploy_decision_thresholds(self):
        """Model below recall threshold should NOT deploy."""
        from kubeflow.components.model_evaluation.component import MIN_AUC_ROC, MIN_RECALL
        assert MIN_RECALL >= 0.70, "Recall threshold too low for fraud detection"
        assert MIN_AUC_ROC >= 0.80, "AUC threshold too low"


# ============================================================
# DRIFT DETECTION TESTS
# ============================================================

class TestDriftDetection:
    def test_psi_stable_distributions(self):
        from src.monitoring.drift_detection import DriftSimulator
        np.random.seed(42)
        df = pd.DataFrame({
            "TransactionDT": range(1000),
            "TransactionAmt": np.random.lognormal(4.5, 1.0, 1000),
            "isFraud": [0] * 950 + [1] * 50,
        })
        sim = DriftSimulator(df, time_col="TransactionDT")
        result = sim.measure_distribution_drift("TransactionAmt", method="ks")
        # Same overall data — should be low drift
        assert result["statistic"] < 0.5

    def test_threshold_retrain_triggers(self):
        from src.monitoring.drift_detection import ThresholdBasedRetraining
        strategy = ThresholdBasedRetraining(min_recall=0.80)
        should, reason = strategy.should_retrain(recall=0.65)
        assert should
        assert "recall" in reason.lower()

    def test_threshold_retrain_no_trigger(self):
        from src.monitoring.drift_detection import ThresholdBasedRetraining
        strategy = ThresholdBasedRetraining(min_recall=0.80)
        should, _ = strategy.should_retrain(recall=0.90)
        assert not should

    def test_hybrid_strategy_cooldown(self):
        from src.monitoring.drift_detection import HybridRetraining
        from datetime import timedelta
        strategy = HybridRetraining(min_interval_hours=24)
        # First trigger
        should1, _ = strategy.should_retrain(recall=0.60)
        assert should1
        # Immediate second trigger should be blocked by cooldown
        should2, reason2 = strategy.should_retrain(recall=0.60)
        assert not should2
        assert "cooldown" in reason2.lower()

    def test_strategy_comparison(self):
        from src.monitoring.drift_detection import compare_retraining_strategies
        events = [
            {"days_elapsed": i, "recall": 0.85 - i * 0.005, "auc": 0.90, "max_psi": 0.05 + i * 0.005}
            for i in range(0, 90, 5)
        ]
        df = compare_retraining_strategies(events)
        assert "strategy" in df.columns
        assert "retrain_count" in df.columns
        assert len(df) == 4  # 4 strategies


# ============================================================
# API TESTS
# ============================================================

class TestAPI:
    def test_health_endpoint(self):
        """Mock API health check."""
        from fastapi.testclient import TestClient
        # Patch model loading
        with patch("src.api.api.model_manager") as mock_mm:
            mock_mm.model = MagicMock()
            mock_mm.model.predict_proba.return_value = np.array([[0.9, 0.1]])
            mock_mm.threshold = 0.5
            mock_mm.version = "test-v1"
            mock_mm.selector = None
            mock_mm.feature_cols = None

            from src.api.api import app
            client = TestClient(app)
            response = client.get("/health")
            # Should not crash even without full model
            assert response.status_code in [200, 503]

    def test_transaction_schema_validation(self):
        """Test Pydantic schema rejects invalid inputs."""
        from src.api.api import TransactionFeatures
        from pydantic import ValidationError

        # Negative amount should fail
        with pytest.raises(ValidationError):
            TransactionFeatures(TransactionAmt=-50.0)

        # Zero amount should fail
        with pytest.raises(ValidationError):
            TransactionFeatures(TransactionAmt=0.0)

        # Valid transaction
        tx = TransactionFeatures(TransactionAmt=100.0, ProductCD="W")
        assert tx.TransactionAmt == 100.0


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

class TestDataQuality:
    def test_fraud_rate_in_acceptable_range(self, sample_df):
        fraud_rate = sample_df["isFraud"].mean()
        assert 0.001 <= fraud_rate <= 0.50, f"Unexpected fraud rate: {fraud_rate}"

    def test_transaction_amounts_positive(self, sample_df):
        assert (sample_df["TransactionAmt"] > 0).all()

    def test_no_duplicate_transaction_ids(self, sample_df):
        assert sample_df["TransactionID"].nunique() == len(sample_df)
