"""
Component 5: Model Training
Trains:
  1. XGBoost (standard + cost-sensitive)
  2. LightGBM (standard + cost-sensitive)
  3. Hybrid: RandomForest feature selection + XGBoost
Imbalance strategies:
  A. SMOTE
  B. Class weighting
Compares all combinations and saves best model + all metrics.
"""

import argparse
import json
import logging
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, average_precision_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
FRAUD_RECALL_THRESHOLD = 0.80   # Minimum acceptable recall


# ============================================================
# IMBALANCE HANDLING
# ============================================================

def apply_smote(X: pd.DataFrame, y: pd.Series, sampling_strategy: float = 0.1) -> tuple:
    """Strategy A: SMOTE oversampling of minority class."""
    logger.info("Applying SMOTE...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def get_class_weight(y: pd.Series) -> dict:
    """Strategy B: Compute class weights for cost-sensitive learning."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale = n_neg / n_pos
    logger.info(f"Class weight scale_pos_weight = {scale:.2f}")
    return scale


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_xgboost(scale_pos_weight: float = 1.0, cost_sensitive: bool = False) -> XGBClassifier:
    # Cost-sensitive: higher scale_pos_weight penalizes FN more
    spw = scale_pos_weight * 3.0 if cost_sensitive else scale_pos_weight
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def build_lightgbm(scale_pos_weight: float = 1.0, cost_sensitive: bool = False) -> LGBMClassifier:
    spw = scale_pos_weight * 3.0 if cost_sensitive else scale_pos_weight
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_hybrid_rf_xgb(X_train: pd.DataFrame, y_train: pd.Series,
                        scale_pos_weight: float) -> tuple:
    """
    Hybrid: RandomForest feature selection → XGBoost on selected features.
    Returns (fitted_model, selected_feature_names)
    """
    logger.info("Building hybrid RF→XGBoost model...")
    # Step 1: RF for feature importance
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Step 2: Select top features
    selector = SelectFromModel(rf, threshold="median", prefit=True)
    X_sel = selector.transform(X_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    logger.info(f"Selected {len(selected_features)} features via RF")

    # Step 3: Train XGBoost on selected features
    xgb = build_xgboost(scale_pos_weight=scale_pos_weight, cost_sensitive=False)
    xgb.fit(X_sel, y_train)

    return xgb, selector, selected_features


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn + 1e-8)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_prob, min_recall: float = FRAUD_RECALL_THRESHOLD) -> float:
    """Find threshold that maximizes F1 while meeting minimum recall."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
    return best_t


# ============================================================
# TRAINING EXPERIMENTS
# ============================================================

def run_all_experiments(data_dir: str, output_dir: str, metrics_path: str,
                        mlflow_uri: str = "mlflow") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(data_dir, "y_train.parquet")).squeeze()
    X_val = pd.read_parquet(os.path.join(data_dir, "X_val.parquet"))
    y_val = pd.read_parquet(os.path.join(data_dir, "y_val.parquet")).squeeze()

    logger.info(f"Train: {X_train.shape}, fraud: {y_train.mean():.4f}")

    spw = get_class_weight(y_train)

    # SMOTE training set
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("fraud-detection")

    all_results = {}

    experiments = [
        # (name, model, X, y, description)
        ("xgb_class_weight", build_xgboost(spw, False), X_train, y_train, "XGBoost + class weight"),
        ("xgb_smote", build_xgboost(1.0, False), X_train_smote, y_train_smote, "XGBoost + SMOTE"),
        ("xgb_cost_sensitive", build_xgboost(spw, True), X_train, y_train, "XGBoost + cost-sensitive"),
        ("lgbm_class_weight", build_lightgbm(spw, False), X_train, y_train, "LightGBM + class weight"),
        ("lgbm_smote", build_lightgbm(1.0, False), X_train_smote, y_train_smote, "LightGBM + SMOTE"),
        ("lgbm_cost_sensitive", build_lightgbm(spw, True), X_train, y_train, "LightGBM + cost-sensitive"),
    ]

    for exp_name, model, X_tr, y_tr, description in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {description}")

        with mlflow.start_run(run_name=exp_name):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)] if hasattr(model, "eval_set") else None)

            y_prob = model.predict_proba(X_val)[:, 1]
            best_t = find_best_threshold(y_val, y_prob)
            metrics = compute_metrics(y_val, y_prob, threshold=best_t)
            metrics["best_threshold"] = best_t
            metrics["description"] = description

            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_param("experiment", exp_name)

            all_results[exp_name] = metrics
            model_path = os.path.join(output_dir, f"{exp_name}.joblib")
            joblib.dump({"model": model, "threshold": best_t}, model_path)

            logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  F1:        {metrics['f1']:.4f}")

    # Hybrid model
    logger.info("\nTraining Hybrid RF→XGBoost...")
    with mlflow.start_run(run_name="hybrid_rf_xgb"):
        xgb_hybrid, selector, sel_features = build_hybrid_rf_xgb(X_train, y_train, spw)
        X_val_sel = selector.transform(X_val)
        y_prob = xgb_hybrid.predict_proba(X_val_sel)[:, 1]
        best_t = find_best_threshold(y_val, y_prob)
        metrics = compute_metrics(y_val, y_prob, threshold=best_t)
        metrics["best_threshold"] = best_t
        metrics["description"] = "Hybrid RF feature selection + XGBoost"
        metrics["n_selected_features"] = len(sel_features)

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        all_results["hybrid_rf_xgb"] = metrics
        joblib.dump({"model": xgb_hybrid, "selector": selector,
                     "threshold": best_t, "selected_features": sel_features},
                    os.path.join(output_dir, "hybrid_rf_xgb.joblib"))
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")

    # Select best model by AUC-ROC
    best_name = max(all_results, key=lambda k: all_results[k]["auc_roc"])
    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"  AUC-ROC: {all_results[best_name]['auc_roc']:.4f}")
    logger.info(f"  Recall:  {all_results[best_name]['recall']:.4f}")

    # Save best model separately
    import shutil
    best_src = os.path.join(output_dir, f"{best_name}.joblib")
    shutil.copy(best_src, os.path.join(output_dir, "best_model.joblib"))

    # Save all metrics
    with open(metrics_path, "w") as f:
        json.dump({"results": all_results, "best_model": best_name}, f, indent=2)

    logger.info(f"All metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics-path", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default="http://mlflow:5000")
    args = parser.parse_args()

    run_all_experiments(args.data_dir, args.output_dir, args.metrics_path, args.mlflow_uri)
