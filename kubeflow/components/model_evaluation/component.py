"""
Component 6: Model Evaluation
Evaluates best model on hold-out test set.
Produces: full metrics, confusion matrix, ROC/PR curves, business impact analysis.
Conditional deployment: only passes if AUC-ROC >= threshold.
"""

import argparse
import json
import logging
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, classification_report
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Deployment thresholds
MIN_AUC_ROC   = 0.85
MIN_RECALL    = 0.75
MIN_PRECISION = 0.20   # Fraud detection: recall > precision


# ============================================================
# BUSINESS IMPACT ANALYSIS
# ============================================================

def business_impact(y_true, y_pred, y_prob,
                     avg_fraud_amount: float = 500.0,
                     review_cost: float = 10.0) -> dict:
    """
    Compute financial impact:
    - FN (missed fraud): avg_fraud_amount per case lost
    - FP (false alarm): review_cost per investigation
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fraud_loss_standard  = fn * avg_fraud_amount   # FN * loss per fraud
    false_alarm_cost     = fp * review_cost         # FP * cost per review
    total_cost_standard  = fraud_loss_standard + false_alarm_cost

    # Perfect detector baseline
    total_fraud_amount = int(y_true.sum()) * avg_fraud_amount

    # Cost-sensitive: reduced FN at cost of higher FP
    savings = (int(y_true.sum()) - fn) * avg_fraud_amount  # avoided fraud losses

    return {
        "avg_fraud_amount":    avg_fraud_amount,
        "review_cost_per_fp":  review_cost,
        "true_positives":      int(tp),
        "false_positives":     int(fp),
        "true_negatives":      int(tn),
        "false_negatives":     int(fn),
        "missed_fraud_loss":   float(fraud_loss_standard),
        "false_alarm_cost":    float(false_alarm_cost),
        "total_cost":          float(total_cost_standard),
        "fraud_savings":       float(savings),
        "total_fraud_in_test": float(total_fraud_amount),
        "net_value":           float(savings - false_alarm_cost),
    }


def plot_roc_curve(y_true, y_prob, output_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Fraud Detection")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_prob, output_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recall, precision, color="b", alpha=0.8, where="post",
            label=f"PR Curve (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Fraud Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Non-Fraud", "Predicted Fraud"])
    ax.set_yticklabels(["Actual Non-Fraud", "Actual Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Confusion Matrix - Fraud Detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_distribution(y_true, y_prob, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Non-Fraud", color="blue", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Fraud",     color="red",  density=True)
    ax.set_xlabel("Predicted Probability (Fraud)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(model_dir: str, data_dir: str, output_dir: str,
             eval_metrics_path: str, deploy_decision_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_path = os.path.join(model_dir, "best_model.joblib")
    artifact = joblib.load(model_path)
    model     = artifact["model"]
    threshold = artifact.get("threshold", 0.5)
    selector  = artifact.get("selector", None)

    # Load test data
    X_test = pd.read_parquet(os.path.join(data_dir, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(data_dir, "y_test.parquet")).squeeze()

    # Handle hybrid selector
    if selector is not None:
        X_eval = selector.transform(X_test)
    else:
        X_eval = X_test

    # Predict
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    auc_roc   = roc_auc_score(y_test, y_prob)
    auc_pr    = average_precision_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    logger.info(f"Test AUC-ROC:   {auc_roc:.4f}")
    logger.info(f"Test AUC-PR:    {auc_pr:.4f}")
    logger.info(f"Test Recall:    {recall:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test F1:        {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Non-Fraud','Fraud'])}")

    # Business impact
    impact = business_impact(y_test.values, y_pred, y_prob)

    # Standard vs cost-sensitive comparison
    # (load training metrics for comparison)
    train_metrics_path = os.path.join(model_dir, "..", "training_metrics.json")
    training_comparison = {}
    if os.path.exists(train_metrics_path):
        with open(train_metrics_path) as f:
            training_comparison = json.load(f)

    # Deploy decision
    deploy = (auc_roc >= MIN_AUC_ROC and recall >= MIN_RECALL)
    deploy_reason = []
    if auc_roc < MIN_AUC_ROC:
        deploy_reason.append(f"AUC-ROC {auc_roc:.4f} < {MIN_AUC_ROC}")
    if recall < MIN_RECALL:
        deploy_reason.append(f"Recall {recall:.4f} < {MIN_RECALL}")

    # Plots
    plot_roc_curve(y_test, y_prob, os.path.join(output_dir, "roc_curve.png"))
    plot_pr_curve(y_test, y_prob, os.path.join(output_dir, "pr_curve.png"))
    plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))
    plot_confidence_distribution(y_test.values, y_prob,
                                  os.path.join(output_dir, "confidence_dist.png"))

    # Save full eval metrics
    eval_metrics = {
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "business_impact":  impact,
        "deploy":           deploy,
        "deploy_reasons":   deploy_reason,
    }
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    with open(deploy_decision_path, "w") as f:
        f.write("DEPLOY" if deploy else "NO_DEPLOY")

    logger.info(f"\nDeploy decision: {'✅ DEPLOY' if deploy else '❌ NO_DEPLOY'}")
    if deploy_reason:
        logger.warning(f"Reasons: {deploy_reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",          type=str, required=True)
    parser.add_argument("--data-dir",           type=str, required=True)
    parser.add_argument("--output-dir",         type=str, required=True)
    parser.add_argument("--eval-metrics-path",  type=str, required=True)
    parser.add_argument("--deploy-decision-path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.model_dir, args.data_dir, args.output_dir,
             args.eval_metrics_path, args.deploy_decision_path)
