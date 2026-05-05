"""
notebooks/run_full_analysis.py
Runnable script (or convert to Jupyter notebook) for local experimentation.
Demonstrates all tasks: preprocessing, training, imbalance comparison,
cost-sensitive analysis, drift simulation, retraining strategy comparison, SHAP.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "outputs/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """
    Generate realistic synthetic IEEE CIS-like data for local testing.
    Use this if you haven't downloaded the Kaggle dataset yet.
    """
    logger.info(f"Generating {n} synthetic transactions...")
    np.random.seed(42)
    fraud_n = int(n * 0.035)

    df = pd.DataFrame({
        "TransactionID":  range(n),
        "isFraud":        [1] * fraud_n + [0] * (n - fraud_n),
        "TransactionDT":  np.random.randint(86400, 86400 * 180, n),
        "TransactionAmt": np.concatenate([
            np.random.lognormal(6.5, 1.0, fraud_n),    # Fraud: higher amounts
            np.random.lognormal(4.5, 1.5, n - fraud_n)  # Normal
        ]),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(1000, 20000, n),
        "card4":          np.random.choice(["visa", "mastercard", "discover", "amex"], n),
        "card6":          np.random.choice(["debit", "credit"], n),
        "addr1":          np.random.randint(100, 500, n).astype(float),
        "P_emaildomain":  np.random.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n,
            p=[0.4, 0.3, 0.2, 0.05, 0.05]
        ),
    })

    # Add V features
    for i in range(1, 30):
        vals = np.random.randn(n)
        # Fraud correlates with higher V values for some features
        if i % 3 == 0:
            df.loc[df["isFraud"] == 1, f"V{i}"] = vals[:fraud_n] + 1.5
            df.loc[df["isFraud"] == 0, f"V{i}"] = vals[fraud_n:]
        else:
            df[f"V{i}"] = vals
        # Add some missingness
        if i % 5 == 0:
            mask = np.random.rand(n) < 0.2
            df.loc[mask, f"V{i}"] = np.nan

    # Add D and C features
    for i in range(1, 8):
        df[f"D{i}"] = np.random.randint(0, 500, n).astype(float)
        df[f"C{i}"] = np.random.randint(0, 15, n).astype(float)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Fraud rate: {df['isFraud'].mean():.4f}")
    return df


def section_1_preprocessing(df):
    """Task 2: Preprocessing & imbalance handling."""
    logger.info("\n" + "="*60)
    logger.info("SECTION 1: Data Preprocessing")
    logger.info("="*60)

    from kubeflow.components.data_preprocessing.component import (
        handle_missing_values, basic_feature_engineering, encode_categoricals
    )
    from kubeflow.components.feature_engineering.component import (
        aggregate_v_features, email_domain_features, transaction_velocity
    )
    from sklearn.model_selection import train_test_split

    df = basic_feature_engineering(df)
    df = aggregate_v_features(df)
    df = email_domain_features(df)
    df = transaction_velocity(df)

    # Time-based split
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n * 0.70)]
    val_df   = df.iloc[int(n * 0.70):int(n * 0.85)]
    test_df  = df.iloc[int(n * 0.85):]

    TARGET = "isFraud"
    DROP   = ["TransactionID", "TransactionDT"]

    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=DROP, inplace=True, errors="ignore")

    train_df, fv = handle_missing_values(train_df.drop(columns=[TARGET]), is_train=True)
    val_df,   _  = handle_missing_values(val_df.drop(columns=[TARGET]),  is_train=False, fill_values=fv)
    test_df,  _  = handle_missing_values(test_df.drop(columns=[TARGET]), is_train=False, fill_values=fv)

    y_train = df.iloc[:int(n*0.70)][TARGET]
    y_val   = df.iloc[int(n*0.70):int(n*0.85)][TARGET]
    y_test  = df.iloc[int(n*0.85):][TARGET]

    X_train, X_val, X_test, encoders = encode_categoricals(
        train_df.copy(), y_train, val_df.copy(), test_df.copy()
    )

    feature_cols = list(X_train.columns)
    X_val  = X_val.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    import joblib
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X.to_parquet(f"{OUTPUT_DIR}/data/X_{name}.parquet", index=False)
        y.to_frame().to_parquet(f"{OUTPUT_DIR}/data/y_{name}.parquet", index=False)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def section_2_imbalance_comparison(X_train, y_train, X_val, y_val):
    """Task 2: Compare SMOTE vs class weighting."""
    logger.info("\n" + "="*60)
    logger.info("SECTION 2: Imbalance Strategy Comparison")
    logger.info("="*60)

    from kubeflow.components.model_training.component import (
        build_xgboost, apply_smote, get_class_weight, compute_metrics
    )

    spw = get_class_weight(y_train)
    X_smote, y_smote = apply_smote(X_train, y_train)

    results = {}
    configs = [
        ("XGB_ClassWeight", build_xgboost(spw, False), X_train, y_train),
        ("XGB_SMOTE",       build_xgboost(1.0, False), X_smote, y_smote),
    ]
    for name, model, X_tr, y_tr in configs:
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_prob, threshold=0.5)
        results[name] = metrics
        logger.info(f"{name}: AUC={metrics['auc_roc']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, metric in enumerate(["auc_roc", "recall", "f1"]):
        vals = [results[k][metric] for k in results]
        axes[i].bar(list(results.keys()), vals, color=["steelblue", "darkorange"])
        axes[i].set_title(metric.upper().replace("_", "-"))
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=15)
        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center")
    plt.suptitle("Imbalance Strategy Comparison: Class Weight vs SMOTE", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/imbalance_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved imbalance comparison plot")
    return results


def section_3_model_training(X_train, y_train, X_val, y_val):
    """Task 3: Train all models."""
    logger.info("\n" + "="*60)
    logger.info("SECTION 3: Model Training (XGBoost, LightGBM, Hybrid)")
    logger.info("="*60)

    from kubeflow.components.model_training.component import (
        build_xgboost, build_lightgbm, build_hybrid_rf_xgb,
        get_class_weight, compute_metrics, find_best_threshold
    )
    import joblib

    spw = get_class_weight(y_train)
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)

    all_results = {}
    models_to_save = {}

    for name, model in [
        ("xgb_class_weight", build_xgboost(spw, False)),
        ("xgb_cost_sensitive", build_xgboost(spw, True)),
        ("lgbm_class_weight", build_lightgbm(spw, False)),
        ("lgbm_cost_sensitive", build_lightgbm(spw, True)),
    ]:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        t = find_best_threshold(y_val, y_prob)
        metrics = compute_metrics(y_val, y_prob, threshold=t)
        all_results[name] = metrics
        models_to_save[name] = {"model": model, "threshold": t}
        logger.info(f"{name}: AUC={metrics['auc_roc']:.4f}, Recall={metrics['recall']:.4f}")

    # Hybrid
    xgb_h, selector, sel_feats = build_hybrid_rf_xgb(X_train, y_train, spw)
    X_val_sel = selector.transform(X_val)
    y_prob = xgb_h.predict_proba(X_val_sel)[:, 1]
    t = find_best_threshold(y_val, y_prob)
    metrics = compute_metrics(y_val, y_prob, threshold=t)
    all_results["hybrid_rf_xgb"] = metrics
    models_to_save["hybrid_rf_xgb"] = {"model": xgb_h, "selector": selector, "threshold": t}
    logger.info(f"hybrid_rf_xgb: AUC={metrics['auc_roc']:.4f}, Recall={metrics['recall']:.4f}")

    # Save best
    best = max(all_results, key=lambda k: all_results[k]["auc_roc"])
    joblib.dump(models_to_save[best], f"{OUTPUT_DIR}/models/best_model.joblib")
    with open(f"{OUTPUT_DIR}/training_results.json", "w") as f:
        json.dump({"results": all_results, "best_model": best}, f, indent=2)

    logger.info(f"Best model: {best} (AUC={all_results[best]['auc_roc']:.4f})")
    return all_results, models_to_save, best


def section_4_explainability(X_train, X_test, y_test, models_to_save):
    """Task 9: SHAP analysis."""
    logger.info("\n" + "="*60)
    logger.info("SECTION 4: SHAP Explainability")
    logger.info("="*60)

    from src.explainability.shap_analysis import FraudExplainer
    import joblib

    best_name = max(models_to_save,
                    key=lambda k: k)  # Use first available
    artifact = models_to_save[best_name]
    tmp_path = f"{OUTPUT_DIR}/shap_model.joblib"
    joblib.dump(artifact, tmp_path)

    explainer = FraudExplainer(tmp_path)
    explainer.build_explainer(X_train, n_background=200)

    os.makedirs(f"{OUTPUT_DIR}/shap", exist_ok=True)
    importance = explainer.plot_bar_importance(
        X_test.sample(min(500, len(X_test)), random_state=42),
        f"{OUTPUT_DIR}/shap/bar_importance.png"
    )
    logger.info(f"Top features: {[f['feature'] for f in importance[:5]]}")
    return importance


def section_5_drift_simulation(df):
    """Task 7: Drift simulation."""
    logger.info("\n" + "="*60)
    logger.info("SECTION 5: Drift Simulation")
    logger.info("="*60)

    from src.monitoring.drift_detection import DriftSimulator, compare_retraining_strategies

    sim = DriftSimulator(df, time_col="TransactionDT")
    drift_df = sim.compute_all_drift(top_n=20)
    logger.info(f"Top drifted features:\n{drift_df.head()}")

    # Retraining comparison
    events = []
    for day in range(0, 90, 3):
        recall = max(0.60, 0.85 - day * 0.003 + np.random.normal(0, 0.02))
        auc    = max(0.70, 0.91 - day * 0.002)
        psi    = min(0.50, 0.05 + day * 0.005)
        events.append({"days_elapsed": day, "recall": recall, "auc": auc, "max_psi": psi})

    comparison = compare_retraining_strategies(events)
    logger.info(f"\nRetraining Strategy Comparison:\n{comparison.to_string(index=False)}")
    comparison.to_csv(f"{OUTPUT_DIR}/retraining_comparison.csv", index=False)
    return comparison


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Try to load real data first, fall back to synthetic
    real_data_paths = [
        "data/raw/train_transaction.csv",
        "../data/raw/train_transaction.csv",
    ]

    df = None
    for p in real_data_paths:
        if os.path.exists(p):
            logger.info(f"Loading real data from {p}")
            tx = pd.read_csv(p)
            try:
                id_path = p.replace("train_transaction", "train_identity")
                if os.path.exists(id_path):
                    identity = pd.read_csv(id_path)
                    df = tx.merge(identity, on="TransactionID", how="left")
                else:
                    df = tx
            except Exception:
                df = tx
            break

    if df is None:
        logger.info("Real data not found — using synthetic data for demonstration")
        df = generate_synthetic_data(n=30_000)

    X_tr, y_tr, X_v, y_v, X_te, y_te = section_1_preprocessing(df)
    section_2_imbalance_comparison(X_tr, y_tr, X_v, y_v)
    all_results, models, best = section_3_model_training(X_tr, y_tr, X_v, y_v)
    section_4_explainability(X_tr, X_te, y_te, models)
    section_5_drift_simulation(df)

    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete! Outputs saved to: {OUTPUT_DIR}/")
    logger.info(f"Best model: {best} — AUC={all_results[best]['auc_roc']:.4f}, "
                f"Recall={all_results[best]['recall']:.4f}")
