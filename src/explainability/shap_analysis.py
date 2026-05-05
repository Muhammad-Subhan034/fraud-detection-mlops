"""
Task 9: Explainability using SHAP
- Global feature importance
- Local explanations (why did this transaction get flagged?)
- SHAP summary plots
- Waterfall plots for individual predictions
"""

import os
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FraudExplainer:
    def __init__(self, model_artifact_path: str):
        artifact = joblib.load(model_artifact_path)
        self.model     = artifact["model"]
        self.threshold = artifact.get("threshold", 0.5)
        self.selector  = artifact.get("selector", None)
        self.explainer = None
        self.model_type = type(self.model).__name__
        logger.info(f"Loaded {self.model_type} for explanation")

    def build_explainer(self, X_background: pd.DataFrame, n_background: int = 500):
        """
        Build SHAP explainer.
        - TreeExplainer for XGBoost/LightGBM/RandomForest (exact, fast)
        - KernelExplainer as fallback
        """
        X = X_background.sample(min(n_background, len(X_background)), random_state=42)
        if self.selector is not None:
            X = pd.DataFrame(self.selector.transform(X))

        tree_models = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier")
        if self.model_type in tree_models:
            logger.info("Using TreeExplainer (fast, exact)")
            self.explainer = shap.TreeExplainer(self.model)
        else:
            logger.info("Using KernelExplainer (slower, model-agnostic)")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(X, 100)
            )

        self._background = X
        return self

    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 2000) -> np.ndarray:
        """Compute SHAP values for dataset (subsample if large)."""
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        if self.selector is not None:
            X = pd.DataFrame(self.selector.transform(X))
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)
        # For binary classifiers, get the fraud class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values, X

    def plot_summary(self, X: pd.DataFrame, output_path: str, top_n: int = 30):
        """Global feature importance: SHAP beeswarm plot."""
        shap_values, X_used = self.compute_shap_values(X)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_used,
            max_display=top_n,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Feature Importance - Fraud Detection", pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved: {output_path}")

    def plot_bar_importance(self, X: pd.DataFrame, output_path: str, top_n: int = 20):
        """Bar chart of mean |SHAP| per feature."""
        shap_values, X_used = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_names = list(X_used.columns) if hasattr(X_used, "columns") else \
                        [f"f{i}" for i in range(X_used.shape[1])]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(importance_df["feature"][::-1],
                       importance_df["mean_abs_shap"][::-1],
                       color="steelblue")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Most Important Features (SHAP)")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        # Return as dict for reporting
        return importance_df.to_dict(orient="records")

    def explain_single_prediction(self, x: pd.Series, output_path: str = None):
        """
        Waterfall plot for a single transaction.
        Explains: why is this predicted as fraud?
        """
        X_df = pd.DataFrame([x])
        if self.selector is not None:
            X_arr = self.selector.transform(X_df)
            X_df = pd.DataFrame(X_arr)

        if isinstance(self.explainer, shap.TreeExplainer):
            sv = self.explainer.shap_values(X_df)
            if isinstance(sv, list):
                sv = sv[1]
            expected = self.explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = expected[1]
            explanation = shap.Explanation(
                values=sv[0],
                base_values=expected,
                data=X_df.iloc[0].values,
                feature_names=list(X_df.columns),
            )
        else:
            sv = self.explainer.shap_values(X_df)
            if isinstance(sv, list):
                sv = sv[1]
            explanation = shap.Explanation(
                values=sv[0],
                base_values=self.explainer.expected_value,
                data=X_df.iloc[0].values,
                feature_names=list(X_df.columns),
            )

        if output_path:
            plt.figure()
            shap.waterfall_plot(explanation, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

        # Return top contributing features
        contrib = pd.DataFrame({
            "feature": list(X_df.columns),
            "shap_value": explanation.values,
        }).sort_values("shap_value", key=abs, ascending=False)
        return contrib.head(10).to_dict(orient="records")

    def dependency_plot(self, feature: str, X: pd.DataFrame, output_path: str):
        """SHAP dependence plot for a specific feature."""
        shap_values, X_used = self.compute_shap_values(X)
        feature_names = list(X_used.columns) if hasattr(X_used, "columns") else \
                        [f"f{i}" for i in range(X_used.shape[1])]
        if feature not in feature_names:
            logger.warning(f"Feature {feature} not found")
            return
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature, shap_values, X_used,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def run_explainability_analysis(
    model_path: str,
    data_dir: str,
    output_dir: str,
) -> dict:
    """Run full explainability analysis and save outputs."""
    import json
    os.makedirs(output_dir, exist_ok=True)

    X_test = pd.read_parquet(os.path.join(data_dir, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(data_dir, "y_test.parquet")).squeeze()
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))

    explainer = FraudExplainer(model_path)
    explainer.build_explainer(X_train, n_background=500)

    # 1. Summary plot (all test samples)
    explainer.plot_summary(
        X_test.sample(min(1000, len(X_test)), random_state=42),
        os.path.join(output_dir, "shap_summary.png"),
        top_n=30,
    )

    # 2. Bar importance plot
    importance = explainer.plot_bar_importance(
        X_test.sample(min(1000, len(X_test)), random_state=42),
        os.path.join(output_dir, "shap_bar_importance.png"),
        top_n=20,
    )

    # 3. Explain top fraud predictions
    fraud_indices = y_test[y_test == 1].index[:5]
    fraud_explanations = []
    for i, idx in enumerate(fraud_indices):
        explanation = explainer.explain_single_prediction(
            X_test.loc[idx],
            output_path=os.path.join(output_dir, f"waterfall_fraud_{i}.png"),
        )
        fraud_explanations.append({"index": int(idx), "top_features": explanation})

    # 4. Dependency plot for top feature
    if importance:
        top_feat = importance[0]["feature"]
        explainer.dependency_plot(
            top_feat, X_test.sample(500, random_state=42),
            os.path.join(output_dir, f"dependence_{top_feat[:20]}.png"),
        )

    results = {
        "top_features": importance,
        "fraud_explanations": fraud_explanations,
        "output_dir": output_dir,
    }

    with open(os.path.join(output_dir, "explainability_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Explainability analysis complete. Outputs in {output_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args()

    run_explainability_analysis(args.model_path, args.data_dir, args.output_dir)
