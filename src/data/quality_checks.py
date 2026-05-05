"""
Data quality checks used by CI pipeline.
"""

import os

import pandas as pd


def run_quality_checks(data_path: str) -> dict:
    if not os.path.exists(data_path):
        return {"passed": True, "errors": [], "warnings": ["Sample data not found - skipping"]}

    df = pd.read_parquet(data_path)
    errors = []
    warnings = []

    if "isFraud" in df.columns:
        fraud_rate = float(df["isFraud"].mean())
        if fraud_rate < 0.001:
            errors.append(f"Fraud rate {fraud_rate:.4f} suspiciously low")
        if fraud_rate > 0.5:
            errors.append(f"Fraud rate {fraud_rate:.4f} suspiciously high")

    if "TransactionAmt" in df.columns and (df["TransactionAmt"] <= 0).any():
        errors.append("Negative/zero transaction amounts found")

    overall_missing = float(df.isnull().mean().mean())
    if overall_missing > 0.5:
        warnings.append(f"High overall missing rate: {overall_missing:.2%}")

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}
