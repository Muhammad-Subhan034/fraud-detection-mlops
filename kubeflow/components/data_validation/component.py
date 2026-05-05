"""
Component 2: Data Validation
Validates schema, checks for critical missing values, class distribution,
and rejects data that fails quality thresholds.
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Schema Definition ----
REQUIRED_COLUMNS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"]
TARGET_COLUMN = "isFraud"
MAX_MISSING_PCT = 0.90          # Drop columns missing more than 90%
MIN_FRAUD_RATE = 0.001          # Minimum acceptable fraud rate
MAX_FRAUD_RATE = 0.50           # Maximum (sanity check)
MIN_ROWS = 10_000               # Minimum required rows


def validate_schema(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors


def validate_target(df: pd.DataFrame) -> list[str]:
    errors = []
    if df[TARGET_COLUMN].isnull().any():
        errors.append(f"Target column '{TARGET_COLUMN}' contains nulls")
    unique_vals = set(df[TARGET_COLUMN].unique())
    if not unique_vals.issubset({0, 1}):
        errors.append(f"Target column has unexpected values: {unique_vals}")
    fraud_rate = df[TARGET_COLUMN].mean()
    if fraud_rate < MIN_FRAUD_RATE:
        errors.append(f"Fraud rate {fraud_rate:.4f} below minimum {MIN_FRAUD_RATE}")
    if fraud_rate > MAX_FRAUD_RATE:
        errors.append(f"Fraud rate {fraud_rate:.4f} above maximum {MAX_FRAUD_RATE}")
    return errors


def validate_row_count(df: pd.DataFrame) -> list[str]:
    if len(df) < MIN_ROWS:
        return [f"Row count {len(df)} below minimum {MIN_ROWS}"]
    return []


def check_missing_values(df: pd.DataFrame) -> dict:
    """Return per-column missing stats and flag high-missing columns."""
    missing_pct = (df.isnull().mean() * 100).round(2)
    high_missing = missing_pct[missing_pct > MAX_MISSING_PCT * 100].to_dict()
    return {
        "per_column_missing_pct": missing_pct.to_dict(),
        "columns_above_threshold": high_missing,
        "overall_missing_pct": float(df.isnull().mean().mean() * 100),
    }


def check_data_types(df: pd.DataFrame) -> dict:
    return {
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
        "n_numeric": int(df.select_dtypes(include=np.number).shape[1]),
        "n_categorical": int(df.select_dtypes(include="object").shape[1]),
    }


def validate_data(input_path: str, output_report_path: str, validation_status_path: str) -> None:
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Data shape: {df.shape}")

    all_errors = []
    all_errors += validate_schema(df)
    all_errors += validate_target(df)
    all_errors += validate_row_count(df)

    missing_info = check_missing_values(df)
    dtype_info = check_data_types(df)

    report = {
        "status": "PASS" if not all_errors else "FAIL",
        "errors": all_errors,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "missing_info": {
            "overall_missing_pct": missing_info["overall_missing_pct"],
            "n_cols_above_threshold": len(missing_info["columns_above_threshold"]),
            "high_missing_columns": list(missing_info["columns_above_threshold"].keys()),
        },
        "dtype_info": dtype_info,
    }

    import os
    os.makedirs(Path(output_report_path).parent, exist_ok=True)

    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(validation_status_path, "w") as f:
        f.write(report["status"])

    if all_errors:
        logger.error("Validation FAILED:")
        for e in all_errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    else:
        logger.info("Validation PASSED ✓")
        logger.info(f"Fraud rate: {report['fraud_rate']:.4f}")
        logger.info(f"Overall missing: {report['missing_info']['overall_missing_pct']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",             type=str, required=True)
    parser.add_argument("--output-report-path",     type=str, required=True)
    parser.add_argument("--validation-status-path", type=str, required=True)
    args = parser.parse_args()

    validate_data(args.input_path, args.output_report_path, args.validation_status_path)
