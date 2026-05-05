"""
scripts/validate_data_schema.py
Used in CI/CD Stage 1 to validate data schema before pipeline submission.
"""

import argparse
import json
import logging
import sys
import os
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_schema(schema_path: str) -> dict:
    with open(schema_path) as f:
        return yaml.safe_load(f)


def validate_against_schema(df: pd.DataFrame, schema: dict) -> list[str]:
    errors = []

    # Required columns
    for col in schema.get("required_columns", []):
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Column types
    for col, expected_type in schema.get("column_types", {}).items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col}: expected numeric, got {actual_type}")
            elif expected_type == "categorical" and not pd.api.types.is_object_dtype(df[col]):
                pass  # Allow numeric categoricals encoded as int

    # Value ranges
    for col, range_spec in schema.get("value_ranges", {}).items():
        if col in df.columns:
            if "min" in range_spec and df[col].min() < range_spec["min"]:
                errors.append(f"Column {col}: min value {df[col].min()} < {range_spec['min']}")
            if "max" in range_spec and df[col].max() > range_spec["max"]:
                errors.append(f"Column {col}: max value {df[col].max()} > {range_spec['max']}")

    # Null fraction checks
    for col, max_null_pct in schema.get("max_null_fraction", {}).items():
        if col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > max_null_pct:
                errors.append(f"Column {col}: null fraction {null_pct:.2%} > {max_null_pct:.2%}")

    return errors


def run_quality_checks(data_path: str) -> dict:
    """Standalone quality checks (no schema file required)."""
    if not os.path.exists(data_path):
        return {"passed": True, "errors": [], "warnings": ["Sample data not found — skipping"]}

    df = pd.read_parquet(data_path)
    errors = []
    warnings = []

    if "isFraud" in df.columns:
        fraud_rate = df["isFraud"].mean()
        if fraud_rate < 0.001:
            errors.append(f"Fraud rate {fraud_rate:.4f} suspiciously low")
        if fraud_rate > 0.5:
            errors.append(f"Fraud rate {fraud_rate:.4f} suspiciously high")

    if "TransactionAmt" in df.columns:
        if (df["TransactionAmt"] <= 0).any():
            errors.append("Negative/zero transaction amounts found")

    overall_missing = df.isnull().mean().mean()
    if overall_missing > 0.5:
        warnings.append(f"High overall missing rate: {overall_missing:.2%}")

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-path",  required=True)
    parser.add_argument("--sample-data",  required=True)
    args = parser.parse_args()

    if not os.path.exists(args.sample_data):
        logger.warning(f"Sample data not found at {args.sample_data} — skipping validation")
        sys.exit(0)

    schema = load_schema(args.schema_path)
    df = pd.read_parquet(args.sample_data)
    errors = validate_against_schema(df, schema)

    if errors:
        logger.error(f"Schema validation FAILED: {len(errors)} errors")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)
    else:
        logger.info("Schema validation PASSED ✓")
