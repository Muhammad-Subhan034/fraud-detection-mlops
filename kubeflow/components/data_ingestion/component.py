"""
Component 1: Data Ingestion
Loads raw IEEE CIS Fraud Detection data, merges transaction + identity tables,
and saves to the artifact store.
"""

import argparse
import json
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train transaction and identity CSVs."""
    logger.info(f"Loading data from {data_dir}")
    train_transaction = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    train_identity = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    logger.info(f"Transactions shape: {train_transaction.shape}")
    logger.info(f"Identity shape:     {train_identity.shape}")
    return train_transaction, train_identity


def merge_data(transactions: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions on TransactionID."""
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(f"Merged shape: {df.shape}")
    logger.info(f"Fraud rate:   {df['isFraud'].mean():.4f}")
    return df


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Compute and return basic dataset statistics."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "fraud_count": int(df["isFraud"].sum()),
        "non_fraud_count": int((df["isFraud"] == 0).sum()),
        "fraud_rate": float(df["isFraud"].mean()),
        "missing_pct": float(df.isnull().mean().mean()),
        "n_numeric_cols": int(df.select_dtypes(include=np.number).shape[1]),
        "n_categorical_cols": int(df.select_dtypes(include="object").shape[1]),
    }


def ingest_data(data_dir: str, output_path: str, stats_path: str) -> None:
    transactions, identity = load_data(data_dir)
    df = merge_data(transactions, identity)

    stats = compute_basic_stats(df)
    logger.info(f"Dataset stats: {json.dumps(stats, indent=2)}")

    # Save merged data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved merged data to {output_path}")

    # Save stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw CSVs")
    parser.add_argument("--output-path", type=str, required=True, help="Output parquet path")
    parser.add_argument("--stats-path", type=str, required=True, help="Output JSON stats path")
    args = parser.parse_args()

    ingest_data(args.data_dir, args.output_path, args.stats_path)
