"""
Component 4: Feature Engineering
- V-feature aggregations
- Transaction velocity features
- Card/email domain features
- Feature importance pre-selection
"""

import argparse
import json
import logging
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_v_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics from V1-V339 Vesta features."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return df

    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"] = df[v_cols].std(axis=1)
    df["v_max"] = df[v_cols].max(axis=1)
    df["v_min"] = df[v_cols].min(axis=1)
    df["v_missing"] = df[v_cols].isnull().sum(axis=1)
    df["v_nonzero"] = (df[v_cols] != 0).sum(axis=1)
    logger.info(f"Created V-feature aggregates from {len(v_cols)} cols")
    return df


def card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine card attributes into composite keys."""
    if "card1" in df.columns and "card2" in df.columns:
        df["card1_card2"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
    if "card1" in df.columns and "addr1" in df.columns:
        df["card1_addr1"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    if "card4" in df.columns and "card6" in df.columns:
        df["card_type_bank"] = df["card4"].astype(str) + "_" + df["card6"].astype(str)
    return df


def email_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain from email columns and flag risky domains."""
    risky_domains = {"anonymous.com", "protonmail.com", "guerrillamail.com", "mailnull.com"}
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            df[f"{col}_domain"] = df[col].astype(str).str.split(".").str[-1].str.lower()
            df[f"{col}_is_risky"] = df[col].astype(str).str.lower().isin(risky_domains).astype(int)
    return df


def transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling count & amount features per card over time window.
    Approximated without a true time-series join.
    """
    if "card1" not in df.columns or "TransactionAmt" not in df.columns:
        return df

    # Mean / std of amount per card1 (global stats since no real-time join)
    card1_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count"])
    card1_stats.columns = ["card1_amt_mean", "card1_amt_std", "card1_tx_count"]
    df = df.merge(card1_stats, on="card1", how="left")

    # Ratio of current amount vs card mean
    df["amt_vs_card_mean"] = df["TransactionAmt"] / (df["card1_amt_mean"] + 1e-8)
    return df


def d_feature_diff(df: pd.DataFrame) -> pd.DataFrame:
    """D features represent time deltas — compute differences."""
    d_cols = [c for c in df.columns if c.startswith("D")]
    if len(d_cols) >= 2:
        df["D_range"] = df[d_cols].max(axis=1) - df[d_cols].min(axis=1)
        df["D_mean"] = df[d_cols].mean(axis=1)
    return df


def engineer_features(data_dir: str, output_dir: str, feature_meta_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        X_path = os.path.join(data_dir, f"X_{split}.parquet")
        if not os.path.exists(X_path):
            continue

        X = pd.read_parquet(X_path)
        logger.info(f"Engineering features for {split}: {X.shape}")

        X = aggregate_v_features(X)
        X = card_features(X)
        X = email_domain_features(X)
        X = transaction_velocity(X)
        X = d_feature_diff(X)

        # Encode any new object columns as integers
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.Categorical(X[col]).codes

        # Replace inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        out_path = os.path.join(output_dir, f"X_{split}.parquet")
        X.to_parquet(out_path, index=False)
        logger.info(f"Saved {split} features: {X.shape} → {out_path}")

    # Copy y files
    for split in ["train", "val", "test"]:
        y_src = os.path.join(data_dir, f"y_{split}.parquet")
        if os.path.exists(y_src):
            y = pd.read_parquet(y_src)
            y.to_parquet(os.path.join(output_dir, f"y_{split}.parquet"), index=False)

    # Feature metadata
    X_train = pd.read_parquet(os.path.join(output_dir, "X_train.parquet"))
    meta = {
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "dtypes": X_train.dtypes.astype(str).to_dict(),
    }
    with open(feature_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Feature metadata: {X_train.shape[1]} features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--feature-meta-path", type=str, required=True)
    args = parser.parse_args()

    engineer_features(args.data_dir, args.output_dir, args.feature_meta_path)
