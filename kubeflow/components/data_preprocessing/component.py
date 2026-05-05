"""
Component 3: Data Preprocessing
- Advanced missing value strategies
- High-cardinality categorical handling
- Train/validation/test split
- Feature encoding (target encoding for high-cardinality)
"""

import argparse
import json
import logging
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "isFraud"
DROP_COLS = ["TransactionID"]
HIGH_CARDINALITY_THRESHOLD = 50   # n unique values above this → target encode
MISSING_THRESHOLD = 0.90          # Drop columns with >90% missing


# ============================================================
# MISSING VALUE STRATEGIES
# ============================================================

def handle_missing_values(df: pd.DataFrame, is_train: bool = True,
                          fill_values: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Advanced missing value handling:
    - Drop columns with >90% missing
    - Numeric: median imputation
    - Categorical: mode imputation + 'MISSING' category
    - Binary flags for columns with >5% missing (missingness is informative)
    """
    if fill_values is None:
        fill_values = {}

    # 1. Drop extreme missing columns
    missing_pct = df.isnull().mean()
    drop_cols = list(missing_pct[missing_pct > MISSING_THRESHOLD].index)
    # Never drop target
    drop_cols = [c for c in drop_cols if c != TARGET]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped {len(drop_cols)} columns with >{MISSING_THRESHOLD*100}% missing")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET]
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # 2. Add missingness indicator flags for informative missingness
    informative_missing = missing_pct[(missing_pct > 0.05) & (missing_pct <= MISSING_THRESHOLD)].index
    for col in informative_missing:
        if col in df.columns and col != TARGET:
            df[f"{col}_is_missing"] = df[col].isnull().astype(int)

    # 3. Numeric: median imputation
    for col in numeric_cols:
        if df[col].isnull().any():
            if is_train:
                fill_val = df[col].median()
                fill_values[col] = fill_val
            else:
                fill_val = fill_values.get(col, 0)
            df[col] = df[col].fillna(fill_val)

    # 4. Categorical: mode + 'MISSING' category
    for col in cat_cols:
        if df[col].isnull().any():
            if is_train:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "MISSING"
                fill_values[col] = mode_val
            else:
                mode_val = fill_values.get(col, "MISSING")
            df[col] = df[col].fillna(mode_val)

    logger.info(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df, fill_values


# ============================================================
# FEATURE ENGINEERING (basic)
# ============================================================

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and amount-based features."""
    # Transaction hour (TransactionDT is seconds from a reference)
    if "TransactionDT" in df.columns:
        df["transaction_hour"] = (df["TransactionDT"] / 3600) % 24
        df["transaction_day"] = (df["TransactionDT"] / 86400) % 7
        df["transaction_week"] = (df["TransactionDT"] / 604800).astype(int)

    # Transaction amount log transform
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        # Cents component (fraud patterns)
        df["TransactionAmt_cents"] = (df["TransactionAmt"] * 100 % 100).astype(int)

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categoricals(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, X_test: pd.DataFrame,
                        encoders: dict = None) -> tuple:
    """
    - Low-cardinality (<= threshold): Label encoding
    - High-cardinality (> threshold): Target encoding (uses y on train only)
    """
    if encoders is None:
        encoders = {}

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    low_card = [c for c in cat_cols if X_train[c].nunique() <= HIGH_CARDINALITY_THRESHOLD]
    high_card = [c for c in cat_cols if X_train[c].nunique() > HIGH_CARDINALITY_THRESHOLD]

    logger.info(f"Low-cardinality cols ({len(low_card)}): {low_card[:10]}")
    logger.info(f"High-cardinality cols ({len(high_card)}): {high_card[:10]}")

    # Label encode low-cardinality
    for col in low_card:
        if col not in encoders:
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str))
            encoders[col] = le
        le = encoders[col]
        for df in [X_train, X_val, X_test]:
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Target encode high-cardinality
    if high_card:
        if "target_encoder" not in encoders:
            te = ce.TargetEncoder(cols=high_card, smoothing=10)
            te.fit(X_train[high_card], y_train)
            encoders["target_encoder"] = te
        te = encoders["target_encoder"]
        X_train[high_card] = te.transform(X_train[high_card])
        X_val[high_card] = te.transform(X_val[high_card])
        X_test[high_card] = te.transform(X_test[high_card])

    return X_train, X_val, X_test, encoders


# ============================================================
# MAIN
# ============================================================

def preprocess(input_path: str, output_dir: str, encoder_path: str, stats_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)

    # Basic feature engineering
    df = basic_feature_engineering(df)

    # Drop IDs
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Train / val / test split (time-aware: use later data as test)
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        logger.info("Using time-based split")
    else:
        train_df, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df[TARGET])
        val_df, test_df = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp[TARGET])

    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Handle missing values
    train_df, fill_values = handle_missing_values(train_df, is_train=True)
    val_df, _ = handle_missing_values(val_df, is_train=False, fill_values=fill_values)
    test_df, _ = handle_missing_values(test_df, is_train=False, fill_values=fill_values)

    # Separate features and target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Encode categoricals
    X_train, X_val, X_test, encoders = encode_categoricals(X_train, y_train, X_val, X_test)

    # Align columns across splits
    feature_cols = list(X_train.columns)
    X_val = X_val.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    # Save splits
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X.to_parquet(os.path.join(output_dir, f"X_{name}.parquet"), index=False)
        y.to_frame().to_parquet(os.path.join(output_dir, f"y_{name}.parquet"), index=False)
        logger.info(f"Saved {name}: X={X.shape}, fraud_rate={y.mean():.4f}")

    # Save encoders + fill values
    artifact = {"encoders": encoders, "fill_values": fill_values, "feature_cols": feature_cols}
    joblib.dump(artifact, encoder_path)
    logger.info(f"Saved encoders to {encoder_path}")

    # Save stats
    stats = {
        "n_features": len(feature_cols),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "feature_cols": feature_cols[:50],  # first 50
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    args = parser.parse_args()

    preprocess(args.input_path, args.output_dir, args.encoder_path, args.stats_path)
