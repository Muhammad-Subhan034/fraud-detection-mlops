"""
Inference API: FastAPI service for real-time fraud prediction
- Prometheus metrics exposed at /metrics
- Structured logging
- Input validation with Pydantic
- Health + readiness endpoints
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# PROMETHEUS METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["prediction"]  # "fraud" or "non_fraud"
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Distribution of fraud prediction probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_RECALL_GAUGE = Gauge(
    "fraud_model_recall",
    "Current fraud recall (rolling window)"
)
FALSE_POSITIVE_RATE = Gauge(
    "fraud_false_positive_rate",
    "Current false positive rate"
)
MODEL_VERSION = Gauge(
    "fraud_model_version_info",
    "Model version info",
    ["version"]
)
ACTIVE_REQUESTS = Gauge(
    "fraud_api_active_requests",
    "Currently active requests"
)


# ============================================================
# MODEL LOADING
# ============================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.selector = None
        self.threshold = 0.5
        self.feature_cols = None
        self.version = "unknown"
        self.encoder_artifact = None

    def load(self, model_path: str, encoder_path: str = None):
        logger.info(f"Loading model from {model_path}")
        artifact = joblib.load(model_path)
        self.model     = artifact["model"]
        self.threshold = artifact.get("threshold", 0.5)
        self.selector  = artifact.get("selector", None)

        if encoder_path and os.path.exists(encoder_path):
            self.encoder_artifact = joblib.load(encoder_path)
            self.feature_cols = self.encoder_artifact.get("feature_cols", None)

        self.version = os.environ.get("MODEL_VERSION", "v1")
        MODEL_VERSION.labels(version=self.version).set(1)
        logger.info(f"Model loaded: version={self.version}, threshold={self.threshold}")


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_path   = os.environ.get("MODEL_PATH", "/models/best_model.joblib")
    encoder_path = os.environ.get("ENCODER_PATH", "/models/encoders.joblib")
    model_manager.load(model_path, encoder_path)
    yield
    # Shutdown (cleanup if needed)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using IEEE CIS models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class TransactionFeatures(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD:      Optional[str] = Field(None, description="Product code")
    card1:          Optional[int] = None
    card2:          Optional[float] = None
    card3:          Optional[float] = None
    card4:          Optional[str] = None
    card5:          Optional[float] = None
    card6:          Optional[str] = None
    addr1:          Optional[float] = None
    addr2:          Optional[float] = None
    dist1:          Optional[float] = None
    P_emaildomain:  Optional[str] = None
    R_emaildomain:  Optional[str] = None
    TransactionDT:  Optional[int] = None
    # Allow extra fields for V, C, D, M features
    class Config:
        extra = "allow"

    @validator("TransactionAmt")
    def validate_amount(cls, v):
        if v > 1_000_000:
            raise ValueError("TransactionAmt exceeds maximum allowed value")
        return v


class PredictionResponse(BaseModel):
    transaction_id:   Optional[str] = None
    is_fraud:         bool
    fraud_probability: float
    confidence:       str    # HIGH / MEDIUM / LOW
    threshold:        float
    model_version:    str
    processing_time_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    batch_size:  int
    total_time_ms: float


# ============================================================
# PREDICTION LOGIC
# ============================================================

def preprocess_transaction(data: dict) -> pd.DataFrame:
    """Convert raw transaction dict to model-ready DataFrame."""
    df = pd.DataFrame([data])

    # Log-transform amount
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Time features
    if "TransactionDT" in df.columns:
        df["transaction_hour"] = (df["TransactionDT"] / 3600) % 24
        df["transaction_day"]  = (df["TransactionDT"] / 86400) % 7

    # Encode object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes

    # Align to training feature set
    if model_manager.feature_cols:
        df = df.reindex(columns=model_manager.feature_cols, fill_value=0)

    # Fill NaN
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


def get_confidence_label(prob: float, threshold: float) -> str:
    margin = abs(prob - threshold)
    if margin > 0.3:
        return "HIGH"
    elif margin > 0.1:
        return "MEDIUM"
    return "LOW"


def predict_single(features: dict, transaction_id: str = None) -> PredictionResponse:
    start = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        X = preprocess_transaction(features)

        if model_manager.selector is not None:
            X = model_manager.selector.transform(X)

        prob = float(model_manager.model.predict_proba(X)[0, 1])
        is_fraud = prob >= model_manager.threshold

        # Record metrics
        FRAUD_PREDICTIONS.labels(prediction="fraud" if is_fraud else "non_fraud").inc()
        PREDICTION_CONFIDENCE.observe(prob)

        elapsed_ms = (time.time() - start) * 1000

        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(prob, 6),
            confidence=get_confidence_label(prob, model_manager.threshold),
            threshold=model_manager.threshold,
            model_version=model_manager.version,
            processing_time_ms=round(elapsed_ms, 2),
        )
    finally:
        ACTIVE_REQUESTS.dec()


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_manager.model is not None}


@app.get("/ready")
async def ready():
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_info():
    return {
        "version":   model_manager.version,
        "threshold": model_manager.threshold,
        "n_features": len(model_manager.feature_cols) if model_manager.feature_cols else "unknown",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionFeatures):
    start = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code="200").inc()

    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        features = request.dict()
        tid = features.pop("transaction_id", None)
        result = predict_single(features, transaction_id=tid)

    return result


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    start = time.time()
    predictions = []

    for tx in request.transactions:
        features = tx.dict()
        tid = features.pop("transaction_id", None)
        pred = predict_single(features, transaction_id=tid)
        predictions.append(pred)

    total_ms = (time.time() - start) * 1000
    REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status_code="200").inc()

    return BatchResponse(
        predictions=predictions,
        batch_size=len(predictions),
        total_time_ms=round(total_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=2)
