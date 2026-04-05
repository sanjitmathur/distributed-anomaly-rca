"""FastAPI service for fraud detection predictions."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.schemas import (
    TransactionInput,
    PredictionResult,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
)
from models.model_loader import load_model, list_models
from models.train_models import _score_model
from pipeline.feature_engineering import FeatureEngineer
from utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Real-time anomaly detection for financial transactions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load models + feature engineer
# ---------------------------------------------------------------------------
_models_cache: dict = {}
_fe = FeatureEngineer()
_start_time = time.time()


def _get_model(name: str) -> dict:
    if name not in _models_cache:
        try:
            _models_cache[name] = load_model(name)
        except FileNotFoundError:
            raise HTTPException(404, f"Model '{name}' not found. Run train_models.py first.")
    return _models_cache[name]


def _transaction_to_scores(transaction: TransactionInput, model_name: str) -> float:
    """Convert a single transaction to an anomaly score."""
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    df = _fe.transform(df)

    # Use same feature set as training
    from utils.config import ALL_FEATURES
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values

    loaded = _get_model(model_name)
    model = loaded["model"]
    metadata = loaded["metadata"]

    result = {"model": model}
    if "threshold" in metadata:
        result["threshold"] = metadata["threshold"]
    if "backend" in metadata:
        result["backend"] = metadata["backend"]

    scores = _score_model(model_name, result, X)
    return float(scores[0])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        models_loaded=list_models(),
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput, model: str = "isolation_forest"):
    score = _transaction_to_scores(transaction, model)
    return PredictionResult(
        fraud_probability=round(score, 4),
        is_fraud=score >= 0.5,
        model=model,
        anomaly_score=round(score, 4),
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    predictions = []
    for txn in request.transactions:
        score = _transaction_to_scores(txn, request.model)
        predictions.append(PredictionResult(
            fraud_probability=round(score, 4),
            is_fraud=score >= 0.5,
            model=request.model,
            anomaly_score=round(score, 4),
        ))
    fraud_count = sum(1 for p in predictions if p.is_fraud)
    return BatchPredictResponse(
        predictions=predictions,
        fraud_count=fraud_count,
        total=len(predictions),
    )


@app.get("/model_metrics")
def model_metrics():
    """Return cached evaluation metrics for all models."""
    import json
    from utils.config import MODELS_DIR
    metrics = {}
    for name in list_models():
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if meta_path.exists():
            metrics[name] = json.loads(meta_path.read_text())
    return metrics
