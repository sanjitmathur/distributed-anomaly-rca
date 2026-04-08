"""FastAPI inference service — streaming-ready with drift detection + SSE.

Accepts transactions from the producer via POST /ingest, scores them
through the StreamingConsumer pipeline, and exposes an SSE stream at
GET /stream for the dashboard to consume.
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
    title="Financial Fraud Detection — Inference Service",
    description="Streaming-ready anomaly detection with concept drift monitoring",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: lazy-load consumer
# ---------------------------------------------------------------------------
_consumer = None
_fe = FeatureEngineer()
_models_cache: dict = {}
_start_time = time.time()

# SSE broadcast buffer — latest scored transactions for dashboard
_sse_buffer: deque[dict] = deque(maxlen=500)
_sse_counter = 0  # monotonically increasing count of total items added
_sse_event: asyncio.Event = None  # created in startup handler


@app.on_event("startup")
async def _startup():
    global _sse_event
    _sse_event = asyncio.Event()


def _get_consumer():
    global _consumer
    if _consumer is None:
        try:
            from streaming.consumer import StreamingConsumer
            _consumer = StreamingConsumer(model_name="isolation_forest")
            log.info("StreamingConsumer initialized")
        except Exception as e:
            log.warning("StreamingConsumer unavailable: %s — falling back to stateless mode", e)
    return _consumer


def _get_model(name: str) -> dict:
    if name not in _models_cache:
        try:
            _models_cache[name] = load_model(name)
        except FileNotFoundError:
            raise HTTPException(404, f"Model '{name}' not found. Run train_models.py first.")
    return _models_cache[name]


def _transaction_to_scores(transaction: TransactionInput, model_name: str) -> float:
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    df = _fe.transform(df)
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
# Streaming endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(request: Request):
    """Ingest a raw transaction from the producer, score it, broadcast via SSE."""
    body = await request.json()
    consumer = _get_consumer()

    if consumer:
        scored = consumer.process(body)
        result = scored.model_dump(mode="json")
    else:
        # Fallback: stateless scoring
        from streaming.schemas import RawTransaction
        txn = RawTransaction.model_validate(body)
        txn_dict = txn.model_dump(include={f"V{i}" for i in range(1, 29)} | {"Amount", "Time"})
        ti = TransactionInput(**txn_dict)
        score = _transaction_to_scores(ti, "isolation_forest")
        result = {
            "transaction_id": txn.transaction_id,
            "timestamp": txn.timestamp.isoformat(),
            "Amount": txn.Amount,
            "fraud_probability": round(score, 4),
            "is_fraud": score >= 0.5,
            "model": "isolation_forest",
            "anomaly_score": round(score, 4),
            "drift_detected": False,
            "drift_p_value": None,
            "window_features": {},
            "graph_features": {},
        }

    global _sse_counter
    _sse_buffer.append(result)
    _sse_counter += 1
    if _sse_event:
        _sse_event.set()
    return result


@app.get("/stream")
async def stream():
    """Server-Sent Events stream of scored transactions for the dashboard."""

    async def event_generator():
        seen = _sse_counter  # track by global counter, not deque length
        yield f"data: {json.dumps({'type': 'connected', 'buffer_size': len(_sse_buffer)})}\n\n"

        while True:
            if not _sse_event.is_set():
                try:
                    await asyncio.wait_for(_sse_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue
            _sse_event.clear()

            current = _sse_counter
            new_count = current - seen
            if new_count > 0:
                # Grab the last `new_count` items from the deque
                items = list(_sse_buffer)[-new_count:]
                for item in items:
                    yield f"data: {json.dumps(item)}\n\n"
                seen = current
            else:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stats")
def stats():
    """Return consumer pipeline statistics."""
    consumer = _get_consumer()
    if consumer:
        return consumer.stats
    return {"status": "stateless_mode", "total_processed": 0}


@app.get("/drift")
def drift_status():
    """Return current drift detection state."""
    consumer = _get_consumer()
    if consumer and consumer._drift:
        return {
            "active": True,
            "total_alerts": consumer._drift.total_drift_alerts,
            "window_size": consumer._drift._window_size,
            "alpha": consumer._drift._alpha,
        }
    return {"active": False, "reason": "warming up" if consumer else "consumer not initialized"}


# ---------------------------------------------------------------------------
# Original endpoints (preserved for backward compatibility)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        models_loaded=list_models(),
        version="3.0.0",
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
    import json as _json
    from utils.config import MODELS_DIR
    metrics = {}
    for name in list_models():
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if meta_path.exists():
            metrics[name] = _json.loads(meta_path.read_text())
    return metrics
