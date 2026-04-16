"""FastAPI inference service — production fraud detection platform.

Endpoints:
  Scoring:    POST /score_transaction, POST /score_batch
  Streaming:  POST /ingest, GET /stream (SSE)
  Feedback:   POST /feedback, GET /feedback/stats
  Retrain:    POST /retrain, GET /retrain/history
  Monitoring: GET /monitoring, GET /drift, GET /stats
  Risk:       GET /risk/config, PUT /risk/config
  Graph:      GET /graph/clusters, GET /graph/stats, GET /graph/data
  Legacy:     POST /predict, POST /batch_predict, GET /health, GET /model_metrics
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.schemas import (
    TransactionInput, ScoreResponse, FeatureContribution,
    BatchScoreRequest, BatchScoreResponse,
    PredictionResult, BatchPredictRequest, BatchPredictResponse,
    HealthResponse, FeedbackRequest, FeedbackResponse,
    RetrainRequest, RetrainResponse, RiskConfigUpdate,
)
from models.model_loader import load_model, list_models
from models.train_models import _score_model
from monitoring.feedback import FeedbackStore
from monitoring.tracker import MonitoringTracker
from pipeline.feature_engineering import FeatureEngineer
from risk.graph import EntityRiskGraph
from risk.scoring import RiskConfig, RiskScorer
from utils.config import ALL_FEATURES
from utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Financial Fraud Detection Platform",
    description="Production ML fraud scoring with risk graphs, drift monitoring, and retraining",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_models_cache: dict = {}
_fe = FeatureEngineer()
_start_time = time.time()

# Singletons for upgrades
_risk_scorer = RiskScorer()
_entity_graph = EntityRiskGraph()
_monitor = MonitoringTracker()
_feedback_store = FeedbackStore()

# SHAP explainers cache
_shap_cache: dict = {}

# SSE state
_sse_buffer: deque[dict] = deque(maxlen=2000)
_sse_counter = 0
_sse_event: asyncio.Event = None

# Flagged transactions buffer (for investigation panel)
_flagged_buffer: deque[dict] = deque(maxlen=500)


@app.on_event("startup")
async def _startup():
    global _sse_event
    _sse_event = asyncio.Event()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model(name: str) -> dict:
    if name not in _models_cache:
        try:
            _models_cache[name] = load_model(name)
        except FileNotFoundError:
            raise HTTPException(404, f"Model '{name}' not found. Run train_models.py first.")
    return _models_cache[name]


def _prepare_features(transaction: TransactionInput) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Transform transaction -> feature DataFrame + numpy array + column names."""
    df = pd.DataFrame([transaction.model_dump()])
    df = _fe.transform(df)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    return df, X, feature_cols


def _get_shap_explainer(model_name: str, X_background: np.ndarray = None):
    """Lazy-init SHAP explainer for a model."""
    if model_name in _shap_cache:
        return _shap_cache[model_name]

    loaded = _get_model(model_name)
    model = loaded["model"]

    try:
        if model_name == "isolation_forest":
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for other models — needs background data
            if X_background is None:
                X_background = np.zeros((10, model.n_features_in_ if hasattr(model, 'n_features_in_') else 36))
            explainer = shap.KernelExplainer(model.decision_function, X_background[:50])
        _shap_cache[model_name] = explainer
        return explainer
    except Exception as e:
        log.warning("SHAP explainer failed for %s: %s", model_name, e)
        return None


def _compute_top_features(
    model_name: str, X: np.ndarray, feature_names: list[str], top_n: int = 5,
) -> list[FeatureContribution]:
    """Compute SHAP values and return top contributing features."""
    explainer = _get_shap_explainer(model_name)
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim > 1:
            sv = sv[0]

        top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
        contributions = []
        for idx in top_idx:
            if idx < len(feature_names):
                contributions.append(FeatureContribution(
                    feature=feature_names[idx],
                    shap_value=round(float(sv[idx]), 4),
                    direction="increases_risk" if sv[idx] > 0 else "decreases_risk",
                ))
        return contributions
    except Exception as e:
        log.warning("SHAP computation failed: %s", e)
        return []


def _score_single(transaction: TransactionInput, model_name: str) -> float:
    _, X, _ = _prepare_features(transaction)
    loaded = _get_model(model_name)
    result = {"model": loaded["model"]}
    meta = loaded["metadata"]
    if "threshold" in meta:
        result["threshold"] = meta["threshold"]
    if "backend" in meta:
        result["backend"] = meta["backend"]
    return float(_score_model(model_name, result, X)[0])


# ===================================================================
# Upgrade 1 — Real-time Fraud Scoring API
# ===================================================================

@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(
    transaction: TransactionInput,
    model: str = "isolation_forest",
    explain: bool = True,
):
    """Score a single transaction with SHAP explanations and risk assessment."""
    df, X, feature_cols = _prepare_features(transaction)
    score = _score_single(transaction, model)

    # SHAP top features
    top_features = []
    if explain:
        top_features = _compute_top_features(model, X, feature_cols)

    # Risk scoring (Upgrade 6)
    risk = _risk_scorer.score(score)

    # Monitoring (Upgrade 4)
    _monitor.record_prediction(score, score >= 0.5, transaction.model_dump())

    result = ScoreResponse(
        fraud_probability=round(score, 4),
        anomaly_flag=score >= 0.5,
        anomaly_score=round(score, 4),
        risk_score=risk["risk_score"],
        risk_level=risk["risk_level"],
        top_features=top_features,
        model=model,
    )

    # Track flagged
    if score >= 0.5:
        flagged = result.model_dump()
        flagged["transaction_data"] = transaction.model_dump()
        _flagged_buffer.append(flagged)

    return result


@app.post("/score_batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Batch scoring endpoint."""
    results = []
    for txn in request.transactions:
        r = score_transaction(txn, model=request.model, explain=False)
        results.append(r)

    flagged = sum(1 for r in results if r.anomaly_flag)
    avg_risk = np.mean([r.risk_score for r in results]) if results else 0.0

    return BatchScoreResponse(
        results=results, total=len(results), flagged=flagged,
        avg_risk=round(float(avg_risk), 4),
    )


# ===================================================================
# Streaming endpoints (existing, enhanced)
# ===================================================================

@app.post("/ingest")
async def ingest(request: Request):
    """Ingest raw transaction: validate -> score -> graph -> risk -> SSE broadcast."""
    body = await request.json()

    # Validate with streaming schema
    from streaming.schemas import RawTransaction
    txn = RawTransaction.model_validate(body)

    # Score
    txn_dict = txn.model_dump(include={f"V{i}" for i in range(1, 29)} | {"Amount", "Time"})
    ti = TransactionInput(**txn_dict)
    score = _score_single(ti, "isolation_forest")
    is_anomaly = score >= 0.5

    # Entity graph (Upgrade 7)
    graph_signals = _entity_graph.add_transaction(
        card_id=body.get("card_id", "unknown"),
        merchant_id=body.get("merchant_id", "unknown"),
        device_id=txn.device_id,
        ip_address=txn.source_ip,
        location=body.get("location", "unknown"),
        is_anomaly=is_anomaly,
        anomaly_score=score,
    )

    # Risk scoring with graph + velocity signals (Upgrade 6)
    risk = _risk_scorer.score(
        model_score=score,
        graph_features=graph_signals,
        window_features=body.get("window_features", {}),
    )

    # Monitoring (Upgrade 4)
    _monitor.record_prediction(score, is_anomaly, {"Amount": txn.Amount})

    result = {
        "transaction_id": txn.transaction_id,
        "timestamp": txn.timestamp.isoformat(),
        "Amount": txn.Amount,
        "card_id": body.get("card_id", "unknown"),
        "merchant_id": body.get("merchant_id", "unknown"),
        "merchant_category": body.get("merchant_category", "unknown"),
        "device_id": txn.device_id,
        "source_ip": txn.source_ip,
        "location": body.get("location", "unknown"),
        "fraud_probability": round(score, 4),
        "is_fraud": is_anomaly,
        "anomaly_score": round(score, 4),
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "risk_components": risk["components"],
        "graph_signals": graph_signals,
        "model": "isolation_forest",
    }

    # Buffer for SSE + flagged
    global _sse_counter
    _sse_buffer.append(result)
    _sse_counter += 1
    if _sse_event:
        _sse_event.set()

    if is_anomaly:
        _flagged_buffer.append(result)

    return result


@app.get("/stream")
async def stream():
    async def event_generator():
        seen = _sse_counter
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
                for item in list(_sse_buffer)[-new_count:]:
                    yield f"data: {json.dumps(item)}\n\n"
                seen = current
            else:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ===================================================================
# Upgrade 3 — Feedback endpoints
# ===================================================================

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    _feedback_store.record(
        transaction_id=req.transaction_id,
        analyst_decision=req.decision,
        fraud_score=req.fraud_score,
        transaction_data=req.transaction_data,
        analyst_notes=req.analyst_notes,
    )
    stats = _feedback_store.stats
    return FeedbackResponse(
        status="recorded",
        transaction_id=req.transaction_id,
        decision=req.decision,
        total_reviews=stats["total_reviews"],
    )


@app.get("/feedback/stats")
def feedback_stats():
    return _feedback_store.stats


@app.get("/feedback/all")
def feedback_all():
    return _feedback_store.load_all()


@app.get("/flagged")
def get_flagged(limit: int = 50):
    """Return recent flagged transactions for investigation panel."""
    return list(_flagged_buffer)[-limit:]


# ===================================================================
# Upgrade 4 — Monitoring
# ===================================================================

@app.get("/monitoring")
def monitoring_metrics():
    return _monitor.get_dashboard_metrics()


@app.get("/stats")
def stats():
    return _monitor.get_dashboard_metrics()


@app.get("/drift")
def drift_status():
    metrics = _monitor.get_dashboard_metrics()
    return {
        "feature_drift": metrics.get("feature_drift", {}),
        "model_version": metrics["model_version"],
    }


# ===================================================================
# Upgrade 5 — Retraining
# ===================================================================

@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    from models.retrain import retrain_model
    result = retrain_model(model_name=req.model_name, include_feedback=req.include_feedback)
    if result["status"] == "success":
        # Invalidate model cache so next request loads new model
        _models_cache.pop(req.model_name, None)
        _shap_cache.pop(req.model_name, None)
        _monitor.record_retrain(result["version"], result["metrics"])
    return RetrainResponse(**result)


@app.get("/retrain/history")
def retrain_history():
    from models.retrain import get_version_history
    return get_version_history()


# ===================================================================
# Upgrade 6 — Risk config
# ===================================================================

@app.get("/risk/config")
def get_risk_config():
    return _risk_scorer.config.to_dict()


@app.put("/risk/config")
def update_risk_config(update: RiskConfigUpdate):
    kwargs = {k: v for k, v in update.model_dump().items() if v is not None}
    _risk_scorer.update_config(**kwargs)
    return _risk_scorer.config.to_dict()


# ===================================================================
# Upgrade 7 — Entity graph
# ===================================================================

@app.get("/graph/stats")
def graph_stats():
    return _entity_graph.stats


@app.get("/graph/clusters")
def graph_clusters(min_risk: float = 0.3, top_n: int = 10):
    return _entity_graph.get_suspicious_clusters(min_risk=min_risk, top_n=top_n)


@app.get("/graph/data")
def graph_data(max_nodes: int = 150):
    return _entity_graph.get_graph_data_for_viz(max_nodes=max_nodes)


@app.get("/graph/entity/{entity_id:path}")
def graph_entity(entity_id: str):
    return _entity_graph.get_entity_info(entity_id)


# ===================================================================
# Legacy endpoints
# ===================================================================

@app.get("/health", response_model=HealthResponse)
def health():
    from models.retrain import get_current_version
    return HealthResponse(
        status="healthy",
        models_loaded=list_models(),
        version="4.0.0",
        model_version=get_current_version(),
    )


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput, model: str = "isolation_forest"):
    score = _score_single(transaction, model)
    return PredictionResult(
        fraud_probability=round(score, 4), is_fraud=score >= 0.5,
        model=model, anomaly_score=round(score, 4),
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    predictions = []
    for txn in request.transactions:
        score = _score_single(txn, request.model)
        predictions.append(PredictionResult(
            fraud_probability=round(score, 4), is_fraud=score >= 0.5,
            model=request.model, anomaly_score=round(score, 4),
        ))
    return BatchPredictResponse(
        predictions=predictions,
        fraud_count=sum(1 for p in predictions if p.is_fraud),
        total=len(predictions),
    )


@app.get("/model_metrics")
def model_metrics():
    from utils.config import MODELS_DIR
    metrics = {}
    for name in list_models():
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if meta_path.exists():
            metrics[name] = json.loads(meta_path.read_text())
    return metrics
