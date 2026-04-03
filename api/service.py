import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.engine import AnomalyEngine
from data_pipeline.feature_engineering import FeatureEngineer
from data_pipeline.generator import generate_sample_logs

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    error_rate: float
    latency_ms: float
    cpu_pct: float
    memory_pct: float
    timestamp: Optional[str] = None
    pod: Optional[str] = None


class DetectRequest(BaseModel):
    data: list[LogEntry]


class AnomalyResult(BaseModel):
    index: int
    anomaly_score: float
    is_anomaly: bool
    model: str


class DetectResponse(BaseModel):
    anomalies: list[AnomalyResult]
    model_used: str
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Lazy singleton for the anomaly engine
# ---------------------------------------------------------------------------

_engine: Optional[AnomalyEngine] = None
_feature_engineer: Optional[FeatureEngineer] = None


def _get_engine() -> AnomalyEngine:
    global _engine, _feature_engineer
    if _engine is None:
        raw_logs = generate_sample_logs(5000)
        _feature_engineer = FeatureEngineer()
        features = _feature_engineer.transform(raw_logs)
        _engine = AnomalyEngine()
        _engine.train(features)
    return _engine


def _get_feature_engineer() -> FeatureEngineer:
    _get_engine()
    return _feature_engineer  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Anomaly Detection Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    engine = _get_engine()
    return {"models": list(engine.models.keys())}


@app.post("/detect_anomaly", response_model=DetectResponse)
def detect_anomaly(
    request: DetectRequest,
    model: Optional[str] = Query("isolation_forest", description="Model to use"),
):
    engine = _get_engine()
    fe = _get_feature_engineer()

    raw_df = pd.DataFrame([entry.model_dump() for entry in request.data])
    features = fe.transform(raw_df)

    t0 = time.perf_counter()
    results = engine.predict(features, model_name=model)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    anomalies = [
        AnomalyResult(
            index=int(row["index"]),
            anomaly_score=round(float(row["anomaly_score"]), 4),
            is_anomaly=bool(row["is_anomaly"]),
            model=model,
        )
        for _, row in results.iterrows()
    ]

    return DetectResponse(
        anomalies=anomalies,
        model_used=model,
        processing_time_ms=round(elapsed_ms, 2),
    )
