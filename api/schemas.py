"""Pydantic models for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Transaction input
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    Time: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0


# ---------------------------------------------------------------------------
# Scoring responses (Upgrade 1 + 6)
# ---------------------------------------------------------------------------

class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class ScoreResponse(BaseModel):
    """Response for /score_transaction — includes SHAP top features + risk."""
    transaction_id: str = "unknown"
    fraud_probability: float
    anomaly_flag: bool
    anomaly_score: float
    risk_score: float = 0.0
    risk_level: str = "MINIMAL"
    top_features: list[FeatureContribution] = []
    model: str = "isolation_forest"


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionInput]
    model: str = "isolation_forest"


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    total: int
    flagged: int
    avg_risk: float


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model: str
    anomaly_score: float


class BatchPredictRequest(BaseModel):
    transactions: list[TransactionInput]
    model: str = "isolation_forest"


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]
    fraud_count: int
    total: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str
    model_version: str = "v1.0"


# ---------------------------------------------------------------------------
# Feedback (Upgrade 3)
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    transaction_id: str
    decision: str = Field(..., pattern="^(fraud|legitimate)$")
    fraud_score: float = 0.0
    transaction_data: dict = Field(default_factory=dict)
    analyst_notes: str = ""


class FeedbackResponse(BaseModel):
    status: str
    transaction_id: str
    decision: str
    total_reviews: int


# ---------------------------------------------------------------------------
# Retraining (Upgrade 5)
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    model_name: str = "isolation_forest"
    include_feedback: bool = True


class RetrainResponse(BaseModel):
    status: str
    version: str = ""
    model: str = ""
    train_time: float = 0.0
    feedback_incorporated: int = 0
    metrics: dict = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Risk config (Upgrade 6)
# ---------------------------------------------------------------------------

class RiskConfigUpdate(BaseModel):
    low_threshold: Optional[float] = None
    medium_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    model_weight: Optional[float] = None
    graph_weight: Optional[float] = None
    velocity_weight: Optional[float] = None
