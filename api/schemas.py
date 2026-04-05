"""Pydantic models for API request/response validation."""

from pydantic import BaseModel


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
