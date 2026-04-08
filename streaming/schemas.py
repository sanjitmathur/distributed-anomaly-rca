"""Pydantic v2 schemas with robust validation for dirty incoming data.

Handles: missing fields, wrong types, NaN strings, negative amounts,
out-of-range PCA components, and injected garbage columns.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _clamp(v: float, lo: float = -100.0, hi: float = 100.0) -> float:
    if v is None or math.isnan(v) or math.isinf(v):
        return 0.0
    return max(lo, min(hi, v))


class RawTransaction(BaseModel):
    """Ingestion schema — accepts messy producer data and normalises it.

    Design decisions
    ----------------
    * Every numeric field defaults to 0.0 so rows with missing columns survive.
    * A pre-validator coerces stringified NaN / None / empty to 0.0.
    * PCA components are clamped to [-100, 100] (dataset 99.99-percentile).
    * Amount is forced non-negative.
    * Extra fields are silently dropped (model_config forbid → ignore).
    """

    model_config = {"extra": "ignore"}

    transaction_id: str = Field(default="unknown", description="Unique txn identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: str = Field(default="0.0.0.0", description="Originating IP for graph features")
    device_id: str = Field(default="unknown", description="Device fingerprint for graph features")

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

    # --- field-level validators -------------------------------------------

    @field_validator("Amount", mode="before")
    @classmethod
    def _sanitize_amount(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return abs(v)

    @field_validator(
        "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "V8", "V9", "V10", "V11", "V12", "V13", "V14",
        "V15", "V16", "V17", "V18", "V19", "V20", "V21",
        "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Time",
        mode="before",
    )
    @classmethod
    def _sanitize_float(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return _clamp(v)

    @field_validator("source_ip", "device_id", "transaction_id", mode="before")
    @classmethod
    def _sanitize_str(cls, v):
        if v is None:
            return "unknown"
        return str(v).strip()[:256]  # cap length to prevent abuse

    # --- model-level cross-field check ------------------------------------

    @model_validator(mode="after")
    def _final_checks(self):
        if self.Amount > 50_000:
            self.Amount = 50_000.0  # hard ceiling for anomaly scoring
        return self


class ScoredTransaction(BaseModel):
    """Output of the inference service — enriched with scores + drift."""

    transaction_id: str
    timestamp: datetime
    Amount: float
    fraud_probability: float
    is_fraud: bool
    model: str
    anomaly_score: float
    drift_detected: bool = False
    drift_p_value: Optional[float] = None
    window_features: dict = Field(default_factory=dict)
    graph_features: dict = Field(default_factory=dict)
