"""Upgrade 6 — Risk scoring system with normalization, ranking, and configurable thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("risk_scorer")


@dataclass
class RiskConfig:
    """Configurable risk thresholds and weights."""
    low_threshold: float = 0.3
    medium_threshold: float = 0.5
    high_threshold: float = 0.7
    critical_threshold: float = 0.9
    model_weight: float = 0.6
    graph_weight: float = 0.25
    velocity_weight: float = 0.15

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "RiskConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RiskScorer:
    """Normalizes and ranks transactions by composite risk score.

    Combines model anomaly score, graph-based risk, and velocity signals
    into a single 0-1 risk score with categorical labels.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self._config = config or RiskConfig()
        self._lock = Lock()
        self._score_history: list[float] = []
        self._max_history = 10000

    @property
    def config(self) -> RiskConfig:
        return self._config

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._config, k):
                setattr(self._config, k, v)
        log.info("Risk config updated: %s", self._config.to_dict())

    def score(
        self,
        model_score: float,
        graph_features: Optional[dict] = None,
        window_features: Optional[dict] = None,
    ) -> dict:
        """Compute composite risk score from multiple signals."""
        c = self._config
        graph_features = graph_features or {}
        window_features = window_features or {}

        # Normalize graph signal: shared_infra_score / baseline
        shared_infra = graph_features.get("shared_infra_score", 2)
        graph_risk = min(1.0, (shared_infra - 2) / 8.0)  # 2 is baseline (1 ip + 1 device)
        graph_risk = max(0.0, graph_risk)

        # Normalize velocity signal
        velocity = window_features.get("txn_velocity_per_min", 0)
        txn_count = window_features.get("txn_count_10m", 0)
        velocity_risk = min(1.0, velocity / 30.0)  # >30 txn/min is suspicious
        count_risk = min(1.0, txn_count / 20.0)    # >20 in 10 min is suspicious
        vel_signal = max(velocity_risk, count_risk)

        # Weighted composite
        composite = (
            c.model_weight * model_score
            + c.graph_weight * graph_risk
            + c.velocity_weight * vel_signal
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # Categorical label
        if composite >= c.critical_threshold:
            level = "CRITICAL"
        elif composite >= c.high_threshold:
            level = "HIGH"
        elif composite >= c.medium_threshold:
            level = "MEDIUM"
        elif composite >= c.low_threshold:
            level = "LOW"
        else:
            level = "MINIMAL"

        # Track for ranking
        with self._lock:
            self._score_history.append(composite)
            if len(self._score_history) > self._max_history:
                self._score_history = self._score_history[-self._max_history:]
            # Percentile rank
            rank = float(np.searchsorted(
                np.sort(self._score_history), composite
            ) / len(self._score_history))

        return {
            "risk_score": round(composite, 4),
            "risk_level": level,
            "percentile_rank": round(rank, 4),
            "components": {
                "model_score": round(model_score, 4),
                "graph_risk": round(graph_risk, 4),
                "velocity_risk": round(vel_signal, 4),
            },
        }

    def rank_transactions(self, scored_transactions: list[dict], top_n: int = 20) -> list[dict]:
        """Return top-N highest risk transactions sorted by risk_score."""
        return sorted(
            scored_transactions,
            key=lambda t: t.get("risk_score", 0),
            reverse=True,
        )[:top_n]
