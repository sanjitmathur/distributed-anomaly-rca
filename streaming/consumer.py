"""Streaming consumer — the core inference loop.

Pulls raw transactions (from a Python queue or an HTTP ingest endpoint),
validates with Pydantic, enriches with window/graph features, scores with
the loaded model, checks for concept drift, and emits ScoredTransactions.

This module is imported by the FastAPI inference service.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_loader import load_model, list_models
from models.train_models import _score_model
from pipeline.feature_engineering import FeatureEngineer
from streaming.drift_detector import KSDriftDetector
from streaming.feature_store import GraphFeatureExtractor, SlidingWindowAggregator
from streaming.schemas import RawTransaction, ScoredTransaction
from utils.config import ALL_FEATURES
from utils.logger import get_logger

log = get_logger("consumer")


class StreamingConsumer:
    """Stateful consumer that scores transactions in a streaming fashion.

    Lifecycle:
        consumer = StreamingConsumer(model_name="isolation_forest")
        scored = consumer.process(raw_dict)  # → ScoredTransaction
    """

    def __init__(
        self,
        model_name: str = "isolation_forest",
        drift_window: int = 500,
        drift_alpha: float = 0.01,
    ):
        self._model_name = model_name
        self._fe = FeatureEngineer()

        # Load model
        loaded = load_model(model_name)
        self._model = loaded["model"]
        self._metadata = loaded["metadata"]
        self._model_result = {"model": self._model}
        if "threshold" in self._metadata:
            self._model_result["threshold"] = self._metadata["threshold"]
        if "backend" in self._metadata:
            self._model_result["backend"] = self._metadata["backend"]

        # Streaming feature stores
        self._window_agg = SlidingWindowAggregator(window_sec=600.0)
        self._graph = GraphFeatureExtractor()

        # Drift detector — seeded with uniform [0, 1] as placeholder until
        # warm-up scores accumulate
        self._warmup_scores: list[float] = []
        self._drift: Optional[KSDriftDetector] = None
        self._drift_window = drift_window
        self._drift_alpha = drift_alpha

        # Results buffer for SSE consumers
        self.results_buffer: deque[dict] = deque(maxlen=1000)

        self._total_processed = 0
        self._total_fraud = 0
        log.info("Consumer ready: model=%s", model_name)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, raw: dict) -> ScoredTransaction:
        """Validate → enrich → score → drift-check a single transaction."""

        # 1. Validate & sanitize
        txn = RawTransaction.model_validate(raw)

        # 2. Sliding window features (keyed on device_id)
        window_feats = self._window_agg.push(
            entity_id=txn.device_id,
            amount=txn.Amount,
            ts=txn.timestamp.timestamp(),
        )

        # 3. Graph features
        graph_feats = self._graph.update(txn.source_ip, txn.device_id)

        # 4. Build feature DataFrame
        txn_dict = txn.model_dump(
            include={f"V{i}" for i in range(1, 29)} | {"Amount", "Time"}
        )
        df = pd.DataFrame([txn_dict])
        df = self._fe.transform(df)

        # Append streaming features as columns
        for k, v in window_feats.items():
            df[k] = v
        for k, v in graph_feats.items():
            df[k] = v

        # Select model feature columns (original + engineered)
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feature_cols].values

        # 5. Score
        scores = _score_model(self._model_name, self._model_result, X)
        score = float(scores[0])

        # 6. Drift detection
        drift_info = self._update_drift(score)

        # 7. Assemble result
        is_fraud = score >= 0.5
        self._total_processed += 1
        if is_fraud:
            self._total_fraud += 1

        scored = ScoredTransaction(
            transaction_id=txn.transaction_id,
            timestamp=txn.timestamp,
            Amount=txn.Amount,
            fraud_probability=round(score, 4),
            is_fraud=is_fraud,
            model=self._model_name,
            anomaly_score=round(score, 4),
            drift_detected=drift_info["drift_detected"],
            drift_p_value=drift_info["p_value"],
            window_features=window_feats,
            graph_features=graph_feats,
        )

        self.results_buffer.append(scored.model_dump(mode="json"))
        return scored

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    def _update_drift(self, score: float) -> dict:
        if self._drift is None:
            self._warmup_scores.append(score)
            if len(self._warmup_scores) >= self._drift_window:
                ref = np.array(self._warmup_scores, dtype=np.float64)
                self._drift = KSDriftDetector(
                    reference_sample=ref,
                    window_size=self._drift_window,
                    alpha=self._drift_alpha,
                )
                log.info("Drift detector warm-up complete (%d scores)", len(ref))
            return {"drift_detected": False, "p_value": None, "ks_statistic": 0.0, "window_fill": 0.0}
        return self._drift.push(score)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "total_fraud": self._total_fraud,
            "fraud_rate": round(
                self._total_fraud / max(self._total_processed, 1), 4
            ),
            "drift_alerts": (
                self._drift.total_drift_alerts if self._drift else 0
            ),
            "model": self._model_name,
        }


# ---------------------------------------------------------------------------
# Queue-based consumer loop (for in-process / thread mode)
# ---------------------------------------------------------------------------

def run_consumer_loop(
    queue: Queue,
    consumer: StreamingConsumer,
    callback=None,
):
    """Pull from queue, process, optionally invoke callback per scored txn."""
    log.info("Consumer loop started")
    while True:
        try:
            raw = queue.get(timeout=1.0)
        except Empty:
            continue
        if raw is None:  # sentinel
            log.info("Received sentinel — shutting down consumer loop")
            break
        scored = consumer.process(raw)
        if callback:
            callback(scored)
    log.info("Consumer loop exited — %s", consumer.stats)
