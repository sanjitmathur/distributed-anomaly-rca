"""Upgrade 4 — Production ML monitoring: anomaly rate, volume, drift, feature stats."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("monitoring")


@dataclass
class TimeSeriesPoint:
    timestamp: float
    value: float


class MonitoringTracker:
    """Tracks production ML metrics over time windows."""

    def __init__(self, window_minutes: int = 60):
        self._lock = Lock()
        self._window_sec = window_minutes * 60
        self._start_time = time.time()

        # Time series buffers
        self._scores: deque[TimeSeriesPoint] = deque(maxlen=50000)
        self._volumes: deque[TimeSeriesPoint] = deque(maxlen=50000)
        self._anomalies: deque[TimeSeriesPoint] = deque(maxlen=50000)

        # Feature distribution tracking (for drift)
        self._feature_windows: dict[str, deque] = {}
        self._feature_baselines: dict[str, dict] = {}

        # Counters
        self._total_processed = 0
        self._total_anomalies = 0
        self._minute_buckets: dict[int, dict] = {}

        # Model version tracking
        self._model_version = "v1.0"
        self._model_loaded_at = datetime.now(timezone.utc).isoformat()
        self._retrain_history: list[dict] = []

    def record_prediction(
        self,
        score: float,
        is_anomaly: bool,
        features: Optional[dict] = None,
    ):
        """Record a single prediction for monitoring."""
        ts = time.time()
        with self._lock:
            self._total_processed += 1
            if is_anomaly:
                self._total_anomalies += 1

            self._scores.append(TimeSeriesPoint(ts, score))
            self._anomalies.append(TimeSeriesPoint(ts, 1.0 if is_anomaly else 0.0))

            # Bucket by minute for time series charts
            minute_key = int(ts // 60)
            if minute_key not in self._minute_buckets:
                self._minute_buckets[minute_key] = {
                    "count": 0, "anomaly_count": 0, "score_sum": 0.0,
                    "timestamp": datetime.fromtimestamp(minute_key * 60, tz=timezone.utc).isoformat(),
                }
            bucket = self._minute_buckets[minute_key]
            bucket["count"] += 1
            bucket["score_sum"] += score
            if is_anomaly:
                bucket["anomaly_count"] += 1

            # Track feature distributions
            if features:
                for fname, fval in features.items():
                    if isinstance(fval, (int, float)) and not np.isnan(fval):
                        if fname not in self._feature_windows:
                            self._feature_windows[fname] = deque(maxlen=2000)
                        self._feature_windows[fname].append(fval)

    def set_feature_baseline(self, feature_name: str, mean: float, std: float):
        """Set baseline distribution for a feature (from training data)."""
        self._feature_baselines[feature_name] = {"mean": mean, "std": std}

    def record_retrain(self, version: str, metrics: dict):
        """Record a model retrain event."""
        with self._lock:
            self._model_version = version
            self._model_loaded_at = datetime.now(timezone.utc).isoformat()
            self._retrain_history.append({
                "version": version,
                "timestamp": self._model_loaded_at,
                "metrics": metrics,
            })
            log.info("Model version updated to %s", version)

    def get_dashboard_metrics(self) -> dict:
        """Return all metrics for the monitoring dashboard."""
        with self._lock:
            now = time.time()
            cutoff = now - self._window_sec

            # Recent scores
            recent_scores = [p.value for p in self._scores if p.timestamp > cutoff]
            recent_anomalies = [p.value for p in self._anomalies if p.timestamp > cutoff]

            # Time series (last N minute buckets)
            sorted_buckets = sorted(self._minute_buckets.items())[-60:]
            time_series = []
            for _, bucket in sorted_buckets:
                avg_score = bucket["score_sum"] / max(bucket["count"], 1)
                anomaly_rate = bucket["anomaly_count"] / max(bucket["count"], 1)
                time_series.append({
                    "timestamp": bucket["timestamp"],
                    "volume": bucket["count"],
                    "anomaly_count": bucket["anomaly_count"],
                    "anomaly_rate": round(anomaly_rate, 4),
                    "avg_score": round(avg_score, 4),
                })

            # Feature drift signals
            feature_drift = {}
            for fname, window in self._feature_windows.items():
                if len(window) < 100:
                    continue
                vals = list(window)
                current_mean = np.mean(vals)
                current_std = np.std(vals)
                baseline = self._feature_baselines.get(fname)
                if baseline:
                    drift_magnitude = abs(current_mean - baseline["mean"]) / max(baseline["std"], 1e-6)
                    feature_drift[fname] = {
                        "current_mean": round(float(current_mean), 4),
                        "current_std": round(float(current_std), 4),
                        "baseline_mean": round(baseline["mean"], 4),
                        "baseline_std": round(baseline["std"], 4),
                        "drift_magnitude": round(float(drift_magnitude), 4),
                        "drifted": float(drift_magnitude) > 2.0,
                    }

            return {
                "total_processed": self._total_processed,
                "total_anomalies": self._total_anomalies,
                "anomaly_rate": round(
                    self._total_anomalies / max(self._total_processed, 1), 4
                ),
                "window_volume": len(recent_scores),
                "window_anomaly_rate": round(
                    sum(recent_anomalies) / max(len(recent_anomalies), 1), 4
                ),
                "avg_score": round(
                    float(np.mean(recent_scores)) if recent_scores else 0.0, 4
                ),
                "score_p95": round(
                    float(np.percentile(recent_scores, 95)) if recent_scores else 0.0, 4
                ),
                "model_version": self._model_version,
                "model_loaded_at": self._model_loaded_at,
                "uptime_seconds": round(now - self._start_time, 1),
                "time_series": time_series,
                "feature_drift": feature_drift,
                "retrain_history": self._retrain_history[-10:],
            }
