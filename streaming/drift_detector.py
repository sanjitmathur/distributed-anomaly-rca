"""Concept drift detection via two-sample Kolmogorov-Smirnov test.

Compares the distribution of incoming anomaly scores (or any feature)
against a reference window captured during model training / warm-up.

Trigger: if p-value < alpha for any monitored feature, flag drift.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from threading import Lock
from scipy.stats import ks_2samp

from utils.logger import get_logger

log = get_logger("drift_detector")


class KSDriftDetector:
    """Sliding-window KS-test drift detector.

    Parameters
    ----------
    reference_sample : np.ndarray
        1-D array of reference values (e.g., training anomaly scores).
    window_size : int
        Number of recent observations to compare against reference.
    alpha : float
        Significance level. Drift flagged when p < alpha.
    """

    def __init__(
        self,
        reference_sample: np.ndarray,
        window_size: int = 500,
        alpha: float = 0.01,
    ):
        self._reference = np.asarray(reference_sample, dtype=np.float64)
        self._window: deque[float] = deque(maxlen=window_size)
        self._window_size = window_size
        self._alpha = alpha
        self._lock = Lock()
        self._drift_count = 0
        log.info(
            "DriftDetector initialised: ref_size=%d, window=%d, alpha=%.3f",
            len(self._reference), window_size, alpha,  # noqa: RUF001
        )

    def push(self, value: float) -> dict:
        """Add an observation and return drift status.

        Returns
        -------
        dict with keys:
            - drift_detected: bool
            - p_value: float (KS test p-value, None if window not full)
            - ks_statistic: float
            - window_fill: float (fraction of window filled)
        """
        with self._lock:
            self._window.append(value)
            fill = len(self._window) / self._window_size

            if len(self._window) < self._window_size // 2:
                # Not enough data yet
                return {
                    "drift_detected": False,
                    "p_value": None,
                    "ks_statistic": 0.0,
                    "window_fill": round(fill, 2),
                }

            current = np.array(self._window, dtype=np.float64)
            stat, p_value = ks_2samp(self._reference, current)
            drifted = p_value < self._alpha

            if drifted:
                self._drift_count += 1
                if self._drift_count % 50 == 1:
                    log.warning(
                        "DRIFT DETECTED: KS=%.4f, p=%.6f (α=%.3f) [total alerts: %d]",
                        stat, p_value, self._alpha, self._drift_count,
                    )

        return {
            "drift_detected": drifted,
            "p_value": round(float(p_value), 6),
            "ks_statistic": round(float(stat), 4),
            "window_fill": round(fill, 2),
        }

    @property
    def total_drift_alerts(self) -> int:
        return self._drift_count

    def reset_window(self):
        """Flush the observation window (e.g., after model retrain)."""
        with self._lock:
            self._window.clear()
            self._drift_count = 0
            log.info("Drift window reset")
