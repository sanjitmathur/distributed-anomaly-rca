"""Model evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from utils.logger import get_logger

log = get_logger(__name__)


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = None,
) -> dict:
    """Compute full evaluation metrics.

    Parameters
    ----------
    y_true : binary labels (0=normal, 1=fraud)
    scores : anomaly scores (0-1, higher = more anomalous)
    threshold : decision threshold; if None, uses optimal F1 threshold

    Returns
    -------
    dict with precision, recall, f1, roc_auc, avg_precision,
         confusion_matrix, threshold, fpr, tpr, pr_precisions, pr_recalls
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, scores)

    y_pred = (scores >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, scores)
    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, scores)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "pr_precisions": pr_precisions.tolist(),
        "pr_recalls": pr_recalls.tolist(),
    }
    log.info(
        "Metrics — P: %.3f  R: %.3f  F1: %.3f  AUC: %.3f",
        metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"],
    )
    return metrics
