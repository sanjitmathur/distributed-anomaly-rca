"""
Model evaluation and comparison utilities.

Provides precision/recall/F1/ROC-AUC computation, multi-model leaderboard
generation, and threshold sensitivity analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """
    Evaluator for binary anomaly detection models.

    All methods are stateless and can be called independently.
    """

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
    ) -> dict:
        """
        Compute standard classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels (0/1 or bool).
        y_pred : array-like
            Predicted binary labels.
        y_scores : array-like
            Continuous anomaly scores (higher = more anomalous).

        Returns
        -------
        dict
            Keys: precision, recall, f1, roc_auc, confusion_matrix.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)

        # Handle edge case: single class in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            roc = float("nan")
        else:
            roc = float(roc_auc_score(y_true, y_scores))

        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    @staticmethod
    def compare_models(results: dict) -> pd.DataFrame:
        """
        Build a leaderboard from multiple model evaluation results.

        Parameters
        ----------
        results : dict
            Mapping of model_name -> dict returned by evaluate().

        Returns
        -------
        pd.DataFrame
            Leaderboard sorted by F1 descending, with columns:
            model, precision, recall, f1, roc_auc.
        """
        rows = []
        for model_name, metrics in results.items():
            rows.append({
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            })

        leaderboard = pd.DataFrame(rows)
        leaderboard = leaderboard.sort_values("f1", ascending=False).reset_index(
            drop=True
        )
        return leaderboard

    @staticmethod
    def threshold_analysis(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        thresholds: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Evaluate precision, recall, and F1 across a range of thresholds.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels.
        y_scores : array-like
            Continuous anomaly scores (0-1 expected).
        thresholds : array-like, optional
            Score thresholds to evaluate. Defaults to 50 evenly spaced
            values between 0 and 1.

        Returns
        -------
        pd.DataFrame
            Columns: threshold, precision, recall, f1.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)

        if thresholds is None:
            thresholds = np.linspace(0, 1, 50)

        rows = []
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            rows.append({
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            })

        return pd.DataFrame(rows)
