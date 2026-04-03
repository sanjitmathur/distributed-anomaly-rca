"""
Multi-model anomaly detection engine.

Provides a unified interface to train and predict with IsolationForest,
LocalOutlierFactor, DBSCAN, and an autoencoder (MLPRegressor-based
reconstruction error detector). Includes permutation-based feature
importance for model explainability.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# Base metric columns + all engineered features produced by FeatureEngineer
FEATURE_COLS = [
    # Base metrics
    "error_rate",
    "latency_ms",
    "cpu_pct",
    "memory_pct",
    # Rolling stats
    "error_rate_rolling_mean",
    "error_rate_rolling_std",
    "latency_ms_rolling_mean",
    "latency_ms_rolling_std",
    "cpu_pct_rolling_mean",
    "cpu_pct_rolling_std",
    # Z-scores for base metrics
    "error_rate_zscore",
    "latency_ms_zscore",
    "cpu_pct_zscore",
    "memory_pct_zscore",
    # Time features
    "hour",
    "weekday",
    "is_business_hours",
    # Behavioral features
    "error_rate_velocity",
    "latency_deviation",
]


class AnomalyEngine:
    """
    Ensemble anomaly detection engine supporting multiple model backends.

    Models
    ------
    - isolation_forest : Isolation Forest
    - lof : Local Outlier Factor (novelty mode)
    - dbscan : DBSCAN (core/non-core as anomaly signal)
    - autoencoder : MLPRegressor trained to reconstruct features;
      anomaly score = reconstruction error
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._build_models()
        self._is_fitted = False

    @staticmethod
    def _build_models() -> dict:
        """Instantiate all model objects (unfitted)."""
        return {
            "isolation_forest": IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42,
            ),
            "lof": LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,
                novelty=True,
            ),
            "dbscan": DBSCAN(eps=1.5, min_samples=5),
            "autoencoder": MLPRegressor(
                hidden_layer_sizes=(64, 32, 16, 32, 64),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        }

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate feature columns from DataFrame."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        if not available:
            raise ValueError(
                f"No FEATURE_COLS found in DataFrame. "
                f"Expected some of: {FEATURE_COLS}"
            )
        return df[available].values

    def train(self, df: pd.DataFrame) -> None:
        """
        Fit the scaler and all models on the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (output of FeatureEngineer.transform).
        """
        X_raw = self._get_features(df)
        X = self.scaler.fit_transform(X_raw)

        # Store column list used during training
        self._train_cols = [c for c in FEATURE_COLS if c in df.columns]

        # Isolation Forest
        self.models["isolation_forest"].fit(X)

        # Local Outlier Factor (novelty=True -> use fit, then predict later)
        self.models["lof"].fit(X)

        # DBSCAN - fit and store labels for reference, re-fit at predict time
        self.models["dbscan"].fit(X)

        # Autoencoder - train to reconstruct input features
        self.models["autoencoder"].fit(X, X)

        self._is_fitted = True

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str = "isolation_forest",
    ) -> pd.DataFrame:
        """
        Predict anomaly scores with a specific model.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame.
        model_name : str
            One of: isolation_forest, lof, dbscan, autoencoder.

        Returns
        -------
        pd.DataFrame
            Columns: index, anomaly_score (0-1), is_anomaly (bool).
        """
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted. Call train() first.")
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.models.keys())}"
            )

        X_raw = self._get_features(df)
        X = self.scaler.transform(X_raw)

        scores = self._raw_scores(X, model_name)
        # Normalize to 0-1 via sigmoid (shift so that mean ~0.5 for normals)
        normalized = _sigmoid(scores)

        threshold = np.percentile(normalized, 95)
        is_anomaly = normalized >= threshold

        return pd.DataFrame({
            "index": df.index,
            "anomaly_score": normalized,
            "is_anomaly": is_anomaly,
        })

    def predict_all(self, df: pd.DataFrame) -> dict:
        """
        Run predict for every model.

        Returns
        -------
        dict
            Mapping of model_name -> predictions DataFrame.
        """
        return {name: self.predict(df, name) for name in self.models}

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        model_name: str = "isolation_forest",
        n_repeats: int = 5,
        seed: int = 42,
    ) -> dict:
        """
        Permutation-based feature importance.

        For each feature, permute its values and measure the change in
        mean anomaly score. Features whose permutation causes larger
        score shifts are more important.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame.
        model_name : str
            Model to evaluate.
        n_repeats : int
            Number of permutation repeats per feature.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Mapping of feature_name -> importance (float).
        """
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted. Call train() first.")

        rng = np.random.default_rng(seed)
        X_raw = self._get_features(df)
        X = self.scaler.transform(X_raw)

        baseline_scores = self._raw_scores(X, model_name)
        baseline_mean = np.mean(baseline_scores)

        cols = [c for c in FEATURE_COLS if c in df.columns]
        importances = {}

        for i, col in enumerate(cols):
            diffs = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, i] = rng.permutation(X_perm[:, i])
                perm_scores = self._raw_scores(X_perm, model_name)
                diffs.append(abs(np.mean(perm_scores) - baseline_mean))
            importances[col] = float(np.mean(diffs))

        # Normalize so importances sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def _raw_scores(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Compute raw anomaly scores (higher = more anomalous) for scaled data.

        Each model produces scores on different scales; the caller is
        responsible for normalization.
        """
        if model_name == "isolation_forest":
            # decision_function returns negative for anomalies; invert
            return -self.models["isolation_forest"].decision_function(X)

        elif model_name == "lof":
            # Same convention as IsolationForest
            return -self.models["lof"].decision_function(X)

        elif model_name == "dbscan":
            # Use distance to nearest core point as anomaly score.
            # Non-core points labeled -1 get high scores.
            dbscan = self.models["dbscan"]
            labels = dbscan.fit_predict(X)
            core_mask = dbscan.core_sample_indices_

            if len(core_mask) == 0:
                # All points are outliers
                return np.ones(len(X))

            core_points = X[core_mask]
            scores = np.zeros(len(X))
            for i in range(len(X)):
                dists = np.linalg.norm(core_points - X[i], axis=1)
                scores[i] = np.min(dists)
            # Boost score for noise-labeled points
            scores[labels == -1] += np.percentile(scores, 90)
            return scores

        elif model_name == "autoencoder":
            # Reconstruction error as anomaly score
            X_reconstructed = self.models["autoencoder"].predict(X)
            mse = np.mean((X - X_reconstructed) ** 2, axis=1)
            return mse

        else:
            raise ValueError(f"Unknown model: {model_name}")
