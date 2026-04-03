"""
Feature engineering pipeline for anomaly detection.

Transforms raw infrastructure metrics into enriched feature sets
including rolling statistics, z-scores, time features, and behavioral signals.
"""

import numpy as np
import pandas as pd


ROLLING_COLS = ["error_rate", "latency_ms", "cpu_pct"]
ROLLING_WINDOW = 5


class FeatureEngineer:
    """
    Stateless feature transformer that enriches raw log DataFrames
    with rolling statistics, temporal features, and behavioral metrics.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: timestamp, error_rate, latency_ms,
            cpu_pct, memory_pct.

        Returns
        -------
        pd.DataFrame
            Original columns plus all engineered features, NaNs filled with 0.
        """
        df = df.copy()
        df = self._add_rolling_stats(df)
        df = self._add_zscore_features(df)
        df = self._add_time_features(df)
        df = self._add_behavioral_features(df)
        df = df.fillna(0)
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean and std (window=5) for core metric columns."""
        for col in ROLLING_COLS:
            df[f"{col}_rolling_mean"] = (
                df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            )
            df[f"{col}_rolling_std"] = (
                df[col].rolling(window=ROLLING_WINDOW, min_periods=1).std()
            )
        return df

    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization for all numeric columns present at this stage."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_zscore"] = (df[col] - mean) / std
            else:
                df[f"{col}_zscore"] = 0.0
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour, weekday, and business-hours flag from timestamp."""
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["is_business_hours"] = (
            (ts.dt.weekday < 5) & (ts.dt.hour >= 9) & (ts.dt.hour < 17)
        ).astype(int)
        return df

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Behavioral signals:
        - error_rate_velocity: first difference of error_rate (rate of change)
        - latency_deviation: deviation from rolling mean latency
        """
        df["error_rate_velocity"] = df["error_rate"].diff()
        df["latency_deviation"] = (
            df["latency_ms"] - df["latency_ms"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        return df
