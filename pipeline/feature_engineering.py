"""Feature engineering for credit card fraud detection."""

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

V_COLS = [f"V{i}" for i in range(1, 29)]


class FeatureEngineer:
    """Stateless feature transformer for credit card transaction data."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: Time, Amount, V1-V28.

        Returns
        -------
        pd.DataFrame with original + engineered features, no NaNs.
        """
        df = df.copy()
        df = self._add_amount_features(df)
        df = self._add_time_features(df)
        df = self._add_v_features(df)
        df = df.fillna(0)
        log.info("Engineered %d features -> %d total columns", 6, len(df.columns))
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["amount_log"] = np.log1p(df["Amount"])
        mean = df["Amount"].mean()
        std = df["Amount"].std()
        df["amount_zscore"] = (df["Amount"] - mean) / std if std > 0 else 0.0
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        hour = (df["Time"] % 86400) / 3600
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return df

    def _add_v_features(self, df: pd.DataFrame) -> pd.DataFrame:
        v_data = df[V_COLS].values
        df["v_magnitude"] = np.linalg.norm(v_data, axis=1)
        means = v_data.mean(axis=0)
        stds = v_data.std(axis=0)
        stds[stds == 0] = 1.0
        z_scores = np.abs((v_data - means) / stds)
        df["v_outlier_count"] = (z_scores > 3).sum(axis=1)
        return df
