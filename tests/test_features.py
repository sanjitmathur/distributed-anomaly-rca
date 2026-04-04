"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_engineering import FeatureEngineer


def _make_sample_df():
    rng = np.random.default_rng(42)
    n = 100
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = np.arange(n) * 3600.0
    data["Amount"] = rng.exponential(50, n)
    return pd.DataFrame(data)


def test_engineer_adds_expected_columns():
    df = _make_sample_df()
    fe = FeatureEngineer()
    result = fe.transform(df)
    expected_new = [
        "amount_log", "hour_sin", "hour_cos",
        "amount_zscore", "v_magnitude", "v_outlier_count",
    ]
    for col in expected_new:
        assert col in result.columns, f"Missing column: {col}"


def test_amount_log_is_positive():
    df = _make_sample_df()
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert (result["amount_log"] >= 0).all()


def test_cyclic_features_bounded():
    df = _make_sample_df()
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_no_nans():
    df = _make_sample_df()
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert not result.isna().any().any()
