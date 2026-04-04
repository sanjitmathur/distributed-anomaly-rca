"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from pipeline.preprocessing import load_data, preprocess


def _make_fake_csv(tmp_path):
    """Create a small fake credit card CSV for testing."""
    rng = np.random.default_rng(42)
    n = 200
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = np.arange(n) * 100.0
    data["Amount"] = rng.exponential(50, n)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][-4:] = 1  # 4 frauds = 2%
    df = pd.DataFrame(data)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_data(tmp_path):
    path = _make_fake_csv(tmp_path)
    df = load_data(path)
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert len(df) == 200


def test_preprocess_splits(tmp_path):
    path = _make_fake_csv(tmp_path)
    df = load_data(path)
    result = preprocess(df)
    assert "X_train" in result
    assert "X_test" in result
    assert "y_test" in result
    assert "scaler" in result


def test_preprocess_train_has_no_fraud(tmp_path):
    path = _make_fake_csv(tmp_path)
    df = load_data(path)
    result = preprocess(df)
    # Training set should contain only normal transactions
    assert "y_train" not in result or result["y_train"].sum() == 0


def test_preprocess_scaling(tmp_path):
    path = _make_fake_csv(tmp_path)
    df = load_data(path)
    result = preprocess(df)
    means = np.abs(result["X_train"].mean(axis=0))
    assert means.max() < 1.0
