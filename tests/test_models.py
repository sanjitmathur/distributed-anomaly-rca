"""Tests for model training and loading."""

import numpy as np
import pytest

from models.train_models import train_all_models
from models.model_loader import load_model, list_models


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((500, 30))
    X_test = rng.standard_normal((100, 30))
    y_test = np.zeros(100, dtype=int)
    y_test[-5:] = 1
    return X_train, X_test, y_test


def test_train_all_returns_results(dummy_data):
    X_train, X_test, y_test = dummy_data
    results = train_all_models(X_train, X_test, y_test, save=False)
    assert "isolation_forest" in results
    assert "lof" in results
    assert "ocsvm" in results
    assert "autoencoder" in results


def test_each_model_has_scores(dummy_data):
    X_train, X_test, y_test = dummy_data
    results = train_all_models(X_train, X_test, y_test, save=False)
    for name, res in results.items():
        assert "scores" in res, f"{name} missing scores"
        assert len(res["scores"]) == len(X_test), f"{name} score length mismatch"


def test_each_model_has_model_object(dummy_data):
    X_train, X_test, y_test = dummy_data
    results = train_all_models(X_train, X_test, y_test, save=False)
    for name, res in results.items():
        assert "model" in res, f"{name} missing model object"
