"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_predict_returns_score():
    # Only works if models are trained; skip otherwise
    response = client.get("/health")
    if not response.json()["models_loaded"]:
        import pytest
        pytest.skip("No trained models available")

    txn = {"Time": 0, "Amount": 100.0}  # defaults for V1-V28
    response = client.post("/predict", json=txn, params={"model": "isolation_forest"})
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
