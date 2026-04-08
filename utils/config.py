"""Central configuration for the fraud detection platform."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_CSV = DATA_DIR / "creditcard_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
RAW_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
ENGINEERED_FEATURES = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "amount_zscore",
    "v_magnitude",
    "v_outlier_count",
]
V_FEATURES = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = V_FEATURES + [
    "Amount",
    "Time",
    "amount_log",
    "hour_sin",
    "hour_cos",
    "amount_zscore",
    "v_magnitude",
    "v_outlier_count",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "isolation_forest": {
        "n_estimators": 200,
        "contamination": 0.002,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lof": {
        "n_neighbors": 20,
        "novelty": True,
        "contamination": 0.002,
    },
    "ocsvm": {
        "kernel": "rbf",
        "nu": 0.002,
        "gamma": "scale",
    },
    "autoencoder": {
        "encoding_dim": 8,
        "hidden_dim": 16,
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "threshold_sigma": 3,
    },
}

MODEL_NAMES = list(MODEL_PARAMS.keys())

# ---------------------------------------------------------------------------
# Streaming features (appended by consumer, not used in model training)
# ---------------------------------------------------------------------------
STREAMING_FEATURES = [
    "txn_count_10m",
    "txn_amount_mean_10m",
    "txn_amount_std_10m",
    "txn_velocity_per_min",
    "ip_degree",
    "device_degree",
    "shared_infra_score",
]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
