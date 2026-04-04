"""Data loading, splitting, and scaling for credit card fraud detection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.config import TEST_SIZE, RANDOM_STATE, SAMPLE_CSV
from utils.logger import get_logger

log = get_logger(__name__)


def load_data(path=None) -> pd.DataFrame:
    """Load credit card transaction CSV."""
    path = path or SAMPLE_CSV
    log.info("Loading data from %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d transactions (%d fraud)", len(df), int(df["Class"].sum()))
    return df


def preprocess(df: pd.DataFrame) -> dict:
    """Split and scale the dataset for anomaly detection.

    Training set contains only normal transactions (unsupervised paradigm).
    Test set contains both classes for evaluation.

    Returns dict with keys: X_train, X_test, y_test, scaler, feature_names
    """
    feature_cols = [c for c in df.columns if c != "Class"]

    X = df[feature_cols].values
    y = df["Class"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Keep only normal transactions for training
    normal_mask = y_train == 0
    X_train_normal = X_train_raw[normal_mask]
    log.info(
        "Train: %d normal | Test: %d total (%d fraud)",
        len(X_train_normal), len(X_test_raw), int(y_test.sum()),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test_raw)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }
