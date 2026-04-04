# Financial Transaction Anomaly Detection Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the K8s anomaly detection demo with a production-grade financial fraud detection platform using the Credit Card Fraud dataset, 4 ML models, SHAP explainability, FastAPI backend, and 6-tab Streamlit dashboard.

**Architecture:** Data flows from CSV → preprocessing → feature engineering → model training → model registry (.joblib). FastAPI serves predictions loading from registry. Streamlit dashboard provides exploration, detection, comparison, visualization, explainability, and real-time simulation tabs.

**Tech Stack:** Python 3.11, scikit-learn, PyTorch (optional), SHAP, FastAPI, Streamlit, Plotly, pandas, joblib

---

## File Map

### Delete (old K8s system):
- `app.py` — old simple Streamlit dashboard
- `data_pipeline/generator.py` — K8s log generator
- `data_pipeline/feature_engineering.py` — K8s feature engineering
- `data_pipeline/__init__.py`
- `models/engine.py` — old multi-model engine
- `models/__init__.py`
- `evaluation/metrics.py` — old evaluator
- `evaluation/__init__.py`
- `api/service.py` — old API
- `api/__init__.py`
- `dashboard/app.py` — old 6-tab dashboard
- `test_detector.py`, `test_gemini.py`, `evaluate_detector.py`, `evaluate_rca.py`

### Create:
- `utils/__init__.py` — empty
- `utils/config.py` — central configuration (paths, hyperparams, feature lists)
- `utils/logger.py` — structured logging setup
- `pipeline/__init__.py` — empty
- `pipeline/preprocessing.py` — load CSV, stratified split, scale
- `pipeline/feature_engineering.py` — engineered features for fraud data
- `models/__init__.py` — empty
- `models/train_models.py` — train all 4 models, save to registry
- `models/model_loader.py` — load trained models + metadata
- `evaluation/__init__.py` — empty
- `evaluation/metrics.py` — precision, recall, F1, ROC-AUC, confusion matrix
- `evaluation/model_comparison.py` — leaderboard, ROC/PR curves, bar charts
- `api/__init__.py` — empty
- `api/main.py` — FastAPI endpoints
- `api/schemas.py` — Pydantic request/response models
- `dashboard/app.py` — 6-tab Streamlit dashboard
- `data/creditcard_sample.csv` — 10K-row stratified sample
- `tests/test_preprocessing.py` — preprocessing tests
- `tests/test_features.py` — feature engineering tests
- `tests/test_models.py` — model training/prediction tests
- `tests/test_api.py` — API endpoint tests
- `docker/Dockerfile` — container definition
- `docker/docker-compose.yml` — multi-service compose
- `requirements.txt` — updated dependencies
- `.gitignore` — updated
- `README.md` — complete rewrite

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Delete: `app.py`, `data_pipeline/`, `test_detector.py`, `test_gemini.py`, `evaluate_detector.py`, `evaluate_rca.py`, `Dockerfile`, `docker-compose.yml`
- Create: `utils/__init__.py`, `utils/config.py`, `utils/logger.py`, `pipeline/__init__.py`, `models/__init__.py`, `evaluation/__init__.py`, `api/__init__.py`, `tests/__init__.py`
- Modify: `.gitignore`, `requirements.txt`

- [ ] **Step 1: Delete old K8s files**

```bash
rm -f app.py test_detector.py test_gemini.py evaluate_detector.py evaluate_rca.py Dockerfile docker-compose.yml
rm -rf data_pipeline/ __pycache__/
```

- [ ] **Step 2: Clean old module contents (keep directories)**

```bash
rm -f models/engine.py models/__init__.py
rm -f evaluation/metrics.py evaluation/__init__.py
rm -f api/service.py api/__init__.py
rm -f dashboard/app.py
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p utils pipeline tests data models/saved docker
```

- [ ] **Step 4: Create empty __init__.py files**

Create empty files at:
- `utils/__init__.py`
- `pipeline/__init__.py`
- `models/__init__.py`
- `evaluation/__init__.py`
- `api/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Write `utils/config.py`**

```python
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
# Preprocessing
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

- [ ] **Step 6: Write `utils/logger.py`**

```python
"""Structured logging configuration."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 7: Update `.gitignore`**

```
# Python
__pycache__/
*.pyc
*.egg-info/
venv/
.venv/

# Models
models/saved/*.joblib
models/saved/*.json

# Data (full dataset)
data/creditcard.csv

# Environment
.env
*.stackdump

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 8: Update `requirements.txt`**

```
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
shap>=0.43.0
joblib>=1.3.0
python-dotenv>=1.0.0
httpx>=0.25.0
```

Note: `torch` excluded — optional dependency detected at runtime. `httpx` needed for FastAPI test client.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: scaffold financial fraud detection platform

Remove old K8s anomaly detection system. Set up new project
structure with config, logging, and updated dependencies."
```

---

### Task 2: Data Pipeline — Preprocessing

**Files:**
- Create: `pipeline/preprocessing.py`, `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing test for `tests/test_preprocessing.py`**

```python
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
    # Should return dict with expected keys
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
    # Scaled data should have near-zero mean
    means = np.abs(result["X_train"].mean(axis=0))
    assert means.max() < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.preprocessing'`

- [ ] **Step 3: Write `pipeline/preprocessing.py`**

```python
"""Data loading, splitting, and scaling for credit card fraud detection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.config import TEST_SIZE, RANDOM_STATE, SAMPLE_CSV
from utils.logger import get_logger

log = get_logger(__name__)


def load_data(path=None) -> pd.DataFrame:
    """Load credit card transaction CSV.

    Parameters
    ----------
    path : str or Path, optional
        CSV file path. Defaults to the bundled sample.

    Returns
    -------
    pd.DataFrame
    """
    path = path or SAMPLE_CSV
    log.info("Loading data from %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d transactions (%d fraud)", len(df), int(df["Class"].sum()))
    return df


def preprocess(df: pd.DataFrame) -> dict:
    """Split and scale the dataset for anomaly detection.

    Training set contains only normal transactions (unsupervised paradigm).
    Test set contains both classes for evaluation.

    Returns
    -------
    dict with keys: X_train, X_test, y_test, scaler, feature_names
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add data preprocessing pipeline

Load credit card CSV, stratified train/test split, train on
normal-only transactions, StandardScaler normalization."
```

---

### Task 3: Feature Engineering Pipeline

**Files:**
- Create: `pipeline/feature_engineering.py`, `tests/test_features.py`

- [ ] **Step 1: Write failing test for `tests/test_features.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write `pipeline/feature_engineering.py`**

```python
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
        log.info("Engineered %d features → %d total columns", 6, len(df.columns))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/feature_engineering.py tests/test_features.py
git commit -m "feat: add feature engineering pipeline

Amount log-transform and z-score, cyclic time encoding,
V-feature magnitude and outlier count."
```

---

### Task 4: Generate Bundled Sample Dataset

**Files:**
- Create: `data/creditcard_sample.csv`, `data/generate_sample.py`

- [ ] **Step 1: Write `data/generate_sample.py`**

This script downloads the full dataset and creates a stratified 10K sample. It is a one-time utility, not part of the production system.

```python
"""One-time script to generate the bundled 10K-row sample.

Usage:
    1. Download creditcard.csv from Kaggle:
       https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Place it in data/creditcard.csv
    3. Run: python data/generate_sample.py
    4. Output: data/creditcard_sample.csv (10K rows, stratified)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/creditcard.csv"
OUTPUT = "data/creditcard_sample.csv"
SAMPLE_SIZE = 10000


def main():
    df = pd.read_csv(INPUT)
    print(f"Full dataset: {len(df)} rows, {int(df['Class'].sum())} frauds")

    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    # Keep all frauds (492), sample rest to reach 10K
    n_normal = SAMPLE_SIZE - len(fraud)
    normal_sample = normal.sample(n=n_normal, random_state=42)

    sample = pd.concat([normal_sample, fraud]).sample(frac=1, random_state=42)
    sample.to_csv(OUTPUT, index=False)
    print(f"Sample: {len(sample)} rows, {int(sample['Class'].sum())} frauds ({sample['Class'].mean():.2%})")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download the dataset from Kaggle and generate the sample**

```bash
# Option A: If you have kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# Option B: Manual download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in data/

# Then generate sample:
python data/generate_sample.py
```

- [ ] **Step 3: Verify the sample**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/creditcard_sample.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns[:5])} ... {list(df.columns[-3:])}')
print(f'Frauds: {int(df[\"Class\"].sum())} ({df[\"Class\"].mean():.2%})')
"
```

Expected: ~10,000 rows, 492 frauds (~4.9%)

- [ ] **Step 4: Commit**

```bash
git add data/creditcard_sample.csv data/generate_sample.py
git commit -m "feat: add 10K-row stratified sample dataset

All 492 fraud transactions preserved, 9508 normal sampled.
Full dataset downloadable separately from Kaggle."
```

---

### Task 5: Model Training & Registry

**Files:**
- Create: `models/train_models.py`, `models/model_loader.py`, `tests/test_models.py`

- [ ] **Step 1: Write failing test for `tests/test_models.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write `models/train_models.py`**

```python
"""Train all anomaly detection models and save to registry."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from utils.config import MODEL_PARAMS, MODELS_DIR
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Autoencoder — PyTorch preferred, sklearn fallback
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class _TorchAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=16, encoding_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoding_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    TORCH_AVAILABLE = True
    log.info("PyTorch available — using deep autoencoder")
except ImportError:
    TORCH_AVAILABLE = False
    log.info("PyTorch not available — using sklearn autoencoder fallback")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _train_isolation_forest(X_train: np.ndarray) -> dict:
    params = MODEL_PARAMS["isolation_forest"]
    model = IsolationForest(**params)
    model.fit(X_train)
    return {"model": model}


def _train_lof(X_train: np.ndarray) -> dict:
    params = MODEL_PARAMS["lof"]
    model = LocalOutlierFactor(**params)
    model.fit(X_train)
    return {"model": model}


def _train_ocsvm(X_train: np.ndarray) -> dict:
    params = MODEL_PARAMS["ocsvm"]
    model = OneClassSVM(**params)
    model.fit(X_train)
    return {"model": model}


def _train_autoencoder(X_train: np.ndarray) -> dict:
    p = MODEL_PARAMS["autoencoder"]
    input_dim = X_train.shape[1]

    if TORCH_AVAILABLE:
        device = torch.device("cpu")
        model = _TorchAutoencoder(input_dim, p["hidden_dim"], p["encoding_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
        criterion = nn.MSELoss()
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True)

        model.train()
        for epoch in range(p["epochs"]):
            for (batch,) in loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute training reconstruction errors for threshold
        model.eval()
        with torch.no_grad():
            recon = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        train_errors = np.mean((X_train - recon) ** 2, axis=1)
    else:
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=(p["hidden_dim"], p["encoding_dim"], p["hidden_dim"]),
            max_iter=p["epochs"],
            random_state=42,
            early_stopping=True,
        )
        model.fit(X_train, X_train)
        recon = model.predict(X_train)
        train_errors = np.mean((X_train - recon) ** 2, axis=1)

    threshold = train_errors.mean() + p["threshold_sigma"] * train_errors.std()
    return {"model": model, "threshold": threshold, "backend": "torch" if TORCH_AVAILABLE else "sklearn"}


_TRAINERS = {
    "isolation_forest": _train_isolation_forest,
    "lof": _train_lof,
    "ocsvm": _train_ocsvm,
    "autoencoder": _train_autoencoder,
}


def _score_model(name: str, result: dict, X_test: np.ndarray) -> np.ndarray:
    """Compute anomaly scores (0-1, higher = more anomalous)."""
    model = result["model"]

    if name in ("isolation_forest", "lof", "ocsvm"):
        raw = model.decision_function(X_test)
        return _sigmoid(-raw)  # negate: sklearn convention is negative = anomaly

    # Autoencoder
    if result.get("backend") == "torch":
        import torch
        model.eval()
        with torch.no_grad():
            recon = model(torch.FloatTensor(X_test)).numpy()
    else:
        recon = model.predict(X_test)

    errors = np.mean((X_test - recon) ** 2, axis=1)
    threshold = result["threshold"]
    return _sigmoid(errors - threshold)


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> dict:
    """Train all models and return results dict.

    Parameters
    ----------
    X_train : array of shape (n_train, n_features) — normal transactions only
    X_test : array of shape (n_test, n_features)
    y_test : array of shape (n_test,) — ground truth labels
    save : bool — whether to persist to models/saved/

    Returns
    -------
    dict mapping model name -> {"model", "scores", "train_time", ...}
    """
    results = {}

    for name, trainer in _TRAINERS.items():
        log.info("Training %s ...", name)
        t0 = time.time()
        result = trainer(X_train)
        train_time = time.time() - t0
        log.info("  %s trained in %.1fs", name, train_time)

        scores = _score_model(name, result, X_test)
        result["scores"] = scores
        result["train_time"] = train_time
        results[name] = result

        if save:
            _save_model(name, result)

    return results


def _save_model(name: str, result: dict):
    """Save model artifact and metadata to registry."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{name}.joblib"
    meta_path = MODELS_DIR / f"{name}_meta.json"

    joblib.dump(result["model"], model_path)

    meta = {
        "name": name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_time_seconds": round(result["train_time"], 2),
    }
    if "threshold" in result:
        meta["threshold"] = float(result["threshold"])
    if "backend" in result:
        meta["backend"] = result["backend"]

    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("  Saved %s → %s", name, model_path)
```

- [ ] **Step 4: Write `models/model_loader.py`**

```python
"""Load trained models from the registry."""

import json
from pathlib import Path

import joblib

from utils.config import MODELS_DIR, MODEL_NAMES
from utils.logger import get_logger

log = get_logger(__name__)


def list_models() -> list[str]:
    """Return names of models with saved artifacts."""
    available = []
    for name in MODEL_NAMES:
        if (MODELS_DIR / f"{name}.joblib").exists():
            available.append(name)
    return available


def load_model(name: str) -> dict:
    """Load a trained model and its metadata.

    Returns
    -------
    dict with keys: model, metadata
    """
    model_path = MODELS_DIR / f"{name}.joblib"
    meta_path = MODELS_DIR / f"{name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No saved model: {name}. Run train_models.py first.")

    model = joblib.load(model_path)
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    log.info("Loaded model: %s", name)
    return {"model": model, "metadata": metadata}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add models/train_models.py models/model_loader.py tests/test_models.py
git commit -m "feat: add model training pipeline and registry

Isolation Forest, LOF, One-Class SVM, Autoencoder (PyTorch/sklearn).
Unified scoring interface, joblib persistence, metadata tracking."
```

---

### Task 6: Evaluation & Model Comparison

**Files:**
- Create: `evaluation/metrics.py`, `evaluation/model_comparison.py`

- [ ] **Step 1: Write `evaluation/metrics.py`**

```python
"""Model evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from utils.logger import get_logger

log = get_logger(__name__)


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = None,
) -> dict:
    """Compute full evaluation metrics.

    Parameters
    ----------
    y_true : binary labels (0=normal, 1=fraud)
    scores : anomaly scores (0-1, higher = more anomalous)
    threshold : decision threshold; if None, uses optimal F1 threshold

    Returns
    -------
    dict with precision, recall, f1, roc_auc, avg_precision,
         confusion_matrix, threshold, fpr, tpr, pr_precisions, pr_recalls
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, scores)

    y_pred = (scores >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, scores)
    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, scores)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "pr_precisions": pr_precisions.tolist(),
        "pr_recalls": pr_recalls.tolist(),
    }
    log.info(
        "Metrics — P: %.3f  R: %.3f  F1: %.3f  AUC: %.3f",
        metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"],
    )
    return metrics
```

- [ ] **Step 2: Write `evaluation/model_comparison.py`**

```python
"""Model comparison visualizations using Plotly."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_leaderboard(all_metrics: dict) -> pd.DataFrame:
    """Create a leaderboard DataFrame sorted by F1.

    Parameters
    ----------
    all_metrics : dict mapping model_name -> metrics dict

    Returns
    -------
    pd.DataFrame with columns: Model, Precision, Recall, F1, ROC-AUC, Avg Precision
    """
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Avg Precision": round(m["avg_precision"], 4),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def plot_roc_curves(all_metrics: dict) -> go.Figure:
    """Overlaid ROC curves for all models."""
    fig = go.Figure()
    for name, m in all_metrics.items():
        fig.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"],
            mode="lines",
            name=f"{name} (AUC={m['roc_auc']:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )
    return fig


def plot_pr_curves(all_metrics: dict) -> go.Figure:
    """Overlaid Precision-Recall curves for all models."""
    fig = go.Figure()
    for name, m in all_metrics.items():
        fig.add_trace(go.Scatter(
            x=m["pr_recalls"], y=m["pr_precisions"],
            mode="lines",
            name=f"{name} (AP={m['avg_precision']:.3f})",
        ))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
    )
    return fig


def plot_metric_bars(all_metrics: dict) -> go.Figure:
    """Bar chart comparing F1, Precision, Recall across models."""
    models = list(all_metrics.keys())
    fig = go.Figure()
    for metric_name in ["Precision", "Recall", "F1"]:
        key = metric_name.lower() if metric_name != "F1" else "f1"
        values = [all_metrics[m][key] for m in models]
        fig.add_trace(go.Bar(name=metric_name, x=models, y=values))
    fig.update_layout(
        title="Model Comparison",
        barmode="group",
        yaxis_title="Score",
        height=400,
    )
    return fig
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/metrics.py evaluation/model_comparison.py
git commit -m "feat: add evaluation metrics and model comparison charts

Precision, recall, F1, ROC-AUC, average precision, confusion matrix.
Leaderboard table, overlaid ROC/PR curves, metric bar charts."
```

---

### Task 7: FastAPI Backend

**Files:**
- Create: `api/main.py`, `api/schemas.py`, `tests/test_api.py`

- [ ] **Step 1: Write `api/schemas.py`**

```python
"""Pydantic models for API request/response validation."""

from pydantic import BaseModel


class TransactionInput(BaseModel):
    Time: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0


class PredictionResult(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model: str
    anomaly_score: float


class BatchPredictRequest(BaseModel):
    transactions: list[TransactionInput]
    model: str = "isolation_forest"


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]
    fraud_count: int
    total: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str
```

- [ ] **Step 2: Write `api/main.py`**

```python
"""FastAPI service for fraud detection predictions."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.schemas import (
    TransactionInput,
    PredictionResult,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
)
from models.model_loader import load_model, list_models
from models.train_models import _score_model
from pipeline.feature_engineering import FeatureEngineer
from utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Real-time anomaly detection for financial transactions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load models + feature engineer
# ---------------------------------------------------------------------------
_models_cache: dict = {}
_fe = FeatureEngineer()
_start_time = time.time()


def _get_model(name: str) -> dict:
    if name not in _models_cache:
        try:
            _models_cache[name] = load_model(name)
        except FileNotFoundError:
            raise HTTPException(404, f"Model '{name}' not found. Run train_models.py first.")
    return _models_cache[name]


def _transaction_to_scores(transaction: TransactionInput, model_name: str) -> float:
    """Convert a single transaction to an anomaly score."""
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    df = _fe.transform(df)

    # Use same feature set as training
    from utils.config import ALL_FEATURES
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values

    loaded = _get_model(model_name)
    model = loaded["model"]
    metadata = loaded["metadata"]

    result = {"model": model}
    if "threshold" in metadata:
        result["threshold"] = metadata["threshold"]
    if "backend" in metadata:
        result["backend"] = metadata["backend"]

    scores = _score_model(model_name, result, X)
    return float(scores[0])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        models_loaded=list_models(),
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput, model: str = "isolation_forest"):
    score = _transaction_to_scores(transaction, model)
    return PredictionResult(
        fraud_probability=round(score, 4),
        is_fraud=score >= 0.5,
        model=model,
        anomaly_score=round(score, 4),
    )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    predictions = []
    for txn in request.transactions:
        score = _transaction_to_scores(txn, request.model)
        predictions.append(PredictionResult(
            fraud_probability=round(score, 4),
            is_fraud=score >= 0.5,
            model=request.model,
            anomaly_score=round(score, 4),
        ))
    fraud_count = sum(1 for p in predictions if p.is_fraud)
    return BatchPredictResponse(
        predictions=predictions,
        fraud_count=fraud_count,
        total=len(predictions),
    )


@app.get("/model_metrics")
def model_metrics():
    """Return cached evaluation metrics for all models."""
    import json
    from utils.config import MODELS_DIR
    metrics = {}
    for name in list_models():
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if meta_path.exists():
            metrics[name] = json.loads(meta_path.read_text())
    return metrics
```

- [ ] **Step 3: Write `tests/test_api.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_api.py -v`
Expected: `test_health` PASS, `test_predict_returns_score` PASS or SKIP

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/schemas.py tests/test_api.py
git commit -m "feat: add FastAPI prediction service

POST /predict, POST /batch_predict, GET /model_metrics, GET /health.
Pydantic validation, model loading from registry, CORS enabled."
```

---

### Task 8: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`

This is the largest task. The dashboard has 6 tabs. All code goes in one file since Streamlit runs as a single script.

- [ ] **Step 1: Write `dashboard/app.py`**

```python
"""Financial Fraud Detection Dashboard — 6-tab Streamlit application."""

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, preprocess
from pipeline.feature_engineering import FeatureEngineer
from models.train_models import train_all_models, _score_model
from models.model_loader import load_model, list_models
from evaluation.metrics import compute_metrics, find_optimal_threshold
from evaluation.model_comparison import (
    create_leaderboard,
    plot_roc_curves,
    plot_pr_curves,
    plot_metric_bars,
)
from utils.config import ALL_FEATURES, MODEL_NAMES, SAMPLE_CSV

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df_raw = None
    st.session_state.preprocessed = None
    st.session_state.df_engineered = None
    st.session_state.model_results = None
    st.session_state.all_metrics = None


def _initialize():
    """Load data, engineer features, train models on first run."""
    if st.session_state.initialized:
        return

    with st.spinner("Loading dataset..."):
        df = load_data()
        st.session_state.df_raw = df

    with st.spinner("Engineering features..."):
        fe = FeatureEngineer()
        df_eng = fe.transform(df.drop(columns=["Class"]))
        df_eng["Class"] = df["Class"].values
        st.session_state.df_engineered = df_eng

    with st.spinner("Preprocessing..."):
        result = preprocess(df_eng)
        st.session_state.preprocessed = result

    with st.spinner("Training models (this may take a minute)..."):
        model_results = train_all_models(
            result["X_train"], result["X_test"], result["y_test"], save=True,
        )
        st.session_state.model_results = model_results

    with st.spinner("Evaluating models..."):
        all_metrics = {}
        for name, res in model_results.items():
            all_metrics[name] = compute_metrics(result["y_test"], res["scores"])
        st.session_state.all_metrics = all_metrics

    st.session_state.initialized = True


_initialize()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.caption("Production-grade ML Platform")
    st.markdown("---")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.metric("Transactions", f"{len(df):,}")
        st.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
        st.metric("Fraud Rate", f"{df['Class'].mean():.2%}")
    st.markdown("---")
    st.markdown("**Models:** 4 trained")
    st.markdown(
        "Isolation Forest · LOF · "
        "One-Class SVM · Autoencoder"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("🛡️ Real-Time Financial Fraud Detection Platform")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Explorer",
    "🔍 Anomaly Detection",
    "📈 Model Comparison",
    "🗺️ Visualization",
    "🧠 Explainability",
    "⚡ Real-Time Simulation",
])

# ===================== Tab 1: Dataset Explorer =====================
with tab1:
    st.subheader("Dataset Overview")
    df = st.session_state.df_raw

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns) - 1}")
        col3.metric("Fraud Count", f"{int(df['Class'].sum()):,}")
        col4.metric("Fraud Ratio", f"{df['Class'].mean():.2%}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Class Distribution")
            class_counts = df["Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Label"] = class_counts["Class"].map({0: "Normal", 1: "Fraud"})
            fig = px.bar(
                class_counts, x="Label", y="Count", color="Label",
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                text="Count",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Feature Distribution")
            feature = st.selectbox(
                "Select feature",
                ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)],
            )
            fig = px.histogram(
                df, x=feature, color="Class", nbins=50,
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                barmode="overlay", opacity=0.7,
                labels={"Class": "Label"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Correlation Heatmap")
        corr_features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 11)]
        corr = df[corr_features].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 2: Anomaly Detection =====================
with tab2:
    st.subheader("Run Anomaly Detection")

    if st.session_state.model_results:
        model_name = st.selectbox("Select Model", MODEL_NAMES, key="detect_model")
        results = st.session_state.model_results
        preprocessed = st.session_state.preprocessed
        scores = results[model_name]["scores"]
        y_test = preprocessed["y_test"]

        threshold = st.slider(
            "Detection Threshold", 0.0, 1.0, 0.5, 0.01, key="detect_thresh",
        )

        y_pred = (scores >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())

        c1, c2, c3, c4 = st.columns(4)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        c1.metric("Precision", f"{precision:.3f}")
        c2.metric("Recall", f"{recall:.3f}")
        c3.metric("F1 Score", f"{f1:.3f}")
        c4.metric("Flagged", f"{int(y_pred.sum()):,}")

        st.markdown("---")
        st.subheader("Flagged Transactions")
        flagged_idx = np.where(y_pred == 1)[0]
        if len(flagged_idx) > 0:
            flagged_df = pd.DataFrame({
                "Index": flagged_idx,
                "Anomaly Score": scores[flagged_idx].round(4),
                "Actual": ["Fraud" if y_test[i] == 1 else "Normal" for i in flagged_idx],
            })
            st.dataframe(flagged_df.head(50), use_container_width=True)
        else:
            st.info("No transactions flagged at this threshold.")

# ===================== Tab 3: Model Comparison =====================
with tab3:
    st.subheader("Model Performance Comparison")

    if st.session_state.all_metrics:
        metrics = st.session_state.all_metrics

        st.markdown("#### Leaderboard")
        leaderboard = create_leaderboard(metrics)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_roc_curves(metrics), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pr_curves(metrics), use_container_width=True)

        st.plotly_chart(plot_metric_bars(metrics), use_container_width=True)

# ===================== Tab 4: Visualization =====================
with tab4:
    st.subheader("Anomaly Visualization")

    if st.session_state.model_results and st.session_state.preprocessed:
        viz_model = st.selectbox("Select Model", MODEL_NAMES, key="viz_model")
        scores = st.session_state.model_results[viz_model]["scores"]
        y_test = st.session_state.preprocessed["y_test"]
        X_test = st.session_state.preprocessed["X_test"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### PCA Projection")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_test[:2000])
            pca_df = pd.DataFrame({
                "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test[:2000]],
                "Score": scores[:2000],
            })
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color="Label",
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                opacity=0.6, hover_data=["Score"],
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Anomaly Score Distribution")
            score_df = pd.DataFrame({
                "Score": scores,
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
            })
            fig = px.histogram(
                score_df, x="Score", color="Label", nbins=50,
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                barmode="overlay", opacity=0.7,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Transaction Timeline")
        timeline_df = pd.DataFrame({
            "Index": np.arange(len(scores)),
            "Anomaly Score": scores,
            "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
        })
        fig = px.scatter(
            timeline_df, x="Index", y="Anomaly Score", color="Label",
            color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
            opacity=0.5,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 5: Explainability =====================
with tab5:
    st.subheader("Model Explainability (SHAP)")

    if st.session_state.model_results and st.session_state.preprocessed:
        explain_model = st.selectbox(
            "Select Model", ["isolation_forest", "lof"], key="explain_model",
            help="SHAP works best with tree-based and distance-based models",
        )

        preprocessed = st.session_state.preprocessed
        X_test = preprocessed["X_test"]
        feature_names = preprocessed["feature_names"]
        scores = st.session_state.model_results[explain_model]["scores"]

        n_explain = st.slider("Samples to explain", 50, 500, 100, 50)

        if st.button("🧠 Compute SHAP Values", type="primary"):
            with st.spinner("Computing SHAP values (this may take a moment)..."):
                import shap

                model_obj = st.session_state.model_results[explain_model]["model"]

                if explain_model == "isolation_forest":
                    explainer = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(X_test[:n_explain])
                else:
                    explainer = shap.KernelExplainer(
                        model_obj.decision_function,
                        shap.sample(X_test, 50),
                    )
                    shap_values = explainer.shap_values(X_test[:n_explain])

                st.session_state.shap_values = shap_values
                st.session_state.shap_X = X_test[:n_explain]
                st.session_state.shap_features = feature_names

        if "shap_values" in st.session_state:
            shap_vals = st.session_state.shap_values
            shap_X = st.session_state.shap_X
            feat_names = st.session_state.shap_features

            st.markdown("#### Feature Importance")
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feat_names[:len(mean_abs)],
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(
                importance_df, x="Mean |SHAP|", y="Feature",
                orientation="h", color="Mean |SHAP|",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Individual Transaction Explanation")
            flagged_idx = np.where(scores[:n_explain] > 0.5)[0]
            if len(flagged_idx) > 0:
                selected = st.selectbox(
                    "Select flagged transaction",
                    flagged_idx,
                    format_func=lambda i: f"Transaction {i} (score: {scores[i]:.3f})",
                )
                txn_shap = shap_vals[selected]
                txn_df = pd.DataFrame({
                    "Feature": feat_names[:len(txn_shap)],
                    "SHAP Value": txn_shap,
                }).sort_values("SHAP Value", key=abs, ascending=True).tail(10)

                fig = px.bar(
                    txn_df, x="SHAP Value", y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                )
                fig.update_layout(height=400, title=f"Why Transaction {selected} Was Flagged")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top contributing features:**")
                top = txn_df.tail(3).iloc[::-1]
                for _, row in top.iterrows():
                    direction = "increases" if row["SHAP Value"] > 0 else "decreases"
                    st.markdown(
                        f"- **{row['Feature']}** {direction} fraud risk "
                        f"(SHAP: {row['SHAP Value']:+.4f})"
                    )
            else:
                st.info("No flagged transactions in the explained sample.")

# ===================== Tab 6: Real-Time Simulation =====================
with tab6:
    st.subheader("Real-Time Transaction Monitoring")

    if st.session_state.model_results and st.session_state.preprocessed:
        sim_model = st.selectbox("Select Model", MODEL_NAMES, key="sim_model")

        c1, c2 = st.columns([1, 3])
        with c1:
            n_transactions = st.number_input("Transactions to simulate", 20, 200, 50)
            speed = st.slider("Speed (ms between transactions)", 50, 500, 100)

        if st.button("▶️ Start Simulation", type="primary"):
            X_test = st.session_state.preprocessed["X_test"]
            y_test = st.session_state.preprocessed["y_test"]
            model_result = st.session_state.model_results[sim_model]

            # Sample random transactions
            rng = np.random.default_rng()
            indices = rng.choice(len(X_test), n_transactions, replace=False)

            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            alert_placeholder = st.empty()

            scores_history = []
            labels_history = []
            alerts = []

            for i, idx in enumerate(indices):
                x = X_test[idx : idx + 1]
                score = float(_score_model(sim_model, model_result, x)[0])
                label = int(y_test[idx])

                scores_history.append(score)
                labels_history.append(label)

                if score > 0.5:
                    alerts.append(f"🚨 Transaction {i+1}: score={score:.3f} ({'FRAUD' if label == 1 else 'FALSE ALARM'})")

                # Update chart
                sim_df = pd.DataFrame({
                    "Transaction": range(1, len(scores_history) + 1),
                    "Anomaly Score": scores_history,
                    "Label": ["Fraud" if l == 1 else "Normal" for l in labels_history],
                })
                fig = px.scatter(
                    sim_df, x="Transaction", y="Anomaly Score",
                    color="Label",
                    color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                              annotation_text="Threshold")
                fig.update_layout(height=350)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Update metrics
                flagged = sum(1 for s in scores_history if s > 0.5)
                actual = sum(labels_history)
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Processed", i + 1)
                    m2.metric("Flagged", flagged)
                    m3.metric("Actual Fraud", actual)
                    m4.metric("Alert Rate", f"{flagged/(i+1):.1%}")

                # Update alerts
                if alerts:
                    with alert_placeholder.container():
                        st.markdown("#### Recent Alerts")
                        for alert in alerts[-5:]:
                            st.markdown(alert)

                time.sleep(speed / 1000)

            st.success(f"Simulation complete: {n_transactions} transactions processed.")
```

- [ ] **Step 2: Verify dashboard launches**

Run: `streamlit run dashboard/app.py --server.headless true`
Expected: App launches on port 8501, loads data, trains models on first run.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add 6-tab fraud detection dashboard

Dataset explorer, anomaly detection with threshold tuning,
model comparison (ROC/PR/leaderboard), PCA visualization,
SHAP explainability, real-time transaction simulation."
```

---

### Task 9: Docker & Deployment

**Files:**
- Create: `docker/Dockerfile`, `docker/docker-compose.yml`
- Modify: `.github/workflows/keep-alive.yml`

- [ ] **Step 1: Write `docker/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models on build
RUN python -c "
from pipeline.preprocessing import load_data, preprocess
from pipeline.feature_engineering import FeatureEngineer
from models.train_models import train_all_models

df = load_data()
fe = FeatureEngineer()
df_eng = fe.transform(df.drop(columns=['Class']))
df_eng['Class'] = df['Class'].values
result = preprocess(df_eng)
train_all_models(result['X_train'], result['X_test'], result['y_test'])
print('Models trained successfully')
"

EXPOSE 8501 8000

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.headless=true"]
```

- [ ] **Step 2: Write `docker/docker-compose.yml`**

```yaml
version: "3.9"

services:
  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    restart: unless-stopped

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    restart: unless-stopped
```

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile docker/docker-compose.yml
git commit -m "feat: add Docker deployment configuration

Multi-service compose: dashboard on 8501, API on 8000.
Models pre-trained during build."
```

---

### Task 10: README & Final Polish

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write `README.md`**

Complete rewrite covering: project overview, architecture diagram, dataset description, quick start (local + Docker), model comparison results table, API docs, dashboard screenshots section, project structure, deployment guide, tech stack, cost breakdown.

The README should include:
- Title with badges (Python, FastAPI, Streamlit, License)
- One-line description
- Live demo link
- Architecture diagram (ASCII)
- Features list
- Quick start (local + Docker)
- Model performance table (filled in after training)
- API endpoint table with example curl commands
- Project structure tree
- Tech stack table
- License

- [ ] **Step 2: Clean up old files**

Remove any remaining old files not caught in Task 1:
```bash
# Verify no old files remain
find . -name "*.pyc" -delete
rm -rf __pycache__
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "docs: complete README rewrite for fraud detection platform

Production-grade ML system documentation with architecture,
model comparison, API docs, and deployment instructions."

git push origin main
```

---

## Execution Dependencies

```
Task 1 (scaffold) → Task 2 (preprocessing) → Task 3 (features)
                                                     ↓
Task 4 (dataset) ─────────────────────────────→ Task 5 (models)
                                                     ↓
                                               Task 6 (evaluation)
                                                     ↓
                                    Task 7 (API) + Task 8 (dashboard)
                                                     ↓
                                    Task 9 (docker) + Task 10 (README)
```

Tasks 7 and 8 can run in parallel. Tasks 9 and 10 can run in parallel.
