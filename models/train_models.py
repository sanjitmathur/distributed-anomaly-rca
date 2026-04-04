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
        return _sigmoid(-raw)

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
    """Train all models and return results dict."""
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
