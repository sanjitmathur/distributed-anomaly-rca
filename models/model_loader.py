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
    """Load a trained model and its metadata."""
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
