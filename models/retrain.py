"""Upgrade 5 — Model retraining pipeline with feedback incorporation and versioning."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation.metrics import compute_metrics
from models.train_models import _TRAINERS, _score_model, _save_model
from monitoring.feedback import FeedbackStore
from pipeline.feature_engineering import FeatureEngineer
from pipeline.preprocessing import load_data, preprocess
from utils.config import ALL_FEATURES, MODELS_DIR, PROJECT_ROOT
from utils.logger import get_logger

log = get_logger("retrain")

VERSION_FILE = MODELS_DIR / "version_registry.json"


def _load_version_registry() -> dict:
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text())
    return {"current_version": "v1.0", "history": []}


def _save_version_registry(registry: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(registry, indent=2))


def get_current_version() -> str:
    return _load_version_registry()["current_version"]


def retrain_model(
    model_name: str = "isolation_forest",
    include_feedback: bool = True,
) -> dict:
    """Full retrain pipeline: load data + feedback -> train -> evaluate -> version -> save.

    Returns dict with version, metrics, and status.
    """
    t0 = time.time()
    log.info("Starting retrain of %s (include_feedback=%s)", model_name, include_feedback)

    # 1. Load base dataset
    df = load_data()
    fe = FeatureEngineer()
    df_eng = fe.transform(df.drop(columns=["Class"]))
    df_eng["Class"] = df["Class"].values

    # 2. Incorporate analyst feedback
    feedback_count = 0
    if include_feedback:
        store = FeedbackStore()
        feedback_data, feedback_labels = store.get_labeled_data()
        if feedback_data:
            feedback_df = pd.DataFrame(feedback_data)
            # Ensure feedback has required columns
            for col in df_eng.columns:
                if col not in feedback_df.columns:
                    feedback_df[col] = 0.0
            feedback_df = feedback_df[df_eng.columns]
            feedback_df["Class"] = feedback_labels
            df_eng = pd.concat([df_eng, feedback_df], ignore_index=True)
            feedback_count = len(feedback_data)
            log.info("Incorporated %d feedback samples", feedback_count)

    # 3. Preprocess
    result = preprocess(df_eng)
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_test = result["y_test"]

    # 4. Train
    if model_name not in _TRAINERS:
        return {"status": "error", "message": f"Unknown model: {model_name}"}

    trainer = _TRAINERS[model_name]
    train_result = trainer(X_train)
    train_time = time.time() - t0

    # 5. Evaluate
    scores = _score_model(model_name, train_result, X_test)
    metrics = compute_metrics(y_test, scores)
    train_result["scores"] = scores
    train_result["train_time"] = train_time

    # 6. Version
    registry = _load_version_registry()
    version_num = len(registry["history"]) + 1
    new_version = f"v{version_num}.0"

    # 7. Save
    _save_model(model_name, train_result)

    # 8. Update registry
    version_entry = {
        "version": new_version,
        "model": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_time_seconds": round(train_time, 2),
        "feedback_samples": feedback_count,
        "train_samples": len(X_train),
        "metrics": {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
        },
    }
    registry["current_version"] = new_version
    registry["history"].append(version_entry)
    _save_version_registry(registry)

    log.info(
        "Retrain complete: %s %s (F1=%.4f, AUC=%.4f) in %.1fs",
        model_name, new_version, metrics["f1"], metrics["roc_auc"], train_time,
    )

    return {
        "status": "success",
        "version": new_version,
        "model": model_name,
        "train_time": round(train_time, 2),
        "feedback_incorporated": feedback_count,
        "metrics": version_entry["metrics"],
    }


def get_version_history() -> list[dict]:
    return _load_version_registry()["history"]
