"""Analyst feedback storage — captures fraud/legit decisions for retraining."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from utils.config import PROJECT_ROOT
from utils.logger import get_logger

log = get_logger("feedback")

FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "analyst_decisions.jsonl"


class FeedbackStore:
    """Append-only JSONL store for analyst fraud/legit decisions."""

    def __init__(self):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(
        self,
        transaction_id: str,
        analyst_decision: str,  # "fraud" or "legitimate"
        fraud_score: float,
        transaction_data: dict,
        analyst_notes: str = "",
    ) -> dict:
        entry = {
            "transaction_id": transaction_id,
            "decision": analyst_decision,
            "label": 1 if analyst_decision == "fraud" else 0,
            "fraud_score": fraud_score,
            "analyst_notes": analyst_notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "transaction_data": transaction_data,
        }
        with self._lock:
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        log.info("Feedback recorded: %s -> %s (score=%.3f)", transaction_id, analyst_decision, fraud_score)
        return entry

    def load_all(self) -> list[dict]:
        if not FEEDBACK_FILE.exists():
            return []
        with self._lock:
            entries = []
            with open(FEEDBACK_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def get_labeled_data(self) -> tuple[list[dict], list[int]]:
        """Return (transaction_data_list, labels) for retraining."""
        entries = self.load_all()
        data = [e["transaction_data"] for e in entries]
        labels = [e["label"] for e in entries]
        return data, labels

    @property
    def stats(self) -> dict:
        entries = self.load_all()
        fraud_count = sum(1 for e in entries if e["label"] == 1)
        return {
            "total_reviews": len(entries),
            "marked_fraud": fraud_count,
            "marked_legitimate": len(entries) - fraud_count,
        }
