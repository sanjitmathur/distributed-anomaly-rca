"""Mock transaction producer — streams synthetic data into a shared queue.

Runs as a standalone service. In Docker, this is the `producer` container.
Locally, it can feed the inference service via a multiprocessing queue or
HTTP POST.

Usage:
    # Standalone (posts to inference service)
    python -m streaming.producer --mode http --target http://localhost:8000/ingest
    # In-process (returns queue)
    from streaming.producer import create_producer_thread
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

log = get_logger("producer")

# ---------------------------------------------------------------------------
# Reference distributions (μ, σ) per V-component from the training set
# Approximated from creditcard.csv summary statistics
# ---------------------------------------------------------------------------
V_STATS: dict[str, tuple[float, float]] = {
    f"V{i}": (0.0, 1.0) for i in range(1, 29)
}
# Overrides for the most discriminative features (from EDA)
V_STATS["V1"] = (0.0, 1.95)
V_STATS["V2"] = (0.0, 1.65)
V_STATS["V3"] = (0.0, 1.52)
V_STATS["V4"] = (0.0, 1.42)
V_STATS["V14"] = (0.0, 1.21)
V_STATS["V17"] = (0.0, 0.73)

# Pool of fake IPs / device IDs for graph feature realism
IP_POOL = [f"192.168.1.{i}" for i in range(1, 51)] + [
    f"10.0.{random.randint(0,5)}.{i}" for i in range(1, 21)
]
DEVICE_POOL = [f"DEV-{uuid.uuid4().hex[:8].upper()}" for _ in range(30)]

# Fraud ring: a small cluster shares IPs/devices → high graph degree
FRAUD_IPS = ["203.0.113.66", "203.0.113.67"]
FRAUD_DEVICES = ["DEV-DEADBEEF", "DEV-CAFEBABE"]


def _generate_normal() -> dict:
    txn = {
        "transaction_id": f"txn-{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": random.choice(IP_POOL),
        "device_id": random.choice(DEVICE_POOL),
        "Time": float(int(time.time()) % 86400),
        "Amount": round(abs(np.random.lognormal(mean=3.5, sigma=1.8)), 2),
    }
    for col, (mu, sigma) in V_STATS.items():
        txn[col] = round(float(np.random.normal(mu, sigma)), 6)
    return txn


def _generate_fraud() -> dict:
    txn = _generate_normal()
    txn["transaction_id"] = f"txn-F-{uuid.uuid4().hex[:10]}"
    # Shift key features to fraud-like ranges
    txn["V1"] = round(float(np.random.normal(-3.0, 2.0)), 6)
    txn["V3"] = round(float(np.random.normal(-4.0, 2.5)), 6)
    txn["V4"] = round(float(np.random.normal(3.5, 1.5)), 6)
    txn["V7"] = round(float(np.random.normal(-5.0, 2.0)), 6)
    txn["V10"] = round(float(np.random.normal(-6.0, 2.0)), 6)
    txn["V14"] = round(float(np.random.normal(-8.0, 3.0)), 6)
    txn["V17"] = round(float(np.random.normal(-4.0, 2.0)), 6)
    txn["Amount"] = round(abs(np.random.lognormal(mean=5.5, sigma=1.2)), 2)
    # Fraud ring — shared infrastructure
    txn["source_ip"] = random.choice(FRAUD_IPS)
    txn["device_id"] = random.choice(FRAUD_DEVICES)
    return txn


def generate_transaction(fraud_rate: float = 0.005) -> dict:
    """Generate a single synthetic transaction."""
    if random.random() < fraud_rate:
        return _generate_fraud()
    return _generate_normal()


# ---------------------------------------------------------------------------
# Queue-based producer (in-process)
# ---------------------------------------------------------------------------

def create_producer_thread(
    queue: Queue,
    tps: float = 10.0,
    fraud_rate: float = 0.005,
    max_txns: Optional[int] = None,
) -> Thread:
    """Spawn a daemon thread that pushes transactions into `queue`."""

    def _run():
        count = 0
        interval = 1.0 / tps
        log.info("Producer started: %.1f TPS, %.1f%% fraud rate", tps, fraud_rate * 100)
        while max_txns is None or count < max_txns:
            txn = generate_transaction(fraud_rate)
            queue.put(txn)
            count += 1
            if count % 100 == 0:
                log.info("Produced %d transactions", count)
            time.sleep(interval)
        log.info("Producer finished: %d transactions", count)
        queue.put(None)  # sentinel

    t = Thread(target=_run, daemon=True)
    return t


# ---------------------------------------------------------------------------
# HTTP-based producer (Docker standalone mode)
# ---------------------------------------------------------------------------

def run_http_producer(target_url: str, tps: float = 10.0, fraud_rate: float = 0.005):
    """POST transactions to the inference service endpoint."""
    import httpx

    client = httpx.Client(timeout=5.0)
    interval = 1.0 / tps
    count = 0
    log.info("HTTP Producer -> %s @ %.1f TPS", target_url, tps)

    while True:
        txn = generate_transaction(fraud_rate)
        try:
            resp = client.post(target_url, json=txn)
            if resp.status_code != 200:
                log.warning("Ingest returned %d: %s", resp.status_code, resp.text[:200])
        except httpx.RequestError as e:
            log.warning("Connection error: %s — retrying in 2s", e)
            time.sleep(2)
            continue
        count += 1
        if count % 100 == 0:
            log.info("Sent %d transactions", count)
        time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock transaction producer")
    parser.add_argument("--mode", choices=["http", "stdout"], default="http")
    parser.add_argument("--target", default="http://inference_service:8000/ingest")
    parser.add_argument("--tps", type=float, default=10.0)
    parser.add_argument("--fraud-rate", type=float, default=0.005)
    args = parser.parse_args()

    if args.mode == "http":
        run_http_producer(args.target, args.tps, args.fraud_rate)
    else:
        # stdout mode for debugging / piping
        while True:
            txn = generate_transaction(args.fraud_rate)
            print(json.dumps(txn), flush=True)
            time.sleep(1.0 / args.tps)
