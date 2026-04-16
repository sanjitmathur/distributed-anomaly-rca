"""Upgrade 2 — Rich transaction stream simulator.

Generates synthetic transactions with realistic distributions for:
amount, merchant category, location, device ID, IP, card hash, etc.
Injects anomalous behavior patterns periodically.
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
# Reference distributions per V-component
# ---------------------------------------------------------------------------
V_STATS: dict[str, tuple[float, float]] = {f"V{i}": (0.0, 1.0) for i in range(1, 29)}
V_STATS.update({
    "V1": (0.0, 1.95), "V2": (0.0, 1.65), "V3": (0.0, 1.52),
    "V4": (0.0, 1.42), "V14": (0.0, 1.21), "V17": (0.0, 0.73),
})

# ---------------------------------------------------------------------------
# Realistic entity pools
# ---------------------------------------------------------------------------
MERCHANTS = [
    ("MRC-GROCERY-001", "grocery", "New York"),
    ("MRC-GROCERY-002", "grocery", "Chicago"),
    ("MRC-ELECTRONICS-001", "electronics", "San Francisco"),
    ("MRC-ELECTRONICS-002", "electronics", "Austin"),
    ("MRC-RESTAURANT-001", "restaurant", "New York"),
    ("MRC-RESTAURANT-002", "restaurant", "Los Angeles"),
    ("MRC-GAS-001", "gas_station", "Houston"),
    ("MRC-GAS-002", "gas_station", "Miami"),
    ("MRC-ONLINE-001", "online_retail", "Seattle"),
    ("MRC-ONLINE-002", "online_retail", "Denver"),
    ("MRC-TRAVEL-001", "travel", "Atlanta"),
    ("MRC-ATM-001", "atm_withdrawal", "Boston"),
    ("MRC-LUXURY-001", "luxury", "Beverly Hills"),
    ("MRC-PHARMA-001", "pharmacy", "Portland"),
    ("MRC-SUBSCRIPTION-001", "subscription", "San Jose"),
]

MERCHANT_AMOUNTS = {
    "grocery": (25.0, 1.0), "electronics": (150.0, 1.5),
    "restaurant": (40.0, 0.8), "gas_station": (35.0, 0.6),
    "online_retail": (60.0, 1.3), "travel": (300.0, 1.2),
    "atm_withdrawal": (100.0, 0.8), "luxury": (500.0, 1.0),
    "pharmacy": (30.0, 0.7), "subscription": (15.0, 0.5),
}

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "San Francisco",
    "Miami", "Boston", "Seattle", "Denver", "Austin", "Atlanta",
    "Portland", "San Jose", "Beverly Hills",
]

CARD_POOL = [f"CARD-{uuid.uuid4().hex[:8].upper()}" for _ in range(100)]
DEVICE_POOL = [f"DEV-{uuid.uuid4().hex[:8].upper()}" for _ in range(40)]
IP_POOL = [f"192.168.{random.randint(1,10)}.{i}" for i in range(1, 60)]

# Fraud ring entities
FRAUD_CARDS = [f"CARD-FRAUD-{i:03d}" for i in range(5)]
FRAUD_DEVICES = ["DEV-DEADBEEF", "DEV-CAFEBABE", "DEV-B4DB4D00"]
FRAUD_IPS = ["203.0.113.66", "203.0.113.67", "198.51.100.99"]


def _generate_normal() -> dict:
    merchant_id, category, merchant_loc = random.choice(MERCHANTS)
    amt_mean, amt_sigma = MERCHANT_AMOUNTS[category]
    amount = round(abs(np.random.lognormal(mean=np.log(amt_mean), sigma=amt_sigma)), 2)

    # Occasionally use a different location than merchant (travel)
    location = merchant_loc if random.random() > 0.1 else random.choice(LOCATIONS)

    txn = {
        "transaction_id": f"txn-{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "card_id": random.choice(CARD_POOL),
        "merchant_id": merchant_id,
        "merchant_category": category,
        "device_id": random.choice(DEVICE_POOL),
        "source_ip": random.choice(IP_POOL),
        "location": location,
        "Time": float(int(time.time()) % 86400),
        "Amount": amount,
    }
    for col, (mu, sigma) in V_STATS.items():
        txn[col] = round(float(np.random.normal(mu, sigma)), 6)
    return txn


def _generate_fraud() -> dict:
    """Generate anomalous transaction with shifted features + fraud ring entities."""
    txn = _generate_normal()
    txn["transaction_id"] = f"txn-F-{uuid.uuid4().hex[:10]}"

    # Anomaly pattern selection
    pattern = random.choice(["high_value", "velocity_burst", "geo_impossible", "device_takeover"])

    if pattern == "high_value":
        txn["Amount"] = round(abs(np.random.lognormal(mean=7.0, sigma=0.8)), 2)
        txn["merchant_category"] = random.choice(["luxury", "electronics", "online_retail"])

    elif pattern == "velocity_burst":
        # Many small transactions in quick succession (same card)
        txn["card_id"] = random.choice(FRAUD_CARDS)
        txn["Amount"] = round(abs(np.random.normal(10, 5)), 2)

    elif pattern == "geo_impossible":
        # Transaction from unusual location with different device
        txn["location"] = random.choice(["Lagos", "Moscow", "Shanghai", "Sao Paulo"])
        txn["device_id"] = f"DEV-NEW-{uuid.uuid4().hex[:6].upper()}"

    elif pattern == "device_takeover":
        # Known fraud ring devices/IPs
        txn["device_id"] = random.choice(FRAUD_DEVICES)
        txn["source_ip"] = random.choice(FRAUD_IPS)
        txn["card_id"] = random.choice(FRAUD_CARDS)

    # Shift PCA features toward fraud-like distributions
    txn["V1"] = round(float(np.random.normal(-3.0, 2.0)), 6)
    txn["V3"] = round(float(np.random.normal(-4.0, 2.5)), 6)
    txn["V4"] = round(float(np.random.normal(3.5, 1.5)), 6)
    txn["V7"] = round(float(np.random.normal(-5.0, 2.0)), 6)
    txn["V10"] = round(float(np.random.normal(-6.0, 2.0)), 6)
    txn["V14"] = round(float(np.random.normal(-8.0, 3.0)), 6)
    txn["V17"] = round(float(np.random.normal(-4.0, 2.0)), 6)

    return txn


def generate_transaction(fraud_rate: float = 0.005) -> dict:
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
        queue.put(None)

    t = Thread(target=_run, daemon=True)
    return t


# ---------------------------------------------------------------------------
# HTTP-based producer (Docker standalone mode)
# ---------------------------------------------------------------------------

def run_http_producer(target_url: str, tps: float = 10.0, fraud_rate: float = 0.005):
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
        while True:
            txn = generate_transaction(args.fraud_rate)
            print(json.dumps(txn), flush=True)
            time.sleep(1.0 / args.tps)
