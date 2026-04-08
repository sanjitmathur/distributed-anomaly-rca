"""Streaming feature store: sliding window aggregations + graph-based features.

Maintains in-memory state for:
1. Sliding Window: per-device/IP transaction frequency and amount stats
   over a configurable window (default 10 min).
2. Graph Features: bipartite graph of (source_ip ↔ device_id) tracking
   shared-infrastructure degree — fraud rings share IPs/devices.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class _WindowEntry:
    timestamp: float
    amount: float


class SlidingWindowAggregator:
    """Per-entity sliding window over the last `window_sec` seconds.

    Tracks:
    - txn_count:  number of transactions in the window
    - txn_amount_mean: mean amount in the window
    - txn_amount_std:  std-dev of amounts in the window
    - txn_velocity: txns per minute
    """

    def __init__(self, window_sec: float = 600.0):
        self._window_sec = window_sec
        self._buffers: dict[str, deque[_WindowEntry]] = defaultdict(deque)
        self._lock = Lock()

    def push(self, entity_id: str, amount: float, ts: Optional[float] = None) -> dict:
        if ts is None:
            ts = time.time()
        entry = _WindowEntry(timestamp=ts, amount=amount)

        with self._lock:
            buf = self._buffers[entity_id]
            buf.append(entry)
            # evict stale entries
            cutoff = ts - self._window_sec
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()

            amounts = [e.amount for e in buf]
            count = len(amounts)
            mean_amt = sum(amounts) / count if count else 0.0
            std_amt = (
                (sum((a - mean_amt) ** 2 for a in amounts) / count) ** 0.5
                if count > 1 else 0.0
            )
            elapsed = max(ts - buf[0].timestamp, 1.0) if count else 1.0
            velocity = count / (elapsed / 60.0)

        return {
            "txn_count_10m": count,
            "txn_amount_mean_10m": round(mean_amt, 4),
            "txn_amount_std_10m": round(std_amt, 4),
            "txn_velocity_per_min": round(velocity, 4),
        }


class GraphFeatureExtractor:
    """Bipartite graph: source_ip ↔ device_id.

    Tracks:
    - ip_degree:     number of distinct devices seen from this IP
    - device_degree: number of distinct IPs seen on this device
    - shared_infra_score: ip_degree + device_degree (higher → more suspicious)
    """

    def __init__(self):
        self._ip_to_devices: dict[str, set[str]] = defaultdict(set)
        self._device_to_ips: dict[str, set[str]] = defaultdict(set)
        self._lock = Lock()

    def update(self, source_ip: str, device_id: str) -> dict:
        with self._lock:
            self._ip_to_devices[source_ip].add(device_id)
            self._device_to_ips[device_id].add(source_ip)

            ip_deg = len(self._ip_to_devices[source_ip])
            dev_deg = len(self._device_to_ips[device_id])

        return {
            "ip_degree": ip_deg,
            "device_degree": dev_deg,
            "shared_infra_score": ip_deg + dev_deg,
        }
