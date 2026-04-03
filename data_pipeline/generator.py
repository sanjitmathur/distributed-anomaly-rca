"""
Synthetic log generator for anomaly detection training and evaluation.

Produces realistic infrastructure metrics with configurable anomaly injection
across four distinct failure modes.
"""

import numpy as np
import pandas as pd


# Anomaly type definitions with distinct metric signatures
ANOMALY_SIGNATURES = {
    "cascade_failure": {
        "error_rate": (0.15, 0.05),      # high error rate
        "latency_ms": (500, 150),         # severe latency spike
        "cpu_pct": (85, 10),              # elevated CPU
        "memory_pct": (70, 10),           # elevated memory
    },
    "resource_exhaustion": {
        "error_rate": (0.05, 0.02),       # moderate errors
        "latency_ms": (200, 50),          # moderate latency
        "cpu_pct": (95, 3),               # near-max CPU
        "memory_pct": (95, 3),            # near-max memory
    },
    "deployment_issue": {
        "error_rate": (0.08, 0.03),       # elevated errors
        "latency_ms": (300, 100),         # high variance latency
        "cpu_pct": (60, 20),              # unstable CPU
        "memory_pct": (50, 15),           # normal-ish memory
    },
    "external_dependency": {
        "error_rate": (0.03, 0.01),       # low-moderate errors
        "latency_ms": (1000, 300),        # extreme latency (upstream timeout)
        "cpu_pct": (25, 5),               # low CPU (waiting on I/O)
        "memory_pct": (40, 10),           # normal memory
    },
}

POD_NAMES = [
    "api-gateway-7b9f4",
    "auth-service-3c8d1",
    "payment-svc-5a2e9",
    "user-svc-1d6f3",
    "order-svc-8e4b2",
]


def generate_sample_logs(
    n: int = 500,
    anomaly_pct: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic infrastructure log data with injected anomalies.

    Parameters
    ----------
    n : int
        Total number of log entries to generate.
    anomaly_pct : float
        Fraction of entries that are anomalous (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, pod, error_rate, latency_ms, cpu_pct,
        memory_pct, is_anomaly, anomaly_type
    """
    rng = np.random.default_rng(seed)
    n_anomalies = int(n * anomaly_pct)
    n_normal = n - n_anomalies

    # Generate timestamps spanning ~7 days at irregular intervals
    base_time = pd.Timestamp("2025-01-01")
    offsets = np.sort(rng.uniform(0, 7 * 24 * 3600, size=n))
    timestamps = [base_time + pd.Timedelta(seconds=float(s)) for s in offsets]

    # --- Normal samples ---
    normal_data = {
        "error_rate": np.clip(rng.normal(0.001, 0.0005, n_normal), 0, None),
        "latency_ms": np.clip(rng.normal(50, 10, n_normal), 1, None),
        "cpu_pct": np.clip(rng.normal(30, 10, n_normal), 0, 100),
        "memory_pct": np.clip(rng.uniform(20, 80, n_normal), 0, 100),
    }
    normal_labels = np.zeros(n_normal, dtype=bool)
    normal_types = ["normal"] * n_normal

    # --- Anomaly samples ---
    anomaly_keys = list(ANOMALY_SIGNATURES.keys())
    chosen_types = rng.choice(anomaly_keys, size=n_anomalies)

    anom_error, anom_latency, anom_cpu, anom_mem = [], [], [], []
    anom_type_labels = []

    for atype in chosen_types:
        sig = ANOMALY_SIGNATURES[atype]
        anom_error.append(np.clip(rng.normal(*sig["error_rate"]), 0, None))
        anom_latency.append(np.clip(rng.normal(*sig["latency_ms"]), 1, None))
        anom_cpu.append(np.clip(rng.normal(*sig["cpu_pct"]), 0, 100))
        anom_mem.append(np.clip(rng.normal(*sig["memory_pct"]), 0, 100))
        anom_type_labels.append(atype)

    anomaly_data = {
        "error_rate": np.array(anom_error),
        "latency_ms": np.array(anom_latency),
        "cpu_pct": np.array(anom_cpu),
        "memory_pct": np.array(anom_mem),
    }
    anomaly_labels = np.ones(n_anomalies, dtype=bool)

    # --- Combine and shuffle ---
    all_error = np.concatenate([normal_data["error_rate"], anomaly_data["error_rate"]])
    all_latency = np.concatenate([normal_data["latency_ms"], anomaly_data["latency_ms"]])
    all_cpu = np.concatenate([normal_data["cpu_pct"], anomaly_data["cpu_pct"]])
    all_mem = np.concatenate([normal_data["memory_pct"], anomaly_data["memory_pct"]])
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    all_types = normal_types + anom_type_labels

    # Shuffle while keeping alignment
    shuffle_idx = rng.permutation(n)

    df = pd.DataFrame({
        "timestamp": [timestamps[i] for i in shuffle_idx],
        "pod": [POD_NAMES[i % len(POD_NAMES)] for i in shuffle_idx],
        "error_rate": all_error[shuffle_idx],
        "latency_ms": all_latency[shuffle_idx],
        "cpu_pct": all_cpu[shuffle_idx],
        "memory_pct": all_mem[shuffle_idx],
        "is_anomaly": all_labels[shuffle_idx],
        "anomaly_type": [all_types[i] for i in shuffle_idx],
    })

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
