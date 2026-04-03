"""Test: Verify Isolation Forest anomaly detector works correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import AnomalyDetector, generate_sample_logs

def main():
    print("=" * 50)
    print("ANOMALY DETECTOR TEST")
    print("=" * 50)

    # Generate training data
    print("\n[1] Generating 5000 training logs...")
    train_df = generate_sample_logs(5000)
    print(f"    Training data: {len(train_df)} rows")

    # Train detector
    print("\n[2] Training Isolation Forest...")
    detector = AnomalyDetector()
    detector.train(train_df)
    print(f"    Trained: {detector.trained}")

    # Generate test data and detect
    print("\n[3] Generating 100 test logs and detecting anomalies...")
    test_df = generate_sample_logs(500, anomaly_pct=0.05, seed=99)
    anomalies = detector.detect(test_df)

    # Results
    print(f"\n[4] Results:")
    print(f"    Total test logs:    {len(test_df)}")
    print(f"    Anomalies found:    {len(anomalies)}")
    print(f"    Anomaly rate:       {len(anomalies)/len(test_df)*100:.1f}%")

    if anomalies:
        avg_score = sum(a.anomaly_score for a in anomalies) / len(anomalies)
        print(f"    Avg anomaly score:  {avg_score:.4f}")
        print(f"\n    First 3 anomalies:")
        for a in anomalies[:3]:
            print(f"      - {a.timestamp} | {a.pod} | score={a.anomaly_score} | "
                  f"err={a.error_rate:.4f} | lat={a.latency_ms:.0f}ms | cpu={a.cpu_pct:.0f}%")

    print("\n" + "=" * 50)
    print("TEST PASSED" if anomalies else "WARNING: No anomalies found")
    print("=" * 50)

if __name__ == "__main__":
    main()
