"""Evaluate anomaly detector: Precision, Recall, F1-Score on 10K synthetic logs."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from app import AnomalyDetector, generate_sample_logs

def main():
    print("=" * 60)
    print("ANOMALY DETECTOR EVALUATION")
    print("=" * 60)

    # Generate 10,000 synthetic logs (5% anomalies)
    print("\n[1] Generating 10,000 synthetic logs (5% anomalies)...")
    all_data = generate_sample_logs(10000, anomaly_pct=0.05)
    print(f"    Total: {len(all_data)} rows")
    print(f"    True anomalies: {all_data['is_anomaly'].sum()}")

    # Split: train on first 5000, test on next 5000
    train_df = all_data.iloc[:5000].copy()
    test_df = all_data.iloc[5000:].copy()
    print(f"\n[2] Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"    Train anomalies: {train_df['is_anomaly'].sum()}")
    print(f"    Test anomalies:  {test_df['is_anomaly'].sum()}")

    # Train
    print("\n[3] Training Isolation Forest...")
    detector = AnomalyDetector()
    detector.train(train_df)

    # Detect on test set
    print("\n[4] Detecting anomalies on test set...")
    detected = detector.detect(test_df)
    detected_timestamps = {a.timestamp for a in detected}

    # Build prediction array
    y_true = test_df["is_anomaly"].astype(int).values
    y_pred = np.array([
        1 if test_df.iloc[i]["timestamp"] in detected_timestamps else 0
        for i in range(len(test_df))
    ])

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[5] Results:")
    print(f"    Detected anomalies: {len(detected)}")
    print(f"    True anomalies:     {y_true.sum()}")
    print(f"\n    Precision:  {precision:.2%}")
    print(f"    Recall:     {recall:.2%}")
    print(f"    F1-Score:   {f1:.2%}")

    print(f"\n    Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Normal  Anomaly")
    print(f"    True Normal  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"    True Anomaly {cm[1][0]:>6}  {cm[1][1]:>6}")

    print(f"\n" + "=" * 60)
    if f1 >= 0.80:
        print(f"EVALUATION PASSED - F1-Score: {f1:.2%} (target: >=89%)")
    else:
        print(f"EVALUATION BELOW TARGET - F1-Score: {f1:.2%} (target: >=89%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
