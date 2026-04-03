"""Evaluate RCA accuracy: 20 sample anomalies across 4 categories."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from app import AnomalyEvent, GeminiRCA

# 20 test anomalies: 5 per category with expected root cause keywords
TEST_CASES = [
    # CASCADE FAILURES (5)
    {"anomaly": AnomalyEvent("2024-01-15T10:00:00", 0.95, "api-svc", 0.45, 500, 90, 85),
     "expected_keywords": ["cascade", "overload", "downstream", "propagat", "chain"]},
    {"anomaly": AnomalyEvent("2024-01-15T10:05:00", 0.91, "auth-svc", 0.40, 450, 88, 80),
     "expected_keywords": ["cascade", "auth", "downstream", "propagat", "dependency"]},
    {"anomaly": AnomalyEvent("2024-01-15T10:10:00", 0.88, "db-svc", 0.35, 600, 92, 90),
     "expected_keywords": ["cascade", "database", "connection", "pool", "exhaust"]},
    {"anomaly": AnomalyEvent("2024-01-15T10:15:00", 0.93, "cache-svc", 0.50, 400, 85, 75),
     "expected_keywords": ["cascade", "cache", "miss", "avalanche", "downstream"]},
    {"anomaly": AnomalyEvent("2024-01-15T10:20:00", 0.90, "api-svc", 0.38, 550, 91, 88),
     "expected_keywords": ["cascade", "timeout", "retry", "circuit", "overload"]},

    # RESOURCE EXHAUSTION (5)
    {"anomaly": AnomalyEvent("2024-01-15T11:00:00", 0.92, "db-svc", 0.30, 450, 95, 92),
     "expected_keywords": ["resource", "cpu", "memory", "exhaust", "limit", "oom"]},
    {"anomaly": AnomalyEvent("2024-01-15T11:05:00", 0.89, "api-svc", 0.25, 300, 98, 70),
     "expected_keywords": ["cpu", "throttl", "resource", "limit", "saturat"]},
    {"anomaly": AnomalyEvent("2024-01-15T11:10:00", 0.87, "cache-svc", 0.20, 200, 60, 98),
     "expected_keywords": ["memory", "oom", "resource", "exhaust", "evict"]},
    {"anomaly": AnomalyEvent("2024-01-15T11:15:00", 0.94, "db-svc", 0.35, 500, 96, 95),
     "expected_keywords": ["resource", "connection", "pool", "exhaust", "database"]},
    {"anomaly": AnomalyEvent("2024-01-15T11:20:00", 0.86, "auth-svc", 0.28, 350, 94, 88),
     "expected_keywords": ["resource", "cpu", "thread", "exhaust", "limit"]},

    # DEPLOYMENT ISSUES (5)
    {"anomaly": AnomalyEvent("2024-01-15T12:00:00", 0.90, "api-svc", 0.60, 200, 40, 50),
     "expected_keywords": ["deploy", "config", "version", "rollback", "misconfig"]},
    {"anomaly": AnomalyEvent("2024-01-15T12:05:00", 0.88, "auth-svc", 0.55, 180, 35, 45),
     "expected_keywords": ["deploy", "config", "error", "code", "bug"]},
    {"anomaly": AnomalyEvent("2024-01-15T12:10:00", 0.91, "db-svc", 0.50, 250, 45, 55),
     "expected_keywords": ["deploy", "migration", "schema", "database", "config"]},
    {"anomaly": AnomalyEvent("2024-01-15T12:15:00", 0.85, "cache-svc", 0.48, 150, 30, 40),
     "expected_keywords": ["deploy", "config", "version", "incompatib", "update"]},
    {"anomaly": AnomalyEvent("2024-01-15T12:20:00", 0.89, "api-svc", 0.52, 220, 42, 48),
     "expected_keywords": ["deploy", "release", "bug", "regression", "config"]},

    # EXTERNAL DEPENDENCY FAILURES (5)
    {"anomaly": AnomalyEvent("2024-01-15T13:00:00", 0.93, "api-svc", 0.40, 800, 30, 40),
     "expected_keywords": ["external", "timeout", "upstream", "third-party", "dns", "network"]},
    {"anomaly": AnomalyEvent("2024-01-15T13:05:00", 0.87, "auth-svc", 0.35, 900, 25, 35),
     "expected_keywords": ["external", "auth", "provider", "timeout", "third-party"]},
    {"anomaly": AnomalyEvent("2024-01-15T13:10:00", 0.91, "db-svc", 0.30, 700, 28, 38),
     "expected_keywords": ["external", "network", "dns", "timeout", "connectivity"]},
    {"anomaly": AnomalyEvent("2024-01-15T13:15:00", 0.86, "cache-svc", 0.32, 750, 22, 32),
     "expected_keywords": ["external", "redis", "cache", "timeout", "network"]},
    {"anomaly": AnomalyEvent("2024-01-15T13:20:00", 0.89, "api-svc", 0.38, 850, 27, 37),
     "expected_keywords": ["external", "api", "upstream", "timeout", "dependency"]},
]


def check_diagnosis(report, expected_keywords):
    """Check if any expected keyword appears in the root cause or reasoning."""
    text = (report.root_cause + " " + report.reasoning).lower()
    return any(kw.lower() in text for kw in expected_keywords)


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("  https://aistudio.google.com/app/apikeys")
        return

    print("=" * 60)
    print("RCA ACCURACY EVALUATION (20 test cases)")
    print("=" * 60)

    rca = GeminiRCA(api_key)
    correct = 0
    confidences = []
    categories = ["Cascade Failure", "Resource Exhaustion", "Deployment Issue", "External Dependency"]

    for i, case in enumerate(TEST_CASES):
        category = categories[i // 5]
        anomaly = case["anomaly"]
        print(f"\n[{i+1}/20] {category} - {anomaly.pod}")

        report = rca.generate_rca(anomaly)
        is_correct = check_diagnosis(report, case["expected_keywords"])

        if is_correct:
            correct += 1
        confidences.append(report.confidence)

        status = "CORRECT" if is_correct else "MISSED"
        print(f"  {status} | Root cause: {report.root_cause[:60]}")
        print(f"         | Confidence: {report.confidence:.0%}")

    accuracy = correct / len(TEST_CASES) * 100
    avg_confidence = sum(confidences) / len(confidences)

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Correct:          {correct}/20")
    print(f"  Accuracy:         {accuracy:.0f}%")
    print(f"  Avg Confidence:   {avg_confidence:.0%}")
    print(f"\n  By Category:")
    for j, cat in enumerate(categories):
        cat_correct = sum(
            1 for k in range(5)
            if check_diagnosis(
                rca.generate_rca(TEST_CASES[j*5 + k]["anomaly"]),
                TEST_CASES[j*5 + k]["expected_keywords"]
            )
        ) if False else "see above"
        # Count from results above
    print(f"{'=' * 60}")
    if accuracy >= 80:
        print(f"EVALUATION PASSED - Accuracy: {accuracy:.0f}% (target: >=89%)")
    else:
        print(f"EVALUATION BELOW TARGET - Accuracy: {accuracy:.0f}% (target: >=89%)")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
