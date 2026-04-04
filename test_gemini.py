"""Test: Verify Gemini API integration works."""

import os
import sys

# Suppress Streamlit warnings when importing from app.py
os.environ.setdefault("STREAMLIT_RUNTIME", "")
sys.path.insert(0, os.path.dirname(__file__))

from app import AnomalyEvent, GeminiRCA

def main():
    print("=" * 50)
    print("GEMINI RCA TEST")
    print("=" * 50)

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY not set!")
        print("\nTo fix:")
        print("  1. Go to: https://aistudio.google.com/app/apikeys")
        print("  2. Create a free API key")
        print("  3. Set it:")
        print("     Windows: set GEMINI_API_KEY=your_key_here")
        print("     Linux:   export GEMINI_API_KEY=your_key_here")
        return

    print(f"\n[1] API key found: {api_key[:10]}...")

    # Create sample anomaly
    sample = AnomalyEvent(
        timestamp="2024-01-15T14:30:00",
        anomaly_score=0.92,
        pod="db-svc",
        error_rate=0.35,
        latency_ms=450.0,
        cpu_pct=85.0,
        memory_pct=72.0,
    )
    print(f"\n[2] Sample anomaly: {sample.pod} | score={sample.anomaly_score}")

    # Call Gemini
    print("\n[3] Calling Gemini 2.0 Flash...")
    rca = GeminiRCA(api_key)
    report = rca.generate_rca(sample)

    print(f"\n[4] RCA Report:")
    print(f"    Root Cause:        {report.root_cause}")
    print(f"    Confidence:        {report.confidence:.0%}")
    print(f"    Reasoning:         {report.reasoning[:100]}...")
    print(f"    Affected Services: {report.affected_services}")
    print(f"    Remediation:       {report.remediation}")

    # Token estimate
    prompt_tokens = 200  # approximate
    response_tokens = 100  # approximate
    print(f"\n[5] Estimated token usage:")
    print(f"    Prompt:   ~{prompt_tokens} tokens")
    print(f"    Response: ~{response_tokens} tokens")
    print(f"    Total:    ~{prompt_tokens + response_tokens} tokens")
    print(f"    Budget:   1M tokens/month = ~{1_000_000 // (prompt_tokens + response_tokens)} analyses")

    print("\n" + "=" * 50)
    success = report.root_cause != "Unable to analyze"
    print("TEST PASSED" if success else "TEST FAILED - check API key")
    print("=" * 50)

if __name__ == "__main__":
    main()
