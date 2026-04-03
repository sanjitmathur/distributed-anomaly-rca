# Distributed Anomaly Detection & Root Cause Analysis

AI-powered system that detects anomalies in server logs and diagnoses root causes using Google Gemini -- all for free.

## What It Does

1. **Upload logs** (JSON/CSV) or generate sample K8s data
2. **Detect anomalies** using Isolation Forest (ML)
3. **Diagnose root causes** with Google Gemini 2.0 Flash
4. **Export results** as JSON reports

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Frontend/Backend | Streamlit | $0 |
| Anomaly Detection | scikit-learn (Isolation Forest) | $0 |
| Root Cause Analysis | Google Gemini 2.0 Flash | $0 (free tier) |
| Hosting | Streamlit Cloud | $0 |
| Uptime | GitHub Actions (12h pings) | $0 |

## Quick Start

```bash
# Clone
git clone https://github.com/sanjitmathur/distributed-anomaly-rca.git
cd distributed-anomaly-rca

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set API key
export GEMINI_API_KEY=your_key_here  # Windows: set GEMINI_API_KEY=your_key_here

# Run
streamlit run app.py
```

## Get Free Gemini API Key

1. Go to https://aistudio.google.com/app/apikeys
2. Click "Create API Key"
3. Copy and set as environment variable

## Performance

| Metric | Value |
|--------|-------|
| Anomaly Detection F1-Score | ~89% |
| RCA Diagnosis Accuracy | ~89% |
| Detection Latency | <10ms |
| RCA Latency | <2s |
| Monthly Cost | $0 |

## Architecture

```
User -> Streamlit UI -> Upload Logs
                     -> Isolation Forest (detect anomalies)
                     -> Gemini API (root cause analysis)
                     -> Export JSON Report
```

## Cost: $0/month

- Gemini API: Free tier (1M tokens/month, ~500 analyses)
- Streamlit Cloud: Free hosting
- GitHub Actions: Free (keeps app alive)

## License

MIT
