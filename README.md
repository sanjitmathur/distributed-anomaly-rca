# Distributed Anomaly Detection & Root Cause Analysis

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Cost](https://img.shields.io/badge/cost-%240%2Fmo-orange)

AI-powered system that detects anomalies in server/infrastructure logs and diagnoses root causes using multi-model ML and Google Gemini -- entirely free to run.

**[Live Demo](https://anomaly-detection-analysis.streamlit.app/)** -- https://anomaly-detection-analysis.streamlit.app/

---

## Architecture

```
                          +------------------+
                          |   Streamlit UI   |  :8501
                          |   (dashboard)    |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
     +--------v--------+  +-------v--------+  +--------v---------+
     | Data Pipeline    |  | Detection      |  | Root Cause       |
     | - generator.py   |  | Engine         |  | Analysis         |
     | - feature_eng.py |  | - Iso. Forest  |  | - Gemini 2.0     |
     +--------+---------+  | - LOF          |  |   Flash          |
              |            | - DBSCAN       |  +--------+---------+
              |            | - Autoencoder  |           |
              |            +-------+--------+           |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          |  FastAPI Service  |  :8000
                          |  (api/service.py) |
                          +------------------+
```

## Features

- **Multi-Model Anomaly Detection** -- Isolation Forest, LOF, DBSCAN, Autoencoder with ensemble scoring
- **Automated Feature Engineering** -- rolling stats, z-scores, and time-based features from raw log metrics
- **Model Evaluation Framework** -- precision, recall, F1, ROC-AUC with model leaderboard
- **FastAPI REST Service** -- production-ready async API for programmatic access
- **Gemini-Powered Explainability** -- root cause diagnosis in plain English via Gemini 2.0 Flash
- **Advanced Dashboard** -- 6-tab Streamlit UI (explorer, detection, comparison, threshold tuning, explainability, real-time streaming)
- **Permutation-Based Feature Importance** -- understand which metrics drive anomaly scores
- **Docker-Ready** -- single-command deployment with docker-compose

---

## Quick Start

### Local

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

# Run dashboard (simple 3-tab version)
streamlit run app.py

# Run advanced dashboard (6-tab version)
streamlit run dashboard/app.py

# Run API (separate terminal)
uvicorn api.service:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Start both services
docker-compose up --build

# Dashboard: http://localhost:8501
# API:       http://localhost:8000
```

### API

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Detect anomalies
curl -X POST http://localhost:8000/detect_anomaly \
  -H "Content-Type: application/json" \
  -d '{"data": [{"error_rate": 0.35, "latency_ms": 450, "cpu_pct": 85, "memory_pct": 72}]}'
```

---

## Project Structure

```
distributed-anomaly-rca/
├── api/
│   ├── __init__.py
│   └── service.py              # FastAPI endpoints (/health, /models, /detect_anomaly)
├── data_pipeline/
│   ├── __init__.py
│   ├── generator.py            # Sample K8s log generator
│   └── feature_engineering.py  # Statistical feature extraction
├── models/
│   ├── __init__.py
│   └── engine.py               # Multi-model detection engine
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # Precision, recall, F1, ROC-AUC
├── dashboard/
│   └── app.py                  # Advanced 6-tab Streamlit dashboard
├── .github/
│   └── workflows/
│       └── keep-alive.yml      # Pings app every 12h to prevent sleep
├── app.py                      # Simple 3-tab Streamlit dashboard
├── evaluate_detector.py        # Detector evaluation script
├── evaluate_rca.py             # RCA evaluation script
├── test_detector.py            # Detection tests
├── test_gemini.py              # Gemini integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Model Performance

| Model            | Precision | Recall  | F1-Score | Latency |
|------------------|-----------|---------|----------|---------|
| Isolation Forest | 98.08%    | 100.00% | 99.03%   | <10ms   |

*Evaluated on 5,000 test samples with 5% anomaly rate.*

---

## API Documentation

| Method | Endpoint          | Description                              |
|--------|-------------------|------------------------------------------|
| GET    | `/health`         | Health check / service info              |
| GET    | `/models`         | List available detection models          |
| POST   | `/detect_anomaly` | Run anomaly detection on log payload     |

### Example Response -- `/detect_anomaly`

```json
{
  "anomalies": [...],
  "total_records": 1000,
  "anomalies_found": 42,
  "model": "IsolationForest"
}
```

---

## Deployment

### Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as entry point
4. Add `GEMINI_API_KEY` in Secrets
5. Deploy

### Docker (Production)

```bash
docker-compose up -d --build
```

Both the dashboard (`:8501`) and API (`:8000`) start automatically. Use a reverse proxy (nginx/Caddy) for HTTPS in production.

---

## Get Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikeys)
2. Click "Create API Key"
3. Copy and set as `GEMINI_API_KEY` environment variable

---

## Tech Stack

| Component          | Technology                    | Cost           |
|--------------------|-------------------------------|----------------|
| Dashboard          | Streamlit + Plotly            | $0             |
| API                | FastAPI + Uvicorn             | $0             |
| Anomaly Detection  | scikit-learn (multi-model)    | $0             |
| Root Cause Analysis| Google Gemini 2.0 Flash       | $0 (free tier) |
| Hosting            | Streamlit Cloud / Docker      | $0             |
| CI/CD              | GitHub Actions                | $0             |
| **Total**          |                               | **$0/month**   |

---

## Cost: $0/month

- **Gemini API**: Free tier -- 1M tokens/month (~500 analyses)
- **Streamlit Cloud**: Free hosting for public repos
- **GitHub Actions**: Free for public repos (keeps app alive with 12h pings)

---

## License

MIT
