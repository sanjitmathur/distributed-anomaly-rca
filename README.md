# Distributed Anomaly Detection & Root Cause Analysis

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Cost](https://img.shields.io/badge/cost-%240%2Fmo-orange)

AI-powered system that detects anomalies in server/infrastructure logs and diagnoses root causes using multi-model ML and Google Gemini -- entirely free to run.

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
     +--------+---------+  | - (pluggable)  |  |   Flash          |
              |            +-------+--------+  +--------+---------+
              |                    |                     |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          |  FastAPI Service  |  :8000
                          |  (api/service.py) |
                          +------------------+
```

## Features

- **Multi-Model Anomaly Detection** -- Isolation Forest with pluggable model engine
- **Automated Feature Engineering** -- statistical features extracted from raw log metrics
- **Model Evaluation** -- precision, recall, F1, ROC-AUC metrics with structured reports
- **FastAPI REST Service** -- production-ready async API for programmatic access
- **Gemini-Powered Explainability** -- root cause diagnosis in plain English via Gemini 2.0 Flash
- **Real-Time Dashboard** -- interactive Streamlit UI with Plotly visualizations
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

# Run dashboard
streamlit run app.py

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
curl http://localhost:8000/

# Detect anomalies (POST)
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d @your_logs.json

# Root cause analysis
curl -X POST http://localhost:8000/rca \
  -H "Content-Type: application/json" \
  -d '{"anomalies": [...]}'
```

---

## Project Structure

```
distributed-anomaly-rca/
├── api/
│   ├── __init__.py
│   └── service.py              # FastAPI endpoints
├── data_pipeline/
│   ├── __init__.py
│   ├── generator.py            # Sample K8s log generator
│   └── feature_engineering.py  # Statistical feature extraction
├── models/
│   ├── __init__.py
│   └── engine.py               # Isolation Forest detection engine
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # Precision, recall, F1, ROC-AUC
├── .github/
│   └── workflows/              # CI/CD and keep-alive pings
├── app.py                      # Streamlit dashboard entry point
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

## Model Comparison

| Model            | Precision | Recall | F1-Score | ROC-AUC | Latency |
|------------------|-----------|--------|----------|---------|---------|
| Isolation Forest | ~0.90     | ~0.88  | ~0.89    | ~0.92   | <10ms   |
| *(add more)*     |           |        |          |         |         |

---

## API Documentation

| Method | Endpoint     | Description                          |
|--------|------------- |--------------------------------------|
| GET    | `/`          | Health check / service info           |
| POST   | `/detect`    | Run anomaly detection on log payload  |
| POST   | `/rca`       | Root cause analysis on detected anomalies |

### Example Response -- `/detect`

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
| Anomaly Detection  | scikit-learn (Isolation Forest) | $0           |
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
