# Financial Transaction Fraud Detection Platform

![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

Production-grade ML platform that detects fraudulent financial transactions using 4 anomaly detection models with SHAP explainability, a FastAPI backend, and a 6-tab Streamlit dashboard.

**[Live Demo](https://anomaly-detection-analysis.streamlit.app/)** · https://anomaly-detection-analysis.streamlit.app/

---

## Architecture

```
CSV Dataset (10K transactions)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Preprocessing   │────▶│   Feature Eng.   │────▶│  Model Training │
│  (split, scale)  │     │  (6 new features)│     │  (4 models)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                   Model Registry    FastAPI :8000   Streamlit :8501
                                   (.joblib files)   /predict        6-tab dashboard
                                                     /batch_predict
                                                     /model_metrics
```

## Features

- **4 Anomaly Detection Models**: Isolation Forest, Local Outlier Factor, One-Class SVM, Autoencoder (scikit-learn + optional PyTorch)
- **Feature Engineering**: Log-scaled amounts, cyclical time encoding, Z-scores, PCA magnitude, outlier counts
- **SHAP Explainability**: Tree and kernel SHAP explanations for individual fraud predictions
- **Real-Time Simulation**: Live transaction monitoring with animated scoring
- **Model Comparison**: Leaderboard, ROC/PR curves, metric bar charts
- **FastAPI Backend**: REST API with Pydantic validation, batch prediction, CORS support
- **Interactive Dashboard**: 6 tabs — Explorer, Detection, Comparison, Visualization, Explainability, Simulation

## Dataset

Uses a **10,000-row stratified sample** from the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle). Features V1-V28 are PCA-transformed, plus Time and Amount. Class label: 0 = normal, 1 = fraud (~0.17% fraud rate).

## Quick Start

### Local

```bash
# Clone and setup
git clone https://github.com/sanjit-mathur/Anomoly-Detection.git
cd Anomoly-Detection
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the dashboard (trains models on first launch)
streamlit run dashboard/app.py

# Or run the API server
uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
cd docker
docker-compose up --build
# Dashboard: http://localhost:8501
# API: http://localhost:8000
```

## Model Performance

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Isolation Forest | — | — | — | — |
| Local Outlier Factor | — | — | — | — |
| One-Class SVM | — | — | — | — |
| Autoencoder | — | — | — | — |

*Metrics populated after first training run on the dashboard.*

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + loaded models |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/batch_predict` | Score multiple transactions |
| `GET` | `/model_metrics` | Cached evaluation metrics |

```bash
# Example: score a transaction
curl -X POST http://localhost:8000/predict?model=isolation_forest \
  -H "Content-Type: application/json" \
  -d '{"Amount": 149.62, "Time": 0, "V1": -1.36, "V2": -0.07}'
```

## Project Structure

```
├── api/
│   ├── main.py              # FastAPI endpoints
│   └── schemas.py           # Pydantic request/response models
├── dashboard/
│   └── app.py               # 6-tab Streamlit dashboard
├── data/
│   ├── creditcard_sample.csv # 10K stratified sample
│   └── generate_sample.py   # Sample generation script
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── evaluation/
│   ├── metrics.py           # Precision, recall, F1, ROC-AUC, PR curves
│   └── model_comparison.py  # Leaderboard, ROC/PR/bar chart plots
├── models/
│   ├── model_loader.py      # Load trained models from registry
│   ├── train_models.py      # Train 4 models, save to registry
│   └── saved/               # Serialized .joblib models + metadata
├── pipeline/
│   ├── preprocessing.py     # Load, split, scale data
│   └── feature_engineering.py # Engineered features
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── utils/
│   ├── config.py            # Central config (paths, hyperparams)
│   └── logger.py            # Structured logging
└── requirements.txt
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| ML Models | scikit-learn (Isolation Forest, LOF, OCSVM), PyTorch (Autoencoder) |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Data | pandas, NumPy |
| Deployment | Docker, Streamlit Cloud |

## License

MIT
