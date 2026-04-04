# Financial Transaction Anomaly Detection Platform — Design Spec

## Problem Statement

Financial institutions process millions of transactions daily. Fraudulent transactions (card theft, account takeover, synthetic identity fraud) cause billions in losses annually. Anomaly detection systems must flag suspicious transactions in real time with high recall (catch fraud) while maintaining precision (minimize false alerts that annoy customers).

This system demonstrates a production-grade ML approach to financial fraud detection using the Credit Card Fraud Detection dataset, multiple unsupervised anomaly detection models, explainability via SHAP, and a real-time simulation pipeline.

**Target users:** ML engineers, data scientists, fraud analysts, and recruiters evaluating ML engineering ability.

## Architecture

```
Credit Card Fraud Dataset (Kaggle, 284K transactions)
         |
         v
+---------------------+
|   Data Pipeline      |
|  - Load CSV/sample   |
|  - Stratified split  |
|  - Train on normals  |
+---------+-----------+
          v
+---------------------+
| Feature Engineering  |
|  - Amount log+scale  |
|  - Cyclic time enc.  |
|  - V-feature stats   |
+---------+-----------+
          v
+---------------------+        +-----------------+
|  Model Training      |------->| Model Registry  |
|  - Isolation Forest  |        | (.joblib + JSON)|
|  - LOF               |        +--------+--------+
|  - One-Class SVM     |                 |
|  - Autoencoder       |                 v
|    (PyTorch/sklearn)  |       +-----------------+
+----------------------+       |   Evaluation    |
                               | - Metrics       |
                               | - ROC/PR curves |
                               | - Comparison    |
                               +--------+--------+
                                        v
              +--------------------------------------+
              |          FastAPI Service              |
              |  POST /predict                       |
              |  POST /batch_predict                 |
              |  GET  /model_metrics                 |
              |  GET  /health                        |
              +------------------+-------------------+
                                 v
              +--------------------------------------+
              |       Streamlit Dashboard            |
              |  1. Dataset Explorer                 |
              |  2. Anomaly Detection                |
              |  3. Model Comparison                 |
              |  4. Anomaly Visualization            |
              |  5. Explainability (SHAP)            |
              |  6. Real-Time Simulation             |
              +--------------------------------------+
```

## Dataset

**Source:** Credit Card Fraud Detection (Kaggle) — 284,807 transactions over 2 days by European cardholders.

**Structure:**
- `Time`: seconds elapsed from first transaction
- `V1`-`V28`: PCA-transformed features (anonymized)
- `Amount`: transaction amount
- `Class`: 0 = normal, 1 = fraud (492 frauds, 0.17%)

**Bundling strategy:**
- 10K-row stratified sample committed to `data/creditcard_sample.csv` for instant demo
- Full dataset downloadable in-app via button (fetched from a public URL)

## Data Pipeline (`pipeline/preprocessing.py`)

1. Load CSV (sample or full)
2. Stratified train/test split (80/20), preserving fraud ratio
3. Training set: filter to normal-only (unsupervised anomaly detection paradigm)
4. Test set: both classes, fraud labels used as ground truth for evaluation
5. StandardScaler fit on training normals, applied to both sets

## Feature Engineering (`pipeline/feature_engineering.py`)

Starting features: Time, V1-V28, Amount.

Engineered features:
- `amount_log`: `log1p(Amount)` — handles right skew
- `hour_sin`, `hour_cos`: cyclic encoding of `(Time % 86400) / 3600` — captures daily patterns
- `amount_zscore`: z-score of Amount relative to training set
- `v_magnitude`: L2 norm of V1-V28 — overall distance from origin in PCA space
- `v_outlier_count`: count of V-features beyond 3 standard deviations — flags multivariate outliers

V1-V28 kept as-is (already PCA-scaled).

## Models

### Isolation Forest
- `n_estimators=200`, `contamination=0.002`, `random_state=42`, `n_jobs=-1`
- Fast, tree-based, handles high dimensionality well
- Scores: anomaly_score from `decision_function`, converted to 0-1 via sigmoid

### Local Outlier Factor
- `n_neighbors=20`, `novelty=True`, `contamination=0.002`
- Density-based, captures local anomalies missed by global methods
- Must be used with `novelty=True` to call `predict` on new data

### One-Class SVM
- `kernel='rbf'`, `nu=0.002`, `gamma='scale'`
- Learns decision boundary around normal data in kernel space
- Slower to train but strong theoretical foundation

### Autoencoder
- **PyTorch backend** (when `torch` available): 2-layer encoder (input -> 16 -> 8), 2-layer decoder (8 -> 16 -> input), ReLU activations, MSE loss, Adam optimizer, 50 epochs
- **sklearn fallback** (Streamlit Cloud): `MLPRegressor` trained to reconstruct input, reconstruction error as anomaly score
- Detection via `torch` availability check at import time
- Anomaly threshold: reconstruction error > mean + 3*std of training errors

### Model Registry (`models/saved/`)
Each trained model saved as:
- `{model_name}.joblib` — serialized model
- `{model_name}_meta.json` — training date, feature list, metrics, threshold

Directory is gitignored; models retrained on first run if missing.

## Evaluation (`evaluation/`)

### Metrics (`metrics.py`)
- Precision, Recall, F1-Score (at optimal threshold)
- ROC-AUC, Average Precision (threshold-independent)
- Confusion Matrix
- Optimal threshold selection via F1 maximization on validation set

### Model Comparison (`model_comparison.py`)
- Leaderboard table (all models, all metrics, sorted by F1)
- Overlaid ROC curves (one line per model)
- Overlaid Precision-Recall curves
- Bar chart comparing F1/Precision/Recall across models

## API (`api/`)

### `main.py` — FastAPI application

**GET /health**
Returns service status, loaded models, uptime.

**POST /predict**
```json
// Request
{"Time": 406, "V1": -1.35, ..., "V28": 0.01, "Amount": 149.62}

// Response
{"fraud_probability": 0.87, "is_fraud": true, "model": "isolation_forest", "anomaly_score": 0.87}
```

**POST /batch_predict**
```json
// Request
{"transactions": [{...}, {...}, ...], "model": "isolation_forest"}

// Response
{"predictions": [...], "fraud_count": 3, "total": 100}
```

**GET /model_metrics**
Returns evaluation metrics for all trained models.

### `schemas.py` — Pydantic models for request/response validation

## Dashboard (`dashboard/app.py`)

Six-tab Streamlit application with Plotly charts.

### Tab 1: Dataset Explorer
- Dataset summary (rows, features, fraud ratio)
- Class distribution bar chart
- Feature distribution histograms (selectable feature)
- Correlation heatmap (Amount + selected V-features)

### Tab 2: Anomaly Detection
- Model selector dropdown
- "Run Detection" button
- Results table with anomaly scores
- Threshold slider to adjust sensitivity
- Metrics update live with threshold changes

### Tab 3: Model Comparison
- Leaderboard table
- ROC curves (overlaid, interactive Plotly)
- PR curves (overlaid)
- Metric bar charts

### Tab 4: Anomaly Visualization
- PCA 2D scatter (normal vs fraud, colored)
- Amount vs anomaly score scatter
- Transaction index timeline with flagged points

### Tab 5: Explainability (SHAP)
- SHAP summary plot (beeswarm) for selected model
- SHAP waterfall for individual transaction
- Feature importance bar chart
- "Why was this flagged?" natural language summary

### Tab 6: Real-Time Simulation
- Simulates transaction stream (samples from test set with timing)
- Live-updating line chart of anomaly scores
- Alert counter for flagged transactions
- Running precision/recall stats

## Utilities

### `utils/config.py`
Central configuration: file paths, model hyperparameters, thresholds, feature lists. Uses dataclass or dict, no external config files needed.

### `utils/logger.py`
Python `logging` module configured with structured format. Log levels: INFO for pipeline steps, WARNING for threshold violations, ERROR for failures.

## Project Structure

```
anomaly-detection-system/
├── data/
│   └── creditcard_sample.csv       # 10K-row bundled sample
├── models/
│   ├── train_models.py             # Train all 4 models
│   ├── model_loader.py             # Load from registry
│   └── saved/                      # .joblib + .json (gitignored)
├── pipeline/
│   ├── preprocessing.py            # Load, split, scale
│   └── feature_engineering.py      # Engineered features
├── api/
│   ├── main.py                     # FastAPI endpoints
│   └── schemas.py                  # Pydantic models
├── dashboard/
│   └── app.py                      # 6-tab Streamlit dashboard
├── evaluation/
│   ├── metrics.py                  # Compute metrics
│   └── model_comparison.py         # Comparison charts
├── utils/
│   ├── logger.py                   # Structured logging
│   └── config.py                   # Central configuration
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

## Deployment

- **Local:** `pip install -r requirements.txt`, `python models/train_models.py`, `streamlit run dashboard/app.py`
- **Docker:** `docker-compose up --build` from `docker/`
- **Streamlit Cloud:** Push to GitHub, connect repo, set `dashboard/app.py` as entry point
- **Keep-alive:** GitHub Actions pings every 12h

## Dependencies

```
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
shap>=0.43.0
joblib>=1.3.0
python-dotenv>=1.0.0
torch>=2.0.0  # optional, autoencoder backend
```

Streamlit Cloud: `torch` excluded from requirements, sklearn autoencoder used automatically.
