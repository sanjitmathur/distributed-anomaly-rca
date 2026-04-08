# System Architecture

## Mermaid Diagram

```mermaid
flowchart TB
    subgraph Docker["docker-compose.yml"]

        subgraph Producer["producer container"]
            PG[Mock Transaction Generator]
            PG -->|"synthetic txn JSON<br/>10 TPS, 0.5% fraud"| POST[HTTP POST /ingest]
        end

        subgraph Inference["inference_service container"]
            direction TB
            INGEST["/ingest endpoint"]
            VAL[Pydantic Validation<br/>RawTransaction schema]
            SW[Sliding Window Aggregator<br/>10-min frequency/amount stats]
            GF[Graph Feature Extractor<br/>IP↔Device bipartite degree]
            FE[Feature Engineering<br/>log-amount, cyclical time, V-magnitude]
            MODEL[Model Scoring<br/>Isolation Forest / LOF / OCSVM / AE]
            DRIFT[KS-Test Drift Detector<br/>p-value monitoring]
            SSE["/stream SSE endpoint"]
            STATS["/stats + /drift endpoints"]

            INGEST --> VAL --> SW --> GF --> FE --> MODEL --> DRIFT
            DRIFT --> SSE
            DRIFT --> STATS
        end

        subgraph Dashboard["dashboard container"]
            direction TB
            ST[Streamlit UI — 6 tabs]
            T1[Dataset Explorer]
            T2[Anomaly Detection]
            T3[Model Comparison]
            T4[Visualization]
            T5[Explainability — SHAP]
            T6[Live Streaming Monitor]

            ST --- T1 & T2 & T3 & T4 & T5 & T6
            T6 -->|"SSE consume"| SSE
        end

        POST --> INGEST
    end

    subgraph External["External"]
        CSV[(creditcard.csv<br/>284K rows)]
        TRAIN[Model Training<br/>at build time]
        CSV --> TRAIN --> MODEL
    end

    style Producer fill:#1a1f2e,stroke:#06d6a0,color:#e8eaed
    style Inference fill:#1a1f2e,stroke:#118ab2,color:#e8eaed
    style Dashboard fill:#1a1f2e,stroke:#ffd166,color:#e8eaed
    style Docker fill:#0a0e17,stroke:#2a3042,color:#9ca3af
    style External fill:#111827,stroke:#2a3042,color:#9ca3af
```

## Data Flow

```
Producer → HTTP POST → /ingest → Pydantic Validation → Sliding Window → Graph Features
    → Feature Engineering → Model Scoring → KS Drift Check → SSE Buffer → Dashboard
```

## Component Details

| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| **Producer** | Python thread / standalone | — | Synthetic transaction generation at configurable TPS |
| **Inference Service** | FastAPI + uvicorn | 8000 | Validation, feature enrichment, scoring, drift detection |
| **Dashboard** | Streamlit | 8501 | 6-tab UI with live streaming monitor |

## Feature Pipeline

| Stage | Features Added |
|-------|---------------|
| **Raw** | Time, V1-V28, Amount |
| **Engineered** | amount_log, amount_zscore, hour_sin, hour_cos, v_magnitude, v_outlier_count |
| **Sliding Window** | txn_count_10m, txn_amount_mean_10m, txn_amount_std_10m, txn_velocity_per_min |
| **Graph** | ip_degree, device_degree, shared_infra_score |

## Concept Drift Detection

- **Method**: Two-sample Kolmogorov-Smirnov test
- **Reference**: First 500 anomaly scores after warm-up
- **Window**: Rolling 500 observations
- **Threshold**: p < 0.01 triggers drift alert
- **Action**: Logged + surfaced in dashboard metrics + `/drift` API endpoint
