"""
Advanced Anomaly Detection Dashboard.

Six-tab Streamlit application providing dataset exploration, multi-model
anomaly detection, model comparison, threshold tuning, explainability
with optional Gemini RCA, and real-time streaming simulation.
"""

import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so sibling packages are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.generator import generate_sample_logs
from data_pipeline.feature_engineering import FeatureEngineer
from models.engine import AnomalyEngine
from evaluation.metrics import ModelEvaluator
from app import GeminiRCA, AnomalyEvent, RCAReport

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="\U0001f50d",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRIC_COLS = ["error_rate", "latency_ms", "cpu_pct", "memory_pct"]
MODEL_OPTIONS = ["isolation_forest", "lof", "dbscan", "autoencoder"]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
DEFAULTS = {
    "logs_df": None,
    "feature_df": None,
    "anomalies": None,
    "model_results": None,
    "selected_model": "isolation_forest",
    "streaming": False,
    "stream_buffer": [],
    "engine": None,
    "gemini_rca": None,
    "gemini_available": False,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _get_gemini_key() -> str:
    """Retrieve Gemini API key from env or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return key


def _init_engine() -> AnomalyEngine:
    """Train the engine once on 5000 sample logs and cache it."""
    if st.session_state.engine is not None:
        return st.session_state.engine
    with st.spinner("Training anomaly engine on 5 000 sample logs..."):
        fe = FeatureEngineer()
        train_df = generate_sample_logs(5000, anomaly_pct=0.05, seed=42)
        train_features = fe.transform(train_df)
        engine = AnomalyEngine()
        engine.train(train_features)
        st.session_state.engine = engine
    return engine


def _init_gemini():
    """Initialise Gemini RCA client if key is available."""
    if st.session_state.gemini_rca is not None:
        return
    key = _get_gemini_key()
    if key:
        try:
            st.session_state.gemini_rca = GeminiRCA(key)
            st.session_state.gemini_available = True
        except Exception:
            st.session_state.gemini_available = False
    else:
        st.session_state.gemini_available = False


# Run initialisers
engine = _init_engine()
_init_gemini()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("\U0001f50d Anomaly Detection System")
    st.caption("Multi-model anomaly detection with Gemini-powered root cause analysis.")
    st.markdown("---")

    st.session_state.selected_model = st.selectbox(
        "Model", MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.selected_model),
    )

    st.markdown("---")
    if st.session_state.gemini_available:
        st.success("Gemini API: Connected")
    else:
        st.warning("Gemini API: Not configured")
        st.caption("Set `GEMINI_API_KEY` env var to enable RCA.")

    st.markdown("---")
    st.markdown(
        "[GitHub Repo](https://github.com/sanjitmathur/distributed-anomaly-rca)"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001f4ca Dataset Explorer",
    "\U0001f6a8 Anomaly Detection",
    "\U0001f3c6 Model Comparison",
    "\U0001f39a\ufe0f Threshold Tuning",
    "\U0001f9e0 Explainability",
    "\U0001f4e1 Real-Time Simulation",
])

# ===================================================================
# TAB 1 - Dataset Explorer
# ===================================================================
with tab1:
    st.header("Dataset Explorer")

    col_sample, col_upload = st.columns(2)
    with col_sample:
        if st.button("Use Sample Data (500 logs)", use_container_width=True):
            st.session_state.logs_df = generate_sample_logs(500)
            st.session_state.feature_df = None
            st.session_state.anomalies = None
            st.success("Generated 500 sample log entries.")

    with col_upload:
        uploaded = st.file_uploader("Upload JSON or CSV", type=["json", "csv"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    st.session_state.logs_df = pd.read_csv(uploaded)
                else:
                    st.session_state.logs_df = pd.DataFrame(json.load(uploaded))
                st.session_state.feature_df = None
                st.session_state.anomalies = None
                st.success(f"Loaded {len(st.session_state.logs_df)} entries.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

    df = st.session_state.logs_df
    if df is not None:
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Summary statistics"):
            st.dataframe(df.describe(), use_container_width=True)

        # Time series subplots
        st.subheader("Metric Time Series")
        ts_df = df.copy()
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Error Rate", "Latency (ms)", "CPU %", "Memory %"],
            vertical_spacing=0.04,
        )
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

        for i, col in enumerate(METRIC_COLS):
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=ts_df["timestamp"], y=ts_df[col],
                    mode="lines", name=col, line=dict(color=colors[i], width=1),
                ),
                row=row, col=1,
            )
            # Highlight known anomalies
            if "is_anomaly" in ts_df.columns:
                anom_mask = ts_df["is_anomaly"].astype(bool)
                if anom_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=ts_df.loc[anom_mask, "timestamp"],
                            y=ts_df.loc[anom_mask, col],
                            mode="markers", name=f"{col} anomaly",
                            marker=dict(color="red", size=4, symbol="x"),
                            showlegend=(row == 1),
                        ),
                        row=row, col=1,
                    )

        fig.update_layout(height=700, showlegend=True, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load a dataset to begin exploring.")

# ===================================================================
# TAB 2 - Anomaly Detection
# ===================================================================
with tab2:
    st.header("Anomaly Detection")

    if st.session_state.logs_df is None:
        st.info("Load a dataset in the Dataset Explorer tab first.")
    else:
        model_name = st.session_state.selected_model
        st.write(f"**Selected model:** `{model_name}`")

        if st.button("Run Detection", type="primary", use_container_width=True):
            with st.spinner("Engineering features & running detection..."):
                fe = FeatureEngineer()
                feat_df = fe.transform(st.session_state.logs_df)
                st.session_state.feature_df = feat_df
                preds = engine.predict(feat_df, model_name)
                st.session_state.anomalies = preds

        preds = st.session_state.anomalies
        if preds is not None:
            total = len(preds)
            n_anom = int(preds["is_anomaly"].sum())
            rate = n_anom / total * 100 if total else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Logs", f"{total:,}")
            m2.metric("Anomalies Found", n_anom)
            m3.metric("Anomaly Rate", f"{rate:.1f}%")

            # Scatter: timestamp vs anomaly_score
            plot_df = st.session_state.logs_df.copy()
            plot_df["anomaly_score"] = preds["anomaly_score"].values
            plot_df["predicted_anomaly"] = preds["is_anomaly"].values
            plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

            fig = px.scatter(
                plot_df, x="timestamp", y="anomaly_score",
                color="predicted_anomaly",
                color_discrete_map={True: "#EF553B", False: "#636EFA"},
                hover_data=["pod", "error_rate", "latency_ms", "cpu_pct"],
                title="Anomaly Scores Over Time",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Table of detected anomalies
            st.subheader("Detected Anomalies")
            anom_rows = plot_df[plot_df["predicted_anomaly"]].sort_values(
                "anomaly_score", ascending=False
            )
            st.dataframe(
                anom_rows[["timestamp", "pod", "anomaly_score"] + METRIC_COLS],
                use_container_width=True,
            )

# ===================================================================
# TAB 3 - Model Comparison
# ===================================================================
with tab3:
    st.header("Model Comparison")

    if st.session_state.logs_df is None:
        st.info("Load a dataset first.")
    elif "is_anomaly" not in st.session_state.logs_df.columns:
        st.warning("Ground truth column `is_anomaly` not found. Comparison needs labels.")
    else:
        if st.button("Compare All Models", type="primary", use_container_width=True):
            with st.spinner("Running all models and evaluating..."):
                fe = FeatureEngineer()
                feat_df = fe.transform(st.session_state.logs_df)
                st.session_state.feature_df = feat_df
                all_preds = engine.predict_all(feat_df)
                y_true = st.session_state.logs_df["is_anomaly"].astype(int).values

                evaluator = ModelEvaluator()
                eval_results = {}
                for name, pred_df in all_preds.items():
                    metrics = evaluator.evaluate(
                        y_true,
                        pred_df["is_anomaly"].astype(int).values,
                        pred_df["anomaly_score"].values,
                    )
                    eval_results[name] = metrics

                st.session_state.model_results = eval_results

        results = st.session_state.model_results
        if results is not None:
            evaluator = ModelEvaluator()
            leaderboard = evaluator.compare_models(results)

            st.subheader("Leaderboard")
            st.dataframe(
                leaderboard.style.format({
                    "precision": "{:.3f}", "recall": "{:.3f}",
                    "f1": "{:.3f}", "roc_auc": "{:.3f}",
                }),
                use_container_width=True,
            )

            # Grouped bar chart
            fig = go.Figure()
            for metric in ["precision", "recall", "f1"]:
                fig.add_trace(go.Bar(
                    x=leaderboard["model"], y=leaderboard[metric], name=metric.title(),
                ))
            fig.update_layout(
                barmode="group", title="Precision / Recall / F1 by Model",
                yaxis_title="Score", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ROC curves
            st.subheader("ROC Curves")
            fe = FeatureEngineer()
            feat_df = st.session_state.feature_df
            if feat_df is None:
                feat_df = fe.transform(st.session_state.logs_df)
            all_preds = engine.predict_all(feat_df)
            y_true = st.session_state.logs_df["is_anomaly"].astype(int).values

            from sklearn.metrics import roc_curve

            fig_roc = go.Figure()
            for name, pred_df in all_preds.items():
                scores = pred_df["anomaly_score"].values
                if len(np.unique(y_true)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, scores)
                auc_val = results[name]["roc_auc"]
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={auc_val:.3f})",
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"), name="Random",
            ))
            fig_roc.update_layout(
                title="ROC Curves", xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate", height=450,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ===================================================================
# TAB 4 - Threshold Tuning
# ===================================================================
with tab4:
    st.header("Threshold Tuning")

    if st.session_state.anomalies is None:
        st.info("Run anomaly detection first (Tab 2).")
    elif "is_anomaly" not in st.session_state.logs_df.columns:
        st.warning("Ground truth labels required for threshold tuning.")
    else:
        scores = st.session_state.anomalies["anomaly_score"].values
        y_true = st.session_state.logs_df["is_anomaly"].astype(int).values

        threshold = st.slider(
            "Anomaly Threshold", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01,
        )

        y_pred_t = (scores >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score

        prec = precision_score(y_true, y_pred_t, zero_division=0)
        rec = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec:.3f}")
        c2.metric("Recall", f"{rec:.3f}")
        c3.metric("F1 Score", f"{f1:.3f}")

        # Full threshold sweep chart
        evaluator = ModelEvaluator()
        thresholds = np.linspace(0, 1, 100)
        sweep_df = evaluator.threshold_analysis(y_true, scores, thresholds)

        fig = go.Figure()
        for metric in ["precision", "recall", "f1"]:
            fig.add_trace(go.Scatter(
                x=sweep_df["threshold"], y=sweep_df[metric],
                mode="lines", name=metric.title(),
            ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Current: {threshold:.2f}")
        fig.update_layout(
            title="Threshold vs Metrics",
            xaxis_title="Threshold", yaxis_title="Score", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 5 - Explainability
# ===================================================================
with tab5:
    st.header("Explainability")

    if st.session_state.anomalies is None or st.session_state.feature_df is None:
        st.info("Run anomaly detection first (Tab 2).")
    else:
        preds = st.session_state.anomalies
        detected = preds[preds["is_anomaly"]]

        if detected.empty:
            st.info("No anomalies detected to explain.")
        else:
            logs_df = st.session_state.logs_df
            options = []
            indices = detected["index"].values
            for idx in indices:
                row = logs_df.iloc[idx]
                label = (
                    f"#{idx} | {row.get('pod', 'N/A')} | "
                    f"score={preds.loc[preds['index'] == idx, 'anomaly_score'].values[0]:.3f}"
                )
                options.append(label)

            selected_option = st.selectbox("Select an anomaly to explain", options)
            selected_idx = indices[options.index(selected_option)]

            # Feature importance
            st.subheader("Feature Importance")
            model_name = st.session_state.selected_model
            with st.spinner("Computing permutation importance..."):
                importance = engine.get_feature_importance(
                    st.session_state.feature_df, model_name, n_repeats=3,
                )

            imp_df = pd.DataFrame(
                sorted(importance.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            fig_imp = px.bar(
                imp_df.head(10), x="Importance", y="Feature",
                orientation="h", title="Top 10 Feature Importances",
                color="Importance", color_continuous_scale="Reds",
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Radar chart: anomaly vs dataset mean
            st.subheader("Anomaly Metrics vs Dataset Mean")
            row_vals = logs_df.iloc[selected_idx]
            means = logs_df[METRIC_COLS].mean()

            categories = METRIC_COLS + [METRIC_COLS[0]]  # close the polygon
            anom_values = [float(row_vals[c]) for c in METRIC_COLS]
            mean_values = [float(means[c]) for c in METRIC_COLS]
            # Normalize to 0-1 for radar comparison
            maxes = [max(abs(a), abs(m), 1e-9) for a, m in zip(anom_values, mean_values)]
            anom_norm = [a / mx for a, mx in zip(anom_values, maxes)] + [anom_values[0] / maxes[0]]
            mean_norm = [m / mx for m, mx in zip(mean_values, maxes)] + [mean_values[0] / maxes[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=anom_norm, theta=categories, fill="toself",
                name="Anomaly", line=dict(color="#EF553B"),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=mean_norm, theta=categories, fill="toself",
                name="Dataset Mean", line=dict(color="#636EFA"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                title="Anomaly vs Mean (Normalised)", height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Raw values comparison
            st.markdown("**Raw metric values:**")
            comp_df = pd.DataFrame({
                "Metric": METRIC_COLS,
                "Anomaly Value": [float(row_vals[c]) for c in METRIC_COLS],
                "Dataset Mean": [float(means[c]) for c in METRIC_COLS],
            })
            st.dataframe(comp_df, use_container_width=True)

            # Gemini RCA
            if st.session_state.gemini_available:
                st.markdown("---")
                st.subheader("AI Root Cause Analysis (Gemini)")
                if st.button("Analyze with Gemini", type="primary"):
                    anomaly_event = AnomalyEvent(
                        timestamp=str(row_vals.get("timestamp", "")),
                        anomaly_score=float(
                            preds.loc[preds["index"] == selected_idx, "anomaly_score"].values[0]
                        ),
                        pod=str(row_vals.get("pod", "unknown")),
                        error_rate=float(row_vals.get("error_rate", 0)),
                        latency_ms=float(row_vals.get("latency_ms", 0)),
                        cpu_pct=float(row_vals.get("cpu_pct", 0)),
                        memory_pct=float(row_vals.get("memory_pct", 0)),
                    )
                    with st.spinner("Querying Gemini..."):
                        rca = st.session_state.gemini_rca.generate_rca(anomaly_event)
                    st.markdown(f"**Root Cause:** {rca.root_cause}")
                    st.markdown(f"**Confidence:** {rca.confidence:.0%}")
                    st.markdown(f"**Reasoning:** {rca.reasoning}")
                    if rca.affected_services:
                        st.markdown(f"**Affected Services:** {', '.join(rca.affected_services)}")
                    st.markdown(f"**Remediation:** {rca.remediation}")
            else:
                st.caption("Set GEMINI_API_KEY to enable AI root cause analysis.")

# ===================================================================
# TAB 6 - Real-Time Simulation
# ===================================================================
with tab6:
    st.header("Real-Time Streaming Simulation")

    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("Start Streaming", type="primary", use_container_width=True)
    with col_stop:
        stop_btn = st.button("Stop", use_container_width=True)

    if stop_btn:
        st.session_state.streaming = False

    if start_btn:
        st.session_state.streaming = True
        st.session_state.stream_buffer = []

    if st.session_state.streaming:
        chart_slot = st.empty()
        alert_slot = st.empty()
        progress_slot = st.empty()

        rng = np.random.default_rng(int(time.time()))
        fe = FeatureEngineer()
        buffer = st.session_state.stream_buffer

        for iteration in range(100):
            if not st.session_state.streaming:
                break

            # Generate one new entry (10% chance of anomaly)
            is_anom = rng.random() < 0.10
            if is_anom:
                entry = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "pod": rng.choice(["api-gw", "auth-svc", "db-svc", "cache-svc"]),
                    "error_rate": float(np.clip(rng.normal(0.2, 0.08), 0, 1)),
                    "latency_ms": float(np.clip(rng.normal(400, 120), 1, None)),
                    "cpu_pct": float(np.clip(rng.normal(80, 10), 0, 100)),
                    "memory_pct": float(rng.uniform(60, 95)),
                }
            else:
                entry = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "pod": rng.choice(["api-gw", "auth-svc", "db-svc", "cache-svc"]),
                    "error_rate": float(np.clip(rng.normal(0.001, 0.0005), 0, 1)),
                    "latency_ms": float(np.clip(rng.normal(50, 10), 1, None)),
                    "cpu_pct": float(np.clip(rng.normal(30, 8), 0, 100)),
                    "memory_pct": float(rng.uniform(20, 60)),
                }

            buffer.append(entry)

            # Keep only last 100
            window = buffer[-100:]
            window_df = pd.DataFrame(window)
            window_df["timestamp"] = pd.to_datetime(window_df["timestamp"])

            # Run detection on the window
            try:
                feat_window = fe.transform(window_df)
                model_name = st.session_state.selected_model
                preds_window = engine.predict(feat_window, model_name)
                window_df["anomaly_score"] = preds_window["anomaly_score"].values
                window_df["detected"] = preds_window["is_anomaly"].values
            except Exception:
                window_df["anomaly_score"] = 0.0
                window_df["detected"] = False

            # Live chart
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=["Latency (ms)", "Anomaly Score"],
                vertical_spacing=0.12,
            )
            normal_mask = ~window_df["detected"]
            anom_mask = window_df["detected"]

            fig.add_trace(go.Scatter(
                x=window_df.loc[normal_mask, "timestamp"],
                y=window_df.loc[normal_mask, "latency_ms"],
                mode="lines+markers", name="Normal",
                marker=dict(color="#636EFA", size=4),
            ), row=1, col=1)

            if anom_mask.any():
                fig.add_trace(go.Scatter(
                    x=window_df.loc[anom_mask, "timestamp"],
                    y=window_df.loc[anom_mask, "latency_ms"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#EF553B", size=8, symbol="x"),
                ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=window_df["timestamp"], y=window_df["anomaly_score"],
                mode="lines+markers", name="Score",
                marker=dict(
                    color=window_df["anomaly_score"],
                    colorscale="Reds", size=5,
                ),
                line=dict(color="#AB63FA"),
            ), row=2, col=1)

            fig.update_layout(height=450, showlegend=True, margin=dict(t=40, b=20))
            chart_slot.plotly_chart(fig, use_container_width=True)

            # Alert
            if anom_mask.any():
                last_anom = window_df[anom_mask].iloc[-1]
                alert_slot.error(
                    f"ALERT: Anomaly detected on **{last_anom['pod']}** "
                    f"at {last_anom['timestamp']} "
                    f"(score: {last_anom['anomaly_score']:.3f})"
                )
            else:
                alert_slot.success("All systems nominal.")

            progress_slot.progress(
                (iteration + 1) / 100,
                text=f"Iteration {iteration + 1}/100 | Buffer: {len(buffer)} entries",
            )

            time.sleep(0.5)

        st.session_state.streaming = False
        st.info("Streaming complete.")
    else:
        st.caption(
            "Click **Start Streaming** to simulate a live data feed. "
            "Each iteration generates one log entry and runs detection on a "
            "rolling window of the last 100 entries."
        )
