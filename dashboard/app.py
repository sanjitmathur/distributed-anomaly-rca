"""Financial Fraud Detection Dashboard — 6-tab Streamlit application."""

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, preprocess
from pipeline.feature_engineering import FeatureEngineer
from models.train_models import train_all_models, _score_model
from models.model_loader import load_model, list_models
from evaluation.metrics import compute_metrics, find_optimal_threshold
from evaluation.model_comparison import (
    create_leaderboard,
    plot_roc_curves,
    plot_pr_curves,
    plot_metric_bars,
)
from utils.config import ALL_FEATURES, MODEL_NAMES, SAMPLE_CSV

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.df_raw = None
    st.session_state.preprocessed = None
    st.session_state.df_engineered = None
    st.session_state.model_results = None
    st.session_state.all_metrics = None


def _initialize():
    """Load data, engineer features, train models on first run."""
    if st.session_state.initialized:
        return

    with st.spinner("Loading dataset..."):
        df = load_data()
        st.session_state.df_raw = df

    with st.spinner("Engineering features..."):
        fe = FeatureEngineer()
        df_eng = fe.transform(df.drop(columns=["Class"]))
        df_eng["Class"] = df["Class"].values
        st.session_state.df_engineered = df_eng

    with st.spinner("Preprocessing..."):
        result = preprocess(df_eng)
        st.session_state.preprocessed = result

    with st.spinner("Training models (this may take a minute)..."):
        model_results = train_all_models(
            result["X_train"], result["X_test"], result["y_test"], save=True,
        )
        st.session_state.model_results = model_results

    with st.spinner("Evaluating models..."):
        all_metrics = {}
        for name, res in model_results.items():
            all_metrics[name] = compute_metrics(result["y_test"], res["scores"])
        st.session_state.all_metrics = all_metrics

    st.session_state.initialized = True


_initialize()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.caption("Production-grade ML Platform")
    st.markdown("---")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.metric("Transactions", f"{len(df):,}")
        st.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
        st.metric("Fraud Rate", f"{df['Class'].mean():.2%}")
    st.markdown("---")
    st.markdown("**Models:** 4 trained")
    st.markdown(
        "Isolation Forest · LOF · "
        "One-Class SVM · Autoencoder"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("🛡️ Real-Time Financial Fraud Detection Platform")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Explorer",
    "🔍 Anomaly Detection",
    "📈 Model Comparison",
    "🗺️ Visualization",
    "🧠 Explainability",
    "⚡ Real-Time Simulation",
])

# ===================== Tab 1: Dataset Explorer =====================
with tab1:
    st.subheader("Dataset Overview")
    df = st.session_state.df_raw

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns) - 1}")
        col3.metric("Fraud Count", f"{int(df['Class'].sum()):,}")
        col4.metric("Fraud Ratio", f"{df['Class'].mean():.2%}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Class Distribution")
            class_counts = df["Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Label"] = class_counts["Class"].map({0: "Normal", 1: "Fraud"})
            fig = px.bar(
                class_counts, x="Label", y="Count", color="Label",
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                text="Count",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Feature Distribution")
            feature = st.selectbox(
                "Select feature",
                ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)],
            )
            fig = px.histogram(
                df, x=feature, color="Class", nbins=50,
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                barmode="overlay", opacity=0.7,
                labels={"Class": "Label"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Correlation Heatmap")
        corr_features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 11)]
        corr = df[corr_features].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 2: Anomaly Detection =====================
with tab2:
    st.subheader("Run Anomaly Detection")

    if st.session_state.model_results:
        model_name = st.selectbox("Select Model", MODEL_NAMES, key="detect_model")
        results = st.session_state.model_results
        preprocessed = st.session_state.preprocessed
        scores = results[model_name]["scores"]
        y_test = preprocessed["y_test"]

        threshold = st.slider(
            "Detection Threshold", 0.0, 1.0, 0.5, 0.01, key="detect_thresh",
        )

        y_pred = (scores >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())

        c1, c2, c3, c4 = st.columns(4)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        c1.metric("Precision", f"{precision:.3f}")
        c2.metric("Recall", f"{recall:.3f}")
        c3.metric("F1 Score", f"{f1:.3f}")
        c4.metric("Flagged", f"{int(y_pred.sum()):,}")

        st.markdown("---")
        st.subheader("Flagged Transactions")
        flagged_idx = np.where(y_pred == 1)[0]
        if len(flagged_idx) > 0:
            flagged_df = pd.DataFrame({
                "Index": flagged_idx,
                "Anomaly Score": scores[flagged_idx].round(4),
                "Actual": ["Fraud" if y_test[i] == 1 else "Normal" for i in flagged_idx],
            })
            st.dataframe(flagged_df.head(50), use_container_width=True)
        else:
            st.info("No transactions flagged at this threshold.")

# ===================== Tab 3: Model Comparison =====================
with tab3:
    st.subheader("Model Performance Comparison")

    if st.session_state.all_metrics:
        metrics = st.session_state.all_metrics

        st.markdown("#### Leaderboard")
        leaderboard = create_leaderboard(metrics)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_roc_curves(metrics), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pr_curves(metrics), use_container_width=True)

        st.plotly_chart(plot_metric_bars(metrics), use_container_width=True)

# ===================== Tab 4: Visualization =====================
with tab4:
    st.subheader("Anomaly Visualization")

    if st.session_state.model_results and st.session_state.preprocessed:
        viz_model = st.selectbox("Select Model", MODEL_NAMES, key="viz_model")
        scores = st.session_state.model_results[viz_model]["scores"]
        y_test = st.session_state.preprocessed["y_test"]
        X_test = st.session_state.preprocessed["X_test"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### PCA Projection")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_test[:2000])
            pca_df = pd.DataFrame({
                "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test[:2000]],
                "Score": scores[:2000],
            })
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color="Label",
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                opacity=0.6, hover_data=["Score"],
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Anomaly Score Distribution")
            score_df = pd.DataFrame({
                "Score": scores,
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
            })
            fig = px.histogram(
                score_df, x="Score", color="Label", nbins=50,
                color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                barmode="overlay", opacity=0.7,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Transaction Timeline")
        timeline_df = pd.DataFrame({
            "Index": np.arange(len(scores)),
            "Anomaly Score": scores,
            "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
        })
        fig = px.scatter(
            timeline_df, x="Index", y="Anomaly Score", color="Label",
            color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
            opacity=0.5,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 5: Explainability =====================
with tab5:
    st.subheader("Model Explainability (SHAP)")

    if st.session_state.model_results and st.session_state.preprocessed:
        explain_model = st.selectbox(
            "Select Model", ["isolation_forest", "lof"], key="explain_model",
            help="SHAP works best with tree-based and distance-based models",
        )

        preprocessed = st.session_state.preprocessed
        X_test = preprocessed["X_test"]
        feature_names = preprocessed["feature_names"]
        scores = st.session_state.model_results[explain_model]["scores"]

        n_explain = st.slider("Samples to explain", 50, 500, 100, 50)

        if st.button("🧠 Compute SHAP Values", type="primary"):
            with st.spinner("Computing SHAP values (this may take a moment)..."):
                import shap

                model_obj = st.session_state.model_results[explain_model]["model"]

                if explain_model == "isolation_forest":
                    explainer = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(X_test[:n_explain])
                else:
                    explainer = shap.KernelExplainer(
                        model_obj.decision_function,
                        shap.sample(X_test, 50),
                    )
                    shap_values = explainer.shap_values(X_test[:n_explain])

                st.session_state.shap_values = shap_values
                st.session_state.shap_X = X_test[:n_explain]
                st.session_state.shap_features = feature_names

        if "shap_values" in st.session_state:
            shap_vals = st.session_state.shap_values
            shap_X = st.session_state.shap_X
            feat_names = st.session_state.shap_features

            st.markdown("#### Feature Importance")
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feat_names[:len(mean_abs)],
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(
                importance_df, x="Mean |SHAP|", y="Feature",
                orientation="h", color="Mean |SHAP|",
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Individual Transaction Explanation")
            flagged_idx = np.where(scores[:n_explain] > 0.5)[0]
            if len(flagged_idx) > 0:
                selected = st.selectbox(
                    "Select flagged transaction",
                    flagged_idx,
                    format_func=lambda i: f"Transaction {i} (score: {scores[i]:.3f})",
                )
                txn_shap = shap_vals[selected]
                txn_df = pd.DataFrame({
                    "Feature": feat_names[:len(txn_shap)],
                    "SHAP Value": txn_shap,
                }).sort_values("SHAP Value", key=abs, ascending=True).tail(10)

                fig = px.bar(
                    txn_df, x="SHAP Value", y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                )
                fig.update_layout(height=400, title=f"Why Transaction {selected} Was Flagged")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top contributing features:**")
                top = txn_df.tail(3).iloc[::-1]
                for _, row in top.iterrows():
                    direction = "increases" if row["SHAP Value"] > 0 else "decreases"
                    st.markdown(
                        f"- **{row['Feature']}** {direction} fraud risk "
                        f"(SHAP: {row['SHAP Value']:+.4f})"
                    )
            else:
                st.info("No flagged transactions in the explained sample.")

# ===================== Tab 6: Real-Time Simulation =====================
with tab6:
    st.subheader("Real-Time Transaction Monitoring")

    if st.session_state.model_results and st.session_state.preprocessed:
        sim_model = st.selectbox("Select Model", MODEL_NAMES, key="sim_model")

        c1, c2 = st.columns([1, 3])
        with c1:
            n_transactions = st.number_input("Transactions to simulate", 20, 200, 50)
            speed = st.slider("Speed (ms between transactions)", 50, 500, 100)

        if st.button("▶️ Start Simulation", type="primary"):
            X_test = st.session_state.preprocessed["X_test"]
            y_test = st.session_state.preprocessed["y_test"]
            model_result = st.session_state.model_results[sim_model]

            # Sample random transactions
            rng = np.random.default_rng()
            indices = rng.choice(len(X_test), n_transactions, replace=False)

            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            alert_placeholder = st.empty()

            scores_history = []
            labels_history = []
            alerts = []

            for i, idx in enumerate(indices):
                x = X_test[idx : idx + 1]
                score = float(_score_model(sim_model, model_result, x)[0])
                label = int(y_test[idx])

                scores_history.append(score)
                labels_history.append(label)

                if score > 0.5:
                    alerts.append(f"🚨 Transaction {i+1}: score={score:.3f} ({'FRAUD' if label == 1 else 'FALSE ALARM'})")

                # Update chart
                sim_df = pd.DataFrame({
                    "Transaction": range(1, len(scores_history) + 1),
                    "Anomaly Score": scores_history,
                    "Label": ["Fraud" if l == 1 else "Normal" for l in labels_history],
                })
                fig = px.scatter(
                    sim_df, x="Transaction", y="Anomaly Score",
                    color="Label",
                    color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                              annotation_text="Threshold")
                fig.update_layout(height=350)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Update metrics
                flagged = sum(1 for s in scores_history if s > 0.5)
                actual = sum(labels_history)
                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Processed", i + 1)
                    m2.metric("Flagged", flagged)
                    m3.metric("Actual Fraud", actual)
                    m4.metric("Alert Rate", f"{flagged/(i+1):.1%}")

                # Update alerts
                if alerts:
                    with alert_placeholder.container():
                        st.markdown("#### Recent Alerts")
                        for alert in alerts[-5:]:
                            st.markdown(alert)

                time.sleep(speed / 1000)

            st.success(f"Simulation complete: {n_transactions} transactions processed.")
