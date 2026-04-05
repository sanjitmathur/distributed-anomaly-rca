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
# Page config & custom theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark modern theme with accent colors
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-card-hover: #1f2537;
        --accent-cyan: #06d6a0;
        --accent-red: #ef476f;
        --accent-amber: #ffd166;
        --accent-blue: #118ab2;
        --text-primary: #e8eaed;
        --text-secondary: #9ca3af;
        --border-color: #2a3042;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0d1321 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
        border-right: 1px solid #2a3042;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #9ca3af;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #06d6a0 0%, #118ab2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #1f2537 100%);
        border: 1px solid #2a3042;
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #06d6a0;
        box-shadow: 0 0 20px rgba(6, 214, 160, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e8eaed !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #111827;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2a3042;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #6b7280;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 10px 18px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e8eaed;
        background: #1a1f2e;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #06d6a0 0%, #118ab2 100%) !important;
        color: #0a0e17 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a3042;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06d6a0 0%, #118ab2 100%);
        color: #0a0e17;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(6, 214, 160, 0.3);
        transform: translateY(-1px);
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        color: #06d6a0;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #2a3042;
    }

    /* Section headers */
    .section-header {
        color: #e8eaed;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #06d6a0;
        display: inline-block;
    }

    /* Card containers */
    .glass-card {
        background: rgba(26, 31, 46, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid #2a3042;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-fraud {
        background: rgba(239, 71, 111, 0.15);
        color: #ef476f;
        border: 1px solid rgba(239, 71, 111, 0.3);
    }
    .status-normal {
        background: rgba(6, 214, 160, 0.15);
        color: #06d6a0;
        border: 1px solid rgba(6, 214, 160, 0.3);
    }

    /* Alert box */
    .alert-fraud {
        background: rgba(239, 71, 111, 0.08);
        border-left: 3px solid #ef476f;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 6px 0;
        color: #e8eaed;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* Glossary section */
    .glossary-container {
        background: linear-gradient(135deg, #111827 0%, #1a1f2e 100%);
        border: 1px solid #2a3042;
        border-radius: 16px;
        padding: 32px;
        margin-top: 3rem;
    }
    .glossary-title {
        background: linear-gradient(135deg, #ffd166 0%, #ef476f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .glossary-term {
        color: #06d6a0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .glossary-def {
        color: #9ca3af;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 0.4rem;
    }

    /* Divider */
    hr {
        border-color: #2a3042 !important;
        margin: 1.5rem 0;
    }

    /* Hide default streamlit footer */
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a1f2e;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme defaults
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,31,46,0.4)",
    font=dict(family="Inter, sans-serif", color="#9ca3af", size=12),
    title_font=dict(color="#e8eaed", size=16, family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#2a3042", zerolinecolor="#2a3042"),
    yaxis=dict(gridcolor="#2a3042", zerolinecolor="#2a3042"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLOR_NORMAL = "#06d6a0"
COLOR_FRAUD = "#ef476f"
COLOR_ACCENT = "#118ab2"
COLOR_AMBER = "#ffd166"

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
    st.markdown('<div class="main-header" style="font-size:1.6rem;">Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.85rem;margin-top:-8px;">ML-Powered Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.metric("Transactions", f"{len(df):,}")
        st.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
        st.metric("Fraud Rate", f"{df['Class'].mean():.2%}")
    st.markdown("---")
    st.markdown("""
    <div style="padding:12px;background:rgba(6,214,160,0.06);border:1px solid rgba(6,214,160,0.15);border-radius:10px;">
        <p style="color:#06d6a0;font-weight:600;font-size:0.8rem;margin:0 0 8px 0;text-transform:uppercase;letter-spacing:0.05em;">Trained Models</p>
        <p style="color:#e8eaed;font-size:0.85rem;margin:0;line-height:1.7;">
        Isolation Forest<br>
        Local Outlier Factor<br>
        One-Class SVM<br>
        Autoencoder
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
    <div style="padding:10px;text-align:center;">
        <p style="color:#4b5563;font-size:0.72rem;margin:0;">Built with Streamlit + scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Real-Time Financial Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Detect anomalous transactions across 4 ML models with explainable AI</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  Dataset  ",
    "  Detection  ",
    "  Comparison  ",
    "  Visualization  ",
    "  Explainability  ",
    "  Simulation  ",
])

# ===================== Tab 1: Dataset Explorer =====================
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
            class_counts = df["Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Label"] = class_counts["Class"].map({0: "Normal", 1: "Fraud"})
            fig = px.bar(
                class_counts, x="Label", y="Count", color="Label",
                color_discrete_map={"Normal": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
                text="Count",
            )
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=350)
            fig.update_traces(textposition="outside", textfont=dict(color="#e8eaed"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Feature Distribution</div>', unsafe_allow_html=True)
            feature = st.selectbox(
                "Select feature",
                ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)],
            )
            fig = px.histogram(
                df, x=feature, color="Class", nbins=50,
                color_discrete_map={0: COLOR_NORMAL, 1: COLOR_FRAUD},
                barmode="overlay", opacity=0.7,
                labels={"Class": "Label"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 11)]
        corr = df[corr_features].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Tealrose",
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 2: Anomaly Detection =====================
with tab2:
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)

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

        # Confusion matrix visualization
        st.markdown("---")
        cm_c1, cm_c2 = st.columns([1, 2])
        with cm_c1:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = np.array([[tn, fp], [fn, tp]])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Predicted Normal", "Predicted Fraud"],
                y=["Actual Normal", "Actual Fraud"],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=16, color="#e8eaed"),
                colorscale=[[0, "#1a1f2e"], [1, "#ef476f"]],
                showscale=False,
            ))
            fig_cm.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=20, r=20, t=20, b=20)}, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_c2:
            st.markdown('<div class="section-header">Flagged Transactions</div>', unsafe_allow_html=True)
            flagged_idx = np.where(y_pred == 1)[0]
            if len(flagged_idx) > 0:
                flagged_df = pd.DataFrame({
                    "Index": flagged_idx,
                    "Anomaly Score": scores[flagged_idx].round(4),
                    "Actual": ["Fraud" if y_test[i] == 1 else "Normal" for i in flagged_idx],
                })
                st.dataframe(flagged_df.head(50), use_container_width=True, hide_index=True)
            else:
                st.info("No transactions flagged at this threshold.")

# ===================== Tab 3: Model Comparison =====================
with tab3:
    st.markdown('<div class="section-header">Model Performance Leaderboard</div>', unsafe_allow_html=True)

    if st.session_state.all_metrics:
        metrics = st.session_state.all_metrics

        leaderboard = create_leaderboard(metrics)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_roc_curves(metrics)
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = plot_pr_curves(metrics)
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig, use_container_width=True)

        fig = plot_metric_bars(metrics)
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 4: Visualization =====================
with tab4:
    st.markdown('<div class="section-header">Anomaly Visualization</div>', unsafe_allow_html=True)

    if st.session_state.model_results and st.session_state.preprocessed:
        viz_model = st.selectbox("Select Model", MODEL_NAMES, key="viz_model")
        scores = st.session_state.model_results[viz_model]["scores"]
        y_test = st.session_state.preprocessed["y_test"]
        X_test = st.session_state.preprocessed["X_test"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">PCA Projection</div>', unsafe_allow_html=True)
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
                color_discrete_map={"Normal": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
                opacity=0.6, hover_data=["Score"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
            score_df = pd.DataFrame({
                "Score": scores,
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
            })
            fig = px.histogram(
                score_df, x="Score", color="Label", nbins=50,
                color_discrete_map={"Normal": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
                barmode="overlay", opacity=0.7,
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Transaction Timeline</div>', unsafe_allow_html=True)
        timeline_df = pd.DataFrame({
            "Index": np.arange(len(scores)),
            "Anomaly Score": scores,
            "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
        })
        fig = px.scatter(
            timeline_df, x="Index", y="Anomaly Score", color="Label",
            color_discrete_map={"Normal": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
            opacity=0.5,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR_AMBER,
                      annotation_text="Threshold", annotation_font_color=COLOR_AMBER)
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

# ===================== Tab 5: Explainability =====================
with tab5:
    st.markdown('<div class="section-header">Model Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.9rem;">SHAP (SHapley Additive exPlanations) breaks down each prediction to show which features pushed the model toward "fraud" or "normal."</p>', unsafe_allow_html=True)

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

        if st.button("Compute SHAP Values", type="primary"):
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

            st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feat_names[:len(mean_abs)],
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(
                importance_df, x="Mean |SHAP|", y="Feature",
                orientation="h", color="Mean |SHAP|",
                color_continuous_scale=[[0, "#1a1f2e"], [0.5, "#ef476f"], [1, "#ffd166"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Individual Transaction Explanation</div>', unsafe_allow_html=True)
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
                    color_continuous_scale=[[0, "#118ab2"], [0.5, "#1a1f2e"], [1, "#ef476f"]],
                    color_continuous_midpoint=0,
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=400, title=f"Why Transaction {selected} Was Flagged", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top contributing features:**")
                top = txn_df.tail(3).iloc[::-1]
                for _, row in top.iterrows():
                    direction = "increases" if row["SHAP Value"] > 0 else "decreases"
                    color = COLOR_FRAUD if row["SHAP Value"] > 0 else COLOR_NORMAL
                    st.markdown(
                        f'<div style="padding:6px 12px;margin:4px 0;border-left:3px solid {color};'
                        f'background:rgba(26,31,46,0.6);border-radius:0 6px 6px 0;">'
                        f'<span style="color:{color};font-weight:600;">{row["Feature"]}</span> '
                        f'<span style="color:#9ca3af;">{direction} fraud risk</span> '
                        f'<span style="color:#6b7280;font-family:JetBrains Mono,monospace;font-size:0.85rem;">'
                        f'(SHAP: {row["SHAP Value"]:+.4f})</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No flagged transactions in the explained sample.")

# ===================== Tab 6: Real-Time Simulation =====================
with tab6:
    st.markdown('<div class="section-header">Real-Time Transaction Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;font-size:0.9rem;">Simulate a live transaction feed and watch the model flag anomalies in real time.</p>', unsafe_allow_html=True)

    if st.session_state.model_results and st.session_state.preprocessed:
        sim_model = st.selectbox("Select Model", MODEL_NAMES, key="sim_model")

        c1, c2 = st.columns([1, 3])
        with c1:
            n_transactions = st.number_input("Transactions to simulate", 20, 200, 50)
            speed = st.slider("Speed (ms between transactions)", 50, 500, 100)

        if st.button("Start Simulation", type="primary"):
            X_test = st.session_state.preprocessed["X_test"]
            y_test = st.session_state.preprocessed["y_test"]
            model_result = st.session_state.model_results[sim_model]

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
                    actual_label = "FRAUD" if label == 1 else "FALSE ALARM"
                    alerts.append(
                        f'<div class="alert-fraud">TXN #{i+1:03d} &nbsp; '
                        f'score={score:.3f} &nbsp; [{actual_label}]</div>'
                    )

                # Update chart
                sim_df = pd.DataFrame({
                    "Transaction": range(1, len(scores_history) + 1),
                    "Anomaly Score": scores_history,
                    "Label": ["Fraud" if l == 1 else "Normal" for l in labels_history],
                })
                fig = px.scatter(
                    sim_df, x="Transaction", y="Anomaly Score",
                    color="Label",
                    color_discrete_map={"Normal": COLOR_NORMAL, "Fraud": COLOR_FRAUD},
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR_AMBER,
                              annotation_text="Threshold", annotation_font_color=COLOR_AMBER)
                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                fig.update_traces(marker=dict(size=8))
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
                        st.markdown('<div class="section-header">Recent Alerts</div>', unsafe_allow_html=True)
                        st.markdown("".join(alerts[-5:]), unsafe_allow_html=True)

                time.sleep(speed / 1000)

            st.success(f"Simulation complete: {n_transactions} transactions processed.")


# ---------------------------------------------------------------------------
# Glossary — Key Terms Explained
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="glossary-container">
    <div class="glossary-title">Key Terms Explained</div>
    <table style="width:100%;border-collapse:separate;border-spacing:0 12px;">
        <tr>
            <td style="width:180px;vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Precision</span>
            </td>
            <td class="glossary-def">
                Of all transactions the model <em>flagged</em> as fraud, what fraction were actually fraud?
                High precision = fewer false alarms. A precision of 0.90 means 90% of flagged transactions are real fraud.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Recall</span>
            </td>
            <td class="glossary-def">
                Of all <em>actual</em> fraud cases, what fraction did the model catch?
                High recall = fewer missed frauds. A recall of 0.80 means the model catches 80% of all real fraud.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">F1 Score</span>
            </td>
            <td class="glossary-def">
                The harmonic mean of precision and recall — a single number that balances both.
                F1 = 2 &times; (Precision &times; Recall) / (Precision + Recall). Ranges from 0 (worst) to 1 (perfect). Useful when you care about <em>both</em> catching fraud and avoiding false alarms.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">ROC-AUC</span>
            </td>
            <td class="glossary-def">
                Area Under the Receiver Operating Characteristic curve. Measures how well the model ranks fraud above normal transactions <em>across all thresholds</em>.
                1.0 = perfect separation, 0.5 = random guessing. A model with AUC 0.95 ranks a random fraud higher than a random normal transaction 95% of the time.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Avg Precision</span>
            </td>
            <td class="glossary-def">
                Area under the Precision-Recall curve. Especially useful for imbalanced datasets (like fraud detection where <1% are fraud) because it focuses on how well the model performs on the rare positive class.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">SHAP Values</span>
            </td>
            <td class="glossary-def">
                SHapley Additive exPlanations — a game-theory approach to explain individual predictions.
                Each feature gets a SHAP value showing how much it pushed the prediction toward fraud (positive) or normal (negative). Allows you to answer: <em>"Why was this specific transaction flagged?"</em>
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Anomaly Score</span>
            </td>
            <td class="glossary-def">
                A normalized score (0 to 1) assigned to each transaction. Higher = more anomalous.
                The detection threshold (default 0.5) determines the cutoff: transactions scoring above are flagged as potential fraud.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Confusion Matrix</span>
            </td>
            <td class="glossary-def">
                A 2&times;2 table showing: True Positives (correctly caught fraud), False Positives (false alarms),
                True Negatives (correctly cleared normal), and False Negatives (missed fraud). Gives a complete picture of model errors.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Isolation Forest</span>
            </td>
            <td class="glossary-def">
                An ensemble algorithm that isolates anomalies by randomly partitioning features. Fraud transactions are "isolated" in fewer splits because they are rare and different.
                Fast, scalable, and effective for high-dimensional data.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">LOF</span>
            </td>
            <td class="glossary-def">
                Local Outlier Factor — measures how isolated a point is relative to its neighbors.
                A transaction in a sparse region (far from neighbors) gets a high LOF score. Good at detecting local anomalies that global methods miss.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">One-Class SVM</span>
            </td>
            <td class="glossary-def">
                A Support Vector Machine trained on <em>only</em> normal data. It learns a boundary around normal behavior;
                anything falling outside is flagged as an anomaly. Works well when fraud examples are scarce or unavailable during training.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">Autoencoder</span>
            </td>
            <td class="glossary-def">
                A neural network trained to compress and reconstruct normal transactions. Fraud transactions reconstruct poorly (high reconstruction error) because the network never saw them during training. The reconstruction error <em>is</em> the anomaly score.
            </td>
        </tr>
        <tr>
            <td style="vertical-align:top;padding-right:16px;">
                <span class="glossary-term">PCA</span>
            </td>
            <td class="glossary-def">
                Principal Component Analysis — reduces high-dimensional data (30 features) to 2 dimensions for visualization, preserving as much variance as possible. In the Visualization tab, each dot is a transaction plotted in PCA space.
            </td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0;">
    <p style="color:#4b5563;font-size:0.78rem;">
        Built by Sanjit Mathur &nbsp;&bull;&nbsp; scikit-learn + FastAPI + Streamlit + SHAP &nbsp;&bull;&nbsp; Credit Card Fraud Dataset (Kaggle)
    </p>
</div>
""", unsafe_allow_html=True)
