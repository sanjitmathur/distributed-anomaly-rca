"""Financial Fraud Detection Platform — Premium 5-tab Streamlit dashboard."""

import sys
import os
import json
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
from monitoring.feedback import FeedbackStore
from monitoring.tracker import MonitoringTracker
from risk.scoring import RiskScorer
from risk.graph import EntityRiskGraph
from utils.config import ALL_FEATURES, MODEL_NAMES, SAMPLE_CSV

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>S</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium design system — Dark Command Center
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ── Keyframes ── */
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 8px rgba(0, 255, 135, 0.05); }
        50% { box-shadow: 0 0 20px rgba(0, 255, 135, 0.12); }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes border-flow {
        0% { border-color: rgba(0, 255, 135, 0.15); }
        33% { border-color: rgba(0, 210, 255, 0.15); }
        66% { border-color: rgba(255, 107, 107, 0.15); }
        100% { border-color: rgba(0, 255, 135, 0.15); }
    }

    :root {
        --bg-void: #030507;
        --bg-primary: #060a10;
        --bg-secondary: #0a0f18;
        --bg-card: #0d1420;
        --bg-card-hover: #111a2a;
        --bg-elevated: #141e30;
        --accent: #00ff87;
        --accent-dim: #00cc6a;
        --accent-glow: rgba(0, 255, 135, 0.08);
        --accent-cyan: #00d2ff;
        --accent-cyan-glow: rgba(0, 210, 255, 0.08);
        --danger: #ff6b6b;
        --danger-glow: rgba(255, 107, 107, 0.08);
        --warning: #ffc857;
        --info: #00d2ff;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #484f58;
        --border: rgba(139, 148, 158, 0.06);
        --border-hover: rgba(0, 255, 135, 0.2);
        --font-display: 'Sora', sans-serif;
        --font-body: 'DM Sans', sans-serif;
        --font-mono: 'IBM Plex Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* ── App background with noise grain ── */
    .stApp {
        background: var(--bg-void);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% 0%, rgba(0, 255, 135, 0.03), transparent),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0, 210, 255, 0.02), transparent);
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-void) 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
        font-family: var(--font-body);
    }

    /* ── Hero ── */
    .hero-container {
        position: relative;
        padding: 8px 0 0 0;
        margin-bottom: 8px;
    }
    .hero-scanline {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        opacity: 0.4;
        animation: scanline 4s linear infinite;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        background: var(--accent-glow);
        border: 1px solid rgba(0, 255, 135, 0.12);
        border-radius: 4px;
        color: var(--accent);
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 16px;
        animation: fadeInUp 0.6s ease-out;
    }
    .hero-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .hero-title {
        font-family: var(--font-display);
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 10px;
        animation: fadeInUp 0.6s ease-out 0.1s both;
    }
    .hero-title span {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-family: var(--font-body);
        font-size: 0.92rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 620px;
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }

    /* ── KPI row ── */
    .kpi-row {
        display: flex;
        gap: 10px;
        margin: 24px 0 16px 0;
        animation: fadeInUp 0.6s ease-out 0.3s both;
    }
    .kpi-card {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-dim), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .kpi-card:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
        transform: translateY(-3px);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(0, 255, 135, 0.04);
    }
    .kpi-card:hover::after { opacity: 1; }
    .kpi-label {
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.62rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .kpi-value.success { color: var(--accent); text-shadow: 0 0 20px rgba(0,255,135,0.2); }
    .kpi-value.danger { color: var(--danger); text-shadow: 0 0 20px rgba(255,107,107,0.2); }
    .kpi-value.warning { color: var(--warning); }
    .kpi-value.accent { color: var(--accent-cyan); text-shadow: 0 0 20px rgba(0,210,255,0.2); }

    /* ── Streamlit metric cards ── */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: border-flow 8s ease infinite;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-hover);
        box-shadow: 0 0 24px rgba(0, 255, 135, 0.04);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.62rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 1.45rem;
        font-weight: 700;
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 6px;
        padding: 3px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-weight: 500;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        padding: 10px 22px;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent);
        background: rgba(0, 255, 135, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 135, 0.1) !important;
        color: var(--accent) !important;
        font-weight: 600;
        box-shadow: inset 0 -2px 0 var(--accent);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: var(--font-display);
        color: var(--text-primary);
        font-size: 1.15rem;
        font-weight: 700;
        margin: 2rem 0 0.5rem 0;
        letter-spacing: -0.02em;
        position: relative;
        padding-left: 14px;
    }
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 3px;
        bottom: 3px;
        width: 3px;
        background: var(--accent);
        border-radius: 2px;
        box-shadow: 0 0 8px rgba(0, 255, 135, 0.3);
    }
    .section-desc {
        color: var(--text-muted);
        font-size: 0.83rem;
        margin-top: -2px;
        margin-bottom: 16px;
        line-height: 1.6;
        padding-left: 14px;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(13,20,32,0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 24px;
        margin: 8px 0;
    }

    /* ── Status pills ── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 12px;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pill-fraud {
        background: rgba(255, 107, 107, 0.1);
        color: var(--danger);
        border: 1px solid rgba(255, 107, 107, 0.15);
    }
    .pill-normal {
        background: rgba(0, 255, 135, 0.08);
        color: var(--accent);
        border: 1px solid rgba(0, 255, 135, 0.12);
    }
    .pill-drift {
        background: rgba(255, 200, 87, 0.1);
        color: var(--warning);
        border: 1px solid rgba(255, 200, 87, 0.15);
    }

    /* ── Alert / insight boxes ── */
    .alert-fraud {
        background: rgba(255, 107, 107, 0.04);
        border-left: 2px solid var(--danger);
        border-radius: 0 6px 6px 0;
        padding: 10px 16px;
        margin: 4px 0;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }
    .insight-card {
        background: rgba(0, 255, 135, 0.03);
        border-left: 2px solid var(--accent-dim);
        border-radius: 0 6px 6px 0;
        padding: 12px 18px;
        margin: 8px 0;
        color: var(--text-secondary);
        font-size: 0.84rem;
        line-height: 1.6;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: var(--bg-void);
        font-family: var(--font-mono);
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        border: none;
        border-radius: 6px;
        padding: 0.65rem 1.8rem;
        transition: all 0.25s;
        text-transform: uppercase;
    }
    .stButton > button[kind="primary"]:hover {
        background: #33ff9f;
        box-shadow: 0 0 30px rgba(0, 255, 135, 0.2), 0 4px 16px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    .stButton > button {
        border-radius: 6px;
        font-family: var(--font-body);
        font-weight: 500;
        transition: all 0.2s;
    }

    /* ── Form elements ── */
    .stSelectbox > div > div {
        border-radius: 6px;
        border-color: var(--border);
        font-family: var(--font-mono);
    }
    .stSlider > div > div > div { color: var(--accent); }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 6px;
        font-family: var(--font-body);
        font-weight: 500;
    }

    /* ── Dividers ── */
    hr {
        border: none !important;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), rgba(0,255,135,0.06), var(--border), transparent) !important;
        margin: 2rem 0;
    }

    /* ── Footer ── */
    footer { visibility: hidden; }

    /* ── Sidebar stat blocks ── */
    .sidebar-stat {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 5px 0;
        transition: border-color 0.3s;
    }
    .sidebar-stat:hover { border-color: var(--border-hover); }
    .sidebar-stat-label {
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.6rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .sidebar-stat-value {
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 3px;
    }

    /* ── Feature callout ── */
    .feature-callout {
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        background: var(--bg-card);
        font-family: var(--font-body);
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 72px 24px;
        color: var(--text-muted);
    }
    .empty-state-icon {
        font-size: 2rem;
        margin-bottom: 16px;
        opacity: 0.3;
    }
    .empty-state-title {
        font-family: var(--font-display);
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .empty-state-desc {
        font-size: 0.83rem;
        max-width: 360px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-void); }
    ::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(6,10,16,0.8)",
    font=dict(family="DM Sans, sans-serif", color="#8b949e", size=12),
    title_font=dict(color="#e6edf3", size=14, family="Sora, sans-serif"),
    xaxis=dict(gridcolor="rgba(139,148,158,0.05)", zerolinecolor="rgba(139,148,158,0.05)"),
    yaxis=dict(gridcolor="rgba(139,148,158,0.05)", zerolinecolor="rgba(139,148,158,0.05)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", family="IBM Plex Mono, monospace", size=11)),
    margin=dict(l=40, r=20, t=50, b=40),
)
C_SUCCESS = "#00ff87"
C_DANGER = "#ff6b6b"
C_WARNING = "#ffc857"
C_ACCENT = "#00d2ff"
C_INFO = "#00d2ff"
C_SECONDARY = "#00ff87"

# ---------------------------------------------------------------------------
# Session state
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

    progress = st.empty()
    status = st.empty()

    with progress.container():
        bar = st.progress(0, text="Initializing platform...")

        status.markdown(
            '<p style="color:#94a3b8;font-size:0.85rem;text-align:center;">'
            'Loading dataset and training models. This runs once on first visit.</p>',
            unsafe_allow_html=True,
        )

        bar.progress(10, text="Loading dataset...")
        df = load_data()
        st.session_state.df_raw = df

        bar.progress(25, text="Engineering features...")
        fe = FeatureEngineer()
        df_eng = fe.transform(df.drop(columns=["Class"]))
        df_eng["Class"] = df["Class"].values
        st.session_state.df_engineered = df_eng

        bar.progress(40, text="Preprocessing & scaling...")
        result = preprocess(df_eng)
        st.session_state.preprocessed = result

        bar.progress(55, text="Training 4 ML models...")
        model_results = train_all_models(
            result["X_train"], result["X_test"], result["y_test"], save=True,
        )
        st.session_state.model_results = model_results

        bar.progress(85, text="Evaluating models...")
        all_metrics = {}
        for name, res in model_results.items():
            all_metrics[name] = compute_metrics(result["y_test"], res["scores"])
        st.session_state.all_metrics = all_metrics

        bar.progress(100, text="Ready.")
        time.sleep(0.3)

    progress.empty()
    status.empty()
    st.session_state.initialized = True


_initialize()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 12px 0;">'
        '<div style="font-family:Sora,sans-serif;font-size:1.2rem;font-weight:800;color:#e6edf3;'
        'letter-spacing:-0.04em;">FRAUD<span style="color:#00ff87;">CMD</span></div>'
        '<div style="color:#484f58;font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        'margin-top:4px;letter-spacing:0.1em;text-transform:uppercase;">Anomaly Detection System</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        fraud_count = int(df["Class"].sum())
        fraud_rate = df["Class"].mean()

        for label, value, css_class in [
            ("Transactions", f"{len(df):,}", ""),
            ("Fraud Cases", f"{fraud_count:,}", "danger"),
            ("Fraud Rate", f"{fraud_rate:.2%}", "warning"),
        ]:
            st.markdown(
                f'<div class="sidebar-stat">'
                f'<div class="sidebar-stat-label">{label}</div>'
                f'<div class="sidebar-stat-value {css_class}">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Best model highlight
    if st.session_state.all_metrics:
        best = max(st.session_state.all_metrics.items(), key=lambda x: x[1]["f1"])
        st.markdown(
            f'<div style="padding:14px 16px;background:rgba(0,255,135,0.04);'
            f'border:1px solid rgba(0,255,135,0.1);border-radius:6px;">'
            f'<div style="color:#00ff87;font-family:IBM Plex Mono,monospace;font-size:0.6rem;'
            f'font-weight:600;text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;">'
            f'Top Performer</div>'
            f'<div style="color:#e6edf3;font-family:Sora,sans-serif;font-size:0.95rem;font-weight:700;'
            f'letter-spacing:-0.02em;">'
            f'{best[0].replace("_", " ").title()}</div>'
            f'<div style="color:#484f58;font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-top:6px;">'
            f'F1 <span style="color:#00ff87;">{best[1]["f1"]:.3f}</span>'
            f' &middot; AUC <span style="color:#00d2ff;">{best[1]["roc_auc"]:.3f}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div style="padding:8px;text-align:center;">'
        '<p style="color:#484f58;font-family:IBM Plex Mono,monospace;font-size:0.6rem;'
        'letter-spacing:0.08em;margin:0;">SCIKIT-LEARN // FASTAPI // STREAMLIT</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-container">'
    '<div class="hero-scanline"></div>'
    '<div class="hero-badge">System Online &mdash; Real-Time ML Platform</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Financial <span>Fraud Detection</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    'Streaming anomaly detection across 4 ML models with SHAP explainability, '
    'concept drift monitoring, and entity-graph fraud ring analysis.'
    '</div>',
    unsafe_allow_html=True,
)

# Hero KPI row
if st.session_state.all_metrics and st.session_state.preprocessed:
    best_m = max(st.session_state.all_metrics.items(), key=lambda x: x[1]["f1"])
    y_test = st.session_state.preprocessed["y_test"]
    n_models = len(st.session_state.all_metrics)
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-card"><div class="kpi-label">Models Active</div>'
        f'<div class="kpi-value accent">{n_models}</div></div>'
        f'<div class="kpi-card"><div class="kpi-label">Best F1</div>'
        f'<div class="kpi-value success">{best_m[1]["f1"]:.3f}</div></div>'
        f'<div class="kpi-card"><div class="kpi-label">ROC-AUC</div>'
        f'<div class="kpi-value" style="color:#e6edf3;">{best_m[1]["roc_auc"]:.3f}</div></div>'
        f'<div class="kpi-card"><div class="kpi-label">Test Samples</div>'
        f'<div class="kpi-value" style="color:#e6edf3;">{len(y_test):,}</div></div>'
        f'<div class="kpi-card"><div class="kpi-label">Fraud Detected</div>'
        f'<div class="kpi-value danger">{int(y_test.sum())}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs (5 consolidated)
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Analysis  ",
    "  Detection  ",
    "  Explainability  ",
    "  Live Operations  ",
    "  Monitoring  ",
])

# ===================== Tab 1: Analysis (Dataset + Visualization) =====================
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    df = st.session_state.df_raw

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("Features", f"{len(df.columns) - 1}")
        c3.metric("Fraud Count", f"{int(df['Class'].sum()):,}")
        c4.metric("Fraud Ratio", f"{df['Class'].mean():.2%}")

        st.markdown(
            '<div class="insight-card">'
            'The dataset exhibits extreme class imbalance (&lt;0.2% fraud). '
            'Models are trained on normal transactions only (unsupervised paradigm) — '
            'fraud patterns are detected as deviations from learned normalcy.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
            class_counts = df["Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Label"] = class_counts["Class"].map({0: "Normal", 1: "Fraud"})
            fig = px.bar(
                class_counts, x="Label", y="Count", color="Label",
                color_discrete_map={"Normal": C_SUCCESS, "Fraud": C_DANGER},
                text="Count",
            )
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=350)
            fig.update_traces(textposition="outside", textfont=dict(color="#f1f5f9"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Feature Distribution</div>', unsafe_allow_html=True)
            feature = st.selectbox(
                "Select feature",
                ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)],
            )
            fig = px.histogram(
                df, x=feature, color="Class", nbins=50,
                color_discrete_map={0: C_SUCCESS, 1: C_DANGER},
                barmode="overlay", opacity=0.7,
                labels={"Class": "Label"},
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Top 10 PCA components + Amount + Time. V14 and V17 show the strongest anti-correlation with fraud.</div>', unsafe_allow_html=True)
        corr_features = ["Amount", "Time"] + [f"V{i}" for i in range(1, 11)]
        corr = df[corr_features].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=[[0, "#0a0f18"], [0.35, "#0d1420"], [0.65, "#00664a"], [1, "#00ff87"]],
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig, use_container_width=True)

        # PCA + Score Distribution section
        if st.session_state.model_results and st.session_state.preprocessed:
            st.markdown("---")
            st.markdown('<div class="section-header">Anomaly Landscape</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-desc">Visual separation between normal and fraudulent transactions in reduced dimensionality space.</div>', unsafe_allow_html=True)

            viz_model = st.selectbox("Select Model", MODEL_NAMES, key="viz_model")
            scores = st.session_state.model_results[viz_model]["scores"]
            y_test = st.session_state.preprocessed["y_test"]
            X_test = st.session_state.preprocessed["X_test"]

            c1, c2 = st.columns(2)

            with c1:
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
                    color_discrete_map={"Normal": C_SUCCESS, "Fraud": C_DANGER},
                    opacity=0.6, hover_data=["Score"],
                    title="PCA Projection",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=420)
                fig.update_traces(marker=dict(size=5))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                score_df = pd.DataFrame({
                    "Score": scores,
                    "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
                })
                fig = px.histogram(
                    score_df, x="Score", color="Label", nbins=50,
                    color_discrete_map={"Normal": C_SUCCESS, "Fraud": C_DANGER},
                    barmode="overlay", opacity=0.7,
                    title="Anomaly Score Distribution",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=420)
                st.plotly_chart(fig, use_container_width=True)

            # Transaction timeline
            timeline_df = pd.DataFrame({
                "Index": np.arange(len(scores)),
                "Anomaly Score": scores,
                "Label": ["Fraud" if y == 1 else "Normal" for y in y_test],
            })
            fig = px.scatter(
                timeline_df, x="Index", y="Anomaly Score", color="Label",
                color_discrete_map={"Normal": C_SUCCESS, "Fraud": C_DANGER},
                opacity=0.5, title="Transaction Timeline",
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color=C_WARNING,
                          annotation_text="Threshold", annotation_font_color=C_WARNING)
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)


# ===================== Tab 2: Detection (Anomaly + Model Comparison) =====================
with tab2:
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Tune the detection threshold and compare model performance across all metrics.</div>', unsafe_allow_html=True)

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
        cm_c1, cm_c2 = st.columns([1, 2])
        with cm_c1:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = np.array([[tn, fp], [fn, tp]])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred Normal", "Pred Fraud"],
                y=["Actual Normal", "Actual Fraud"],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=16, color="#e6edf3", family="IBM Plex Mono"),
                colorscale=[[0, "#0a0f18"], [1, "#ff6b6b"]],
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

        # Model Comparison section
        st.markdown("---")
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Head-to-head performance of all 4 anomaly detection models.</div>', unsafe_allow_html=True)

    if st.session_state.all_metrics:
        metrics = st.session_state.all_metrics
        leaderboard = create_leaderboard(metrics)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_roc_curves(metrics)
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = plot_pr_curves(metrics)
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig, use_container_width=True)

        fig = plot_metric_bars(metrics)
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)


# ===================== Tab 3: Explainability =====================
with tab3:
    st.markdown('<div class="section-header">Model Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">'
        'SHAP (SHapley Additive exPlanations) decomposes each prediction to show which features '
        'pushed the model toward fraud or normal. This enables auditable, trustworthy ML decisions.'
        '</div>',
        unsafe_allow_html=True,
    )

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
            with st.spinner("Computing SHAP values..."):
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
            st.markdown('<div class="section-desc">Features ranked by average absolute SHAP value across all explained samples.</div>', unsafe_allow_html=True)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feat_names[:len(mean_abs)],
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(
                importance_df, x="Mean |SHAP|", y="Feature",
                orientation="h", color="Mean |SHAP|",
                color_continuous_scale=[[0, "#0a0f18"], [0.4, "#00664a"], [0.7, "#00ff87"], [1, "#ffc857"]],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=False, coloraxis_showscale=False)
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
                    color_continuous_scale=[[0, "#00d2ff"], [0.5, "#0a0f18"], [1, "#ff6b6b"]],
                    color_continuous_midpoint=0,
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=380,
                    title=f"Why Transaction {selected} Was Flagged",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                top = txn_df.tail(3).iloc[::-1]
                for _, row in top.iterrows():
                    direction = "increases" if row["SHAP Value"] > 0 else "decreases"
                    color = C_DANGER if row["SHAP Value"] > 0 else C_SUCCESS
                    st.markdown(
                        f'<div class="feature-callout" style="border-left:3px solid {color};">'
                        f'<span style="color:{color};font-weight:600;">{row["Feature"]}</span> '
                        f'<span style="color:#94a3b8;">{direction} fraud risk</span> '
                        f'<span style="color:#64748b;font-family:JetBrains Mono,monospace;font-size:0.82rem;">'
                        f'(SHAP: {row["SHAP Value"]:+.4f})</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No flagged transactions in the explained sample.")
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#9881;</div>'
            '<div class="empty-state-title">Models not yet trained</div>'
            '<div class="empty-state-desc">SHAP explanations require trained models. '
            'Return to the Analysis tab to initialize.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ===================== Tab 4: Live Operations (Stream + Investigation + Graph) =====================
with tab4:
    st.markdown('<div class="section-header">Live Operations Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">'
        'Real-time inference pipeline with concept drift detection, fraud investigation, '
        'and entity-graph fraud ring analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    ops_section = st.radio(
        "Section",
        ["Streaming Monitor", "Investigation Console", "Entity Risk Graph"],
        horizontal=True,
        key="ops_section",
    )

    # ---- Streaming Monitor ----
    if ops_section == "Streaming Monitor":
        stream_mode = st.radio(
            "Stream source",
            ["In-Process (local queue)", "Remote (inference service SSE)"],
            index=0,
            horizontal=True,
            key="stream_mode",
            help="Use In-Process for local runs. Remote requires Docker inference service.",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            n_transactions = st.number_input("Transactions to stream", 50, 2000, 200, key="stream_n")
        with c2:
            tps = st.slider("Transactions per second", 1, 50, 10, key="stream_tps")
        with c3:
            fraud_rate = st.slider("Fraud injection rate", 0.0, 0.10, 0.005, 0.001,
                                   format="%.3f", key="stream_fraud")

        if st.button("Start Stream", type="primary", key="stream_start"):

            chart_ph = st.empty()
            metrics_ph = st.empty()
            drift_ph = st.empty()
            alert_ph = st.empty()
            feature_ph = st.empty()

            scores_hist = []
            amounts_hist = []
            drift_hist = []
            alerts = []
            window_feats_hist = []
            graph_feats_hist = []

            if "In-Process" in stream_mode:
                from queue import Queue, Empty
                from streaming.producer import create_producer_thread, generate_transaction
                from streaming.consumer import StreamingConsumer

                try:
                    consumer = StreamingConsumer(model_name="isolation_forest")
                except Exception as e:
                    st.error(f"Cannot start consumer (train models first): {e}")
                    st.stop()

                q = Queue(maxsize=500)
                producer_thread = create_producer_thread(
                    q, tps=tps, fraud_rate=fraud_rate, max_txns=n_transactions,
                )
                producer_thread.start()

                for i in range(n_transactions):
                    try:
                        raw = q.get(timeout=5.0)
                    except Empty:
                        continue
                    if raw is None:
                        break
                    try:
                        scored = consumer.process(raw)
                    except Exception as e:
                        st.warning(f"Skipped malformed transaction: {e}")
                        continue

                    scores_hist.append(scored.fraud_probability)
                    amounts_hist.append(scored.Amount)
                    drift_hist.append(scored.drift_detected)
                    window_feats_hist.append(scored.window_features)
                    graph_feats_hist.append(scored.graph_features)

                    if scored.is_fraud:
                        alerts.append(
                            f'<div class="alert-fraud">{scored.transaction_id} &nbsp; '
                            f'score={scored.fraud_probability:.3f} &nbsp; '
                            f'amt=${scored.Amount:,.2f}</div>'
                        )

                    if (i + 1) % 5 == 0 or i == n_transactions - 1:
                        sim_df = pd.DataFrame({
                            "Transaction": range(1, len(scores_hist) + 1),
                            "Anomaly Score": scores_hist,
                            "Drift": ["Drift" if d else "Stable" for d in drift_hist],
                        })
                        fig = px.scatter(
                            sim_df, x="Transaction", y="Anomaly Score",
                            color="Drift",
                            color_discrete_map={"Stable": C_SUCCESS, "Drift": C_WARNING},
                        )
                        fig.add_hline(
                            y=0.5, line_dash="dash", line_color=C_DANGER,
                            annotation_text="Threshold", annotation_font_color=C_DANGER,
                        )
                        fig.update_layout(**PLOTLY_LAYOUT, height=350)
                        fig.update_traces(marker=dict(size=6))
                        chart_ph.plotly_chart(fig, use_container_width=True)

                        flagged = sum(1 for s in scores_hist if s >= 0.5)
                        drift_count = sum(drift_hist)
                        stats = consumer.stats
                        with metrics_ph.container():
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Processed", stats["total_processed"])
                            m2.metric("Flagged", flagged)
                            m3.metric("Fraud Rate", f"{stats['fraud_rate']:.2%}")
                            m4.metric("Drift Alerts", drift_count)
                            m5.metric("Avg Score", f"{np.mean(scores_hist):.3f}")

                        if window_feats_hist:
                            latest_w = window_feats_hist[-1]
                            latest_g = graph_feats_hist[-1]
                            with feature_ph.container():
                                fc1, fc2 = st.columns(2)
                                with fc1:
                                    st.markdown("**Sliding Window (10 min)**")
                                    for k, v in latest_w.items():
                                        st.text(f"  {k}: {v}")
                                with fc2:
                                    st.markdown("**Graph Features**")
                                    for k, v in latest_g.items():
                                        st.text(f"  {k}: {v}")

                        if alerts:
                            with alert_ph.container():
                                st.markdown("".join(alerts[-8:]), unsafe_allow_html=True)

                st.success(f"Stream complete: {len(scores_hist)} transactions processed.")

            else:
                import httpx

                inference_url = os.getenv("INFERENCE_URL", "http://localhost:8000")
                st.info(f"Connecting to {inference_url}/stream ...")

                try:
                    with httpx.stream("GET", f"{inference_url}/stream", timeout=None) as resp:
                        count = 0
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = json.loads(line[6:])
                            if data.get("type") in ("connected", "heartbeat"):
                                continue

                            scores_hist.append(data.get("fraud_probability", 0))
                            amounts_hist.append(data.get("Amount", 0))
                            drift_hist.append(data.get("drift_detected", False))
                            window_feats_hist.append(data.get("window_features", {}))
                            graph_feats_hist.append(data.get("graph_features", {}))

                            if data.get("is_fraud"):
                                alerts.append(
                                    f'<div class="alert-fraud">{data["transaction_id"]} '
                                    f'score={data["fraud_probability"]:.3f}</div>'
                                )

                            count += 1
                            if count % 5 == 0 or count >= n_transactions:
                                sim_df = pd.DataFrame({
                                    "Transaction": range(1, len(scores_hist) + 1),
                                    "Anomaly Score": scores_hist,
                                })
                                fig = px.scatter(sim_df, x="Transaction", y="Anomaly Score")
                                fig.add_hline(y=0.5, line_dash="dash", line_color=C_DANGER)
                                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                                chart_ph.plotly_chart(fig, use_container_width=True)

                                with metrics_ph.container():
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Received", count)
                                    m2.metric("Flagged", sum(1 for s in scores_hist if s >= 0.5))
                                    m3.metric("Drift", sum(drift_hist))

                                if alerts:
                                    with alert_ph.container():
                                        st.markdown("".join(alerts[-8:]), unsafe_allow_html=True)

                            if count >= n_transactions:
                                break

                    st.success(f"Received {count} transactions from inference service.")

                except Exception as e:
                    st.error(f"SSE connection failed: {e}. Is the inference service running?")

    # ---- Investigation Console ----
    elif ops_section == "Investigation Console":
        if "feedback_store" not in st.session_state:
            st.session_state.feedback_store = FeedbackStore()
        if "flagged_txns" not in st.session_state:
            st.session_state.flagged_txns = []

        feedback_store = st.session_state.feedback_store

        fb_stats = feedback_store.stats
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Total Reviews", fb_stats["total_reviews"])
        fc2.metric("Marked Fraud", fb_stats["marked_fraud"])
        fc3.metric("Marked Legitimate", fb_stats["marked_legitimate"])

        st.markdown("---")

        if not st.session_state.flagged_txns and st.session_state.model_results:
            preprocessed = st.session_state.preprocessed
            X_test = preprocessed["X_test"]
            y_test = preprocessed["y_test"]
            scores = st.session_state.model_results.get("isolation_forest", {}).get("scores", np.array([]))
            if len(scores) > 0:
                flagged_idx = np.where(scores >= 0.5)[0][:50]
                for idx in flagged_idx:
                    st.session_state.flagged_txns.append({
                        "transaction_id": f"test-{idx:05d}",
                        "fraud_probability": round(float(scores[idx]), 4),
                        "Amount": round(float(X_test[idx][28]) if X_test.shape[1] > 28 else 0, 2),
                        "risk_level": "HIGH" if scores[idx] >= 0.7 else "MEDIUM",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "features": {f"V{i+1}": round(float(X_test[idx][i]), 4) for i in range(min(28, X_test.shape[1]))},
                    })

        if st.session_state.flagged_txns:
            flagged_df = pd.DataFrame(st.session_state.flagged_txns)
            display_cols = [c for c in ["transaction_id", "fraud_probability", "Amount", "risk_level", "timestamp"] if c in flagged_df.columns]
            st.dataframe(flagged_df[display_cols].head(30), use_container_width=True, hide_index=True)

            st.markdown("---")
            txn_ids = [t["transaction_id"] for t in st.session_state.flagged_txns[:30]]
            selected_id = st.selectbox("Select transaction to investigate", txn_ids, key="investigate_txn")

            if selected_id:
                selected_txn = next((t for t in st.session_state.flagged_txns if t["transaction_id"] == selected_id), None)
                if selected_txn:
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown("**Transaction Details**")
                        st.json({
                            "transaction_id": selected_txn["transaction_id"],
                            "amount": selected_txn.get("Amount", 0),
                            "risk_level": selected_txn.get("risk_level", "UNKNOWN"),
                            "fraud_score": selected_txn.get("fraud_probability", 0),
                            "timestamp": selected_txn.get("timestamp", ""),
                        })

                    with ic2:
                        st.markdown("**Fraud Score Breakdown**")
                        score = selected_txn.get("fraud_probability", 0)
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            title={"text": "Fraud Risk %", "font": {"color": "#e6edf3", "family": "Sora"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#484f58"},
                                "bar": {"color": C_DANGER if score >= 0.5 else C_SUCCESS},
                                "steps": [
                                    {"range": [0, 30], "color": "rgba(0,255,135,0.08)"},
                                    {"range": [30, 70], "color": "rgba(255,200,87,0.08)"},
                                    {"range": [70, 100], "color": "rgba(255,107,107,0.08)"},
                                ],
                            },
                        ))
                        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    if "features" in selected_txn and st.session_state.model_results:
                        feats = selected_txn["features"]
                        top_feats = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                        if top_feats:
                            feat_df = pd.DataFrame(top_feats, columns=["Feature", "Value"])
                            fig_feat = px.bar(
                                feat_df, x="Value", y="Feature", orientation="h",
                                color="Value",
                                color_continuous_scale=[[0, "#00d2ff"], [0.5, "#0a0f18"], [1, "#ff6b6b"]],
                                color_continuous_midpoint=0,
                            )
                            fig_feat.update_layout(**PLOTLY_LAYOUT, height=300, title="Top Feature Values", coloraxis_showscale=False)
                            st.plotly_chart(fig_feat, use_container_width=True)

                    st.markdown("---")
                    st.markdown("**Analyst Decision**")
                    dc1, dc2, dc3 = st.columns([1, 1, 2])
                    with dc1:
                        if st.button("Mark as FRAUD", type="primary", key=f"fraud_{selected_id}"):
                            feedback_store.record(
                                transaction_id=selected_id,
                                analyst_decision="fraud",
                                fraud_score=selected_txn.get("fraud_probability", 0),
                                transaction_data=selected_txn.get("features", {}),
                            )
                            st.success(f"Marked {selected_id} as FRAUD")
                            st.rerun()
                    with dc2:
                        if st.button("Mark as LEGITIMATE", key=f"legit_{selected_id}"):
                            feedback_store.record(
                                transaction_id=selected_id,
                                analyst_decision="legitimate",
                                fraud_score=selected_txn.get("fraud_probability", 0),
                                transaction_data=selected_txn.get("features", {}),
                            )
                            st.success(f"Marked {selected_id} as LEGITIMATE")
                            st.rerun()
                    with dc3:
                        notes = st.text_input("Analyst notes (optional)", key=f"notes_{selected_id}")

            st.markdown("---")
            st.markdown('<div class="section-header">Review History</div>', unsafe_allow_html=True)
            past = feedback_store.load_all()
            if past:
                hist_df = pd.DataFrame(past)
                display_h = [c for c in ["transaction_id", "decision", "fraud_score", "reviewed_at"] if c in hist_df.columns]
                st.dataframe(hist_df[display_h].tail(20), use_container_width=True, hide_index=True)
            else:
                st.info("No analyst decisions recorded yet.")
        else:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">&#128269;</div>'
                '<div class="empty-state-title">No flagged transactions</div>'
                '<div class="empty-state-desc">Run the Streaming Monitor first to generate flagged transactions for investigation.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ---- Entity Risk Graph ----
    elif ops_section == "Entity Risk Graph":
        st.markdown(
            '<div class="insight-card">'
            'Fraud network analysis: entities (cards, merchants, devices, IPs) connected through '
            'shared transactions. Risk propagates through the network to surface fraud rings.'
            '</div>',
            unsafe_allow_html=True,
        )

        if "entity_graph" not in st.session_state:
            st.session_state.entity_graph = EntityRiskGraph()
        graph = st.session_state.entity_graph

        if graph.stats["total_transactions"] == 0:
            with st.spinner("Generating sample graph data..."):
                from streaming.producer import generate_transaction
                risk_scorer = RiskScorer()
                for i in range(300):
                    txn = generate_transaction(fraud_rate=0.03)
                    is_anom = "txn-F-" in txn.get("transaction_id", "")
                    score = 0.75 if is_anom else 0.2
                    graph.add_transaction(
                        card_id=txn.get("card_id", "unknown"),
                        merchant_id=txn.get("merchant_id", "unknown"),
                        device_id=txn.get("device_id", "unknown"),
                        ip_address=txn.get("source_ip", "unknown"),
                        location=txn.get("location", "unknown"),
                        is_anomaly=is_anom,
                        anomaly_score=score,
                    )

        gs = graph.stats
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Total Entities", gs["total_entities"])
        gc2.metric("Total Edges", gs["total_edges"])
        gc3.metric("Risky Entities", gs["risky_entities"])
        gc4.metric("Transactions", gs["total_transactions"])

        st.markdown("---")
        st.markdown('<div class="section-header">Suspicious Entity Clusters</div>', unsafe_allow_html=True)
        min_risk = st.slider("Minimum risk threshold", 0.1, 0.9, 0.3, 0.05, key="graph_min_risk")
        clusters = graph.get_suspicious_clusters(min_risk=min_risk)

        if clusters:
            for i, cluster in enumerate(clusters[:5]):
                with st.expander(
                    f"Cluster {i+1}: {cluster['size']} entities, "
                    f"avg risk {cluster['avg_risk']:.2f}, "
                    f"{cluster['total_anomalies']} anomalies"
                ):
                    st.json(cluster)
        else:
            st.info("No suspicious clusters found at this risk threshold.")

        st.markdown("---")
        st.markdown('<div class="section-header">Network Visualization</div>', unsafe_allow_html=True)

        graph_data = graph.get_graph_data_for_viz(max_nodes=100)
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        if nodes:
            import random as _rand
            _rand.seed(42)
            pos = {}
            for node in nodes:
                nid = node["id"]
                type_offsets = {"card": (0, 0), "merchant": (3, 0), "device": (0, 3), "ip": (3, 3), "location": (1.5, 1.5)}
                ox, oy = type_offsets.get(node["type"], (1.5, 1.5))
                pos[nid] = (ox + _rand.gauss(0, 1.2), oy + _rand.gauss(0, 1.2))

            edge_x, edge_y = [], []
            for edge in edges:
                if edge["source"] in pos and edge["target"] in pos:
                    x0, y0 = pos[edge["source"]]
                    x1, y1 = pos[edge["target"]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            fig_graph = go.Figure()
            fig_graph.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.4, color="rgba(139,148,158,0.06)"), hoverinfo="none",
            ))

            type_colors = {"card": C_INFO, "merchant": C_SUCCESS, "device": C_WARNING, "ip": C_DANGER, "location": "#94a3b8"}
            for node in nodes:
                nid = node["id"]
                if nid in pos:
                    x, y = pos[nid]
                    risk = node["risk"]
                    color = C_DANGER if risk > 0.5 else type_colors.get(node["type"], "#64748b")
                    size = max(6, min(20, node["txn_count"] * 2))
                    fig_graph.add_trace(go.Scatter(
                        x=[x], y=[y], mode="markers+text",
                        marker=dict(size=size, color=color, line=dict(width=1, color="#060a10")),
                        text=nid.split(":")[-1][:8] if risk > 0.3 else "",
                        textposition="top center",
                        textfont=dict(size=8, color="#94a3b8"),
                        hovertext=f"{nid}<br>Risk: {risk:.2f}<br>Txns: {node['txn_count']}<br>Anomalies: {node['anomaly_count']}",
                        hoverinfo="text",
                        showlegend=False,
                    ))

            fig_graph.update_layout(
                **PLOTLY_LAYOUT, height=550, showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_graph, use_container_width=True)

            st.markdown(
                '<div style="display:flex;gap:20px;justify-content:center;flex-wrap:wrap;padding:8px 0;">'
                f'<span style="color:{C_INFO};font-size:0.8rem;">&#9679; Card</span>'
                f'<span style="color:{C_SUCCESS};font-size:0.8rem;">&#9679; Merchant</span>'
                f'<span style="color:{C_WARNING};font-size:0.8rem;">&#9679; Device</span>'
                f'<span style="color:{C_DANGER};font-size:0.8rem;">&#9679; IP / High Risk</span>'
                f'<span style="color:#94a3b8;font-size:0.8rem;">&#9679; Location</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No graph data to visualize.")


# ===================== Tab 5: Monitoring =====================
with tab5:
    st.markdown('<div class="section-header">Model Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Production ML metrics: anomaly rates, volume trends, feature drift, and model versioning.</div>', unsafe_allow_html=True)

    if "monitor" not in st.session_state:
        st.session_state.monitor = MonitoringTracker()
    monitor = st.session_state.monitor

    if st.session_state.model_results and monitor._total_processed == 0:
        scores = st.session_state.model_results.get("isolation_forest", {}).get("scores", np.array([]))
        for i in range(min(len(scores), 500)):
            monitor.record_prediction(float(scores[i]), scores[i] >= 0.5)

    metrics = monitor.get_dashboard_metrics()

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total Processed", f"{metrics['total_processed']:,}")
    mc2.metric("Total Anomalies", f"{metrics['total_anomalies']:,}")
    mc3.metric("Anomaly Rate", f"{metrics['anomaly_rate']:.2%}")
    mc4.metric("Avg Score", f"{metrics['avg_score']:.4f}")
    mc5.metric("Model Version", metrics["model_version"])

    st.markdown("---")

    ts_data = metrics.get("time_series", [])
    if ts_data:
        ts_df = pd.DataFrame(ts_data)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown('<div class="section-header">Transaction Volume</div>', unsafe_allow_html=True)
            fig_vol = px.bar(ts_df, x="timestamp", y="volume", color_discrete_sequence=[C_ACCENT])
            fig_vol.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="", yaxis_title="Transactions")
            st.plotly_chart(fig_vol, use_container_width=True)

        with tc2:
            st.markdown('<div class="section-header">Anomaly Rate Trend</div>', unsafe_allow_html=True)
            fig_anom = px.line(ts_df, x="timestamp", y="anomaly_rate", color_discrete_sequence=[C_DANGER])
            fig_anom.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="", yaxis_title="Anomaly Rate")
            st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown('<div class="section-header">Average Score Trend</div>', unsafe_allow_html=True)
        fig_score = px.line(ts_df, x="timestamp", y="avg_score", color_discrete_sequence=[C_WARNING])
        fig_score.add_hline(y=0.5, line_dash="dash", line_color=C_DANGER, annotation_text="Threshold")
        fig_score.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="", yaxis_title="Avg Score")
        st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#128200;</div>'
            '<div class="empty-state-title">No time-series data yet</div>'
            '<div class="empty-state-desc">Start streaming transactions from the Live Operations tab to populate monitoring metrics.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Distribution Drift</div>', unsafe_allow_html=True)
    drift_data = metrics.get("feature_drift", {})
    if drift_data:
        drift_rows = []
        for fname, dinfo in drift_data.items():
            drift_rows.append({
                "Feature": fname,
                "Current Mean": round(dinfo["current_mean"], 4),
                "Baseline Mean": round(dinfo["baseline_mean"], 4),
                "Drift Magnitude": round(dinfo["drift_magnitude"], 4),
                "Status": "DRIFTED" if dinfo["drifted"] else "Stable",
            })
        drift_df = pd.DataFrame(drift_rows)
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("Feature drift tracking requires baseline statistics from training data.")

    st.markdown("---")
    st.markdown('<div class="section-header">Model Version History</div>', unsafe_allow_html=True)
    retrain_hist = metrics.get("retrain_history", [])
    if retrain_hist:
        rh_df = pd.DataFrame(retrain_hist)
        st.dataframe(rh_df, use_container_width=True, hide_index=True)
    else:
        st.info("No retraining events recorded.")

    st.markdown("---")
    rc1, rc2 = st.columns([1, 3])
    with rc1:
        retrain_model_name = st.selectbox("Model to retrain", MODEL_NAMES, key="retrain_model")
    with rc2:
        include_fb = st.checkbox("Include analyst feedback", value=True, key="retrain_feedback")

    if st.button("Trigger Retraining", type="primary", key="retrain_btn"):
        with st.spinner(f"Retraining {retrain_model_name}..."):
            from models.retrain import retrain_model
            result = retrain_model(model_name=retrain_model_name, include_feedback=include_fb)
            if result["status"] == "success":
                monitor.record_retrain(result["version"], result["metrics"])
                st.success(
                    f"Retrained {retrain_model_name} -> {result['version']} "
                    f"(F1={result['metrics']['f1']:.4f}, AUC={result['metrics']['roc_auc']:.4f}) "
                    f"in {result['train_time']:.1f}s"
                )
                st.session_state.pop("model_results", None)
                st.session_state.initialized = False
            else:
                st.error(f"Retraining failed: {result.get('message', 'unknown error')}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;padding:2rem 0 1.5rem 0;">'
    '<p style="color:#484f58;font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
    'letter-spacing:0.1em;text-transform:uppercase;margin:0;">'
    'Built by Sanjit Mathur &nbsp;//&nbsp; scikit-learn + FastAPI + Streamlit + SHAP '
    '&nbsp;//&nbsp; Credit Card Fraud Dataset</p>'
    '</div>',
    unsafe_allow_html=True,
)
