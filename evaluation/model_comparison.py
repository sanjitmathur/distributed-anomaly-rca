"""Model comparison visualizations using Plotly."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_leaderboard(all_metrics: dict) -> pd.DataFrame:
    """Create a leaderboard DataFrame sorted by F1.

    Parameters
    ----------
    all_metrics : dict mapping model_name -> metrics dict

    Returns
    -------
    pd.DataFrame with columns: Model, Precision, Recall, F1, ROC-AUC, Avg Precision
    """
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Avg Precision": round(m["avg_precision"], 4),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def plot_roc_curves(all_metrics: dict) -> go.Figure:
    """Overlaid ROC curves for all models."""
    fig = go.Figure()
    for name, m in all_metrics.items():
        fig.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"],
            mode="lines",
            name=f"{name} (AUC={m['roc_auc']:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )
    return fig


def plot_pr_curves(all_metrics: dict) -> go.Figure:
    """Overlaid Precision-Recall curves for all models."""
    fig = go.Figure()
    for name, m in all_metrics.items():
        fig.add_trace(go.Scatter(
            x=m["pr_recalls"], y=m["pr_precisions"],
            mode="lines",
            name=f"{name} (AP={m['avg_precision']:.3f})",
        ))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
    )
    return fig


def plot_metric_bars(all_metrics: dict) -> go.Figure:
    """Bar chart comparing F1, Precision, Recall across models."""
    models = list(all_metrics.keys())
    fig = go.Figure()
    for metric_name in ["Precision", "Recall", "F1"]:
        key = metric_name.lower() if metric_name != "F1" else "f1"
        values = [all_metrics[m][key] for m in models]
        fig.add_trace(go.Bar(name=metric_name, x=models, y=values))
    fig.update_layout(
        title="Model Comparison",
        barmode="group",
        yaxis_title="Score",
        height=400,
    )
    return fig
