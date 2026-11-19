"""Metrics, risk stratification and evaluation plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RISK_LABELS = ["Low", "Moderate", "High"]


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Threshold-free (ROC-AUC, PR-AUC, Brier) and threshold-based metrics."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "threshold": threshold,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def assign_risk_groups(y_proba, cutoffs=(0.02, 0.20)) -> pd.Categorical:
    """Bucket predicted probabilities into Low / Moderate / High risk."""
    bins = [-np.inf, cutoffs[0], cutoffs[1], np.inf]
    return pd.cut(np.asarray(y_proba), bins=bins, labels=RISK_LABELS, right=False)


def risk_group_table(y_true, y_proba, cutoffs=(0.02, 0.20)) -> pd.DataFrame:
    """Size, observed prevalence and mean predicted risk for each risk group."""
    df = pd.DataFrame({
        "group": assign_risk_groups(y_proba, cutoffs),
        "y": np.asarray(y_true),
        "p": np.asarray(y_proba),
    })
    table = df.groupby("group", observed=False).agg(
        individuals=("y", "size"),
        observed_diabetes=("y", "mean"),
        mean_predicted_risk=("p", "mean"),
    )
    table["share_of_all_cases"] = df.groupby("group", observed=False)["y"].sum() / df["y"].sum()
    return table


# ----------------------------------------------------------------------------- plots

def plot_roc_pr(curves: dict, y_true, ax=None):
    """ROC and precision-recall curves for several models. ``curves`` maps label -> proba."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if ax is None else (ax[0].figure, ax)
    prevalence = np.mean(y_true)
    for label, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, proba)
        axes[1].plot(rec, prec, label=f"{label} (AP={average_precision_score(y_true, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC curve")
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1, label=f"No skill ({prevalence:.3f})")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-recall curve")
    for a in axes:
        a.legend(loc="lower right" if a is axes[0] else "lower left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_calibration(curves: dict, y_true, n_bins: int = 10, ax=None):
    """Reliability diagram (quantile bins) for one or more probability vectors."""
    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    for label, proba in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-",
                label=f"{label} (Brier={brier_score_loss(y_true, proba):.4f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction with diabetes",
           title="Calibration curve")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_proba, threshold: float, title: str = "", ax=None):
    """Annotated confusion matrix at a given threshold."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax)
    cm = confusion_matrix(y_true, (np.asarray(y_proba) >= threshold).astype(int))
    names = np.array([["TN", "FP"], ["FN", "TP"]])
    annot = np.array([[f"{names[i, j]}\n{cm[i, j]:,}" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No diabetes", "Diabetes"], yticklabels=["No diabetes", "Diabetes"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title or f"Threshold = {threshold:.2f}")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(y_true, y_proba, chosen: float | None = None, ax=None):
    """Precision, recall and F1 as a function of the decision threshold."""
    fig, ax = plt.subplots(figsize=(8, 5)) if ax is None else (ax.figure, ax)
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    ax.plot(thr, prec, label="Precision")
    ax.plot(thr, rec, label="Recall (sensitivity)")
    ax.plot(thr, f1, label="F1")
    if chosen is not None:
        ax.axvline(chosen, color="black", ls="--", lw=1, label=f"Chosen = {chosen:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Threshold trade-off (validation)",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_risk_stratification(table: pd.DataFrame, ax=None):
    """Observed vs mean predicted diabetes rate per risk group."""
    fig, ax = plt.subplots(figsize=(7, 5)) if ax is None else (ax.figure, ax)
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["observed_diabetes"] * 100, width=0.4, label="Observed")
    ax.bar(x + 0.2, table["mean_predicted_risk"] * 100, width=0.4, label="Mean predicted")
    for i, (n, obs) in enumerate(zip(table["individuals"], table["observed_diabetes"])):
        ax.text(i - 0.2, obs * 100 + 1, f"n={n:,}", ha="center", fontsize=9)
    ax.set_xticks(x, table.index.astype(str))
    ax.set(ylabel="Diabetes rate (%)", title="Risk stratification (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def save_fig(fig, path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ----------------------------------------------------------------------------- decision curve analysis

def net_benefit(y_true, y_pred, threshold: float) -> float:
    """Net benefit of acting on binary decisions ``y_pred`` at threshold probability ``threshold``.

    NB = TP/n - FP/n * (pt / (1 - pt))  (Vickers & Elkin, 2006). The threshold
    probability expresses how many false positives one true positive is worth.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).astype(bool)
    n = len(y_true)
    tp = np.sum(y_pred & (y_true == 1))
    fp = np.sum(y_pred & (y_true == 0))
    return tp / n - fp / n * threshold / (1 - threshold)


def decision_curve(y_true, strategies: dict, thresholds=None) -> pd.DataFrame:
    """Net benefit across threshold probabilities.

    ``strategies`` maps a name to either predicted probabilities (classified at each
    threshold) or a fixed boolean rule (e.g. ``HbA1c >= 6.5``) whose decisions do not
    depend on the threshold. "Treat all" and "treat none" are added automatically.
    """
    thresholds = np.arange(0.01, 0.51, 0.01) if thresholds is None else np.asarray(thresholds)
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        row = {"threshold": t,
               "Treat all": net_benefit(y_true, np.ones_like(y_true), t),
               "Treat none": 0.0}
        for name, values in strategies.items():
            values = np.asarray(values)
            decisions = values if values.dtype == bool else values >= t
            row[name] = net_benefit(y_true, decisions, t)
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def plot_decision_curve(curve: pd.DataFrame, title: str = "Decision curve analysis",
                        mark: float | None = None, ax=None):
    """Plot net benefit curves; the y-axis is clipped just below zero for readability."""
    fig, ax = plt.subplots(figsize=(8, 5)) if ax is None else (ax.figure, ax)
    styles = {"Treat all": dict(color="grey", lw=1), "Treat none": dict(color="black", lw=1, ls="--")}
    for col in curve.columns:
        ax.plot(curve.index, curve[col], label=col, **styles.get(col, dict(lw=2)))
    if mark is not None:
        ax.axvline(mark, color="black", ls=":", lw=1, label=f"Screening threshold ({mark:.2f})")
    top = curve.drop(columns=["Treat none"]).to_numpy().max()
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title=title,
           ylim=(-0.02, top * 1.1), xlim=(curve.index.min(), curve.index.max()))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
