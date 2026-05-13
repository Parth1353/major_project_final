"""Generate visual confusion matrix heatmaps and metrics bar charts.

Usage:
    python3 plot_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "artifacts"
TEST_PREDICTIONS = ARTIFACT_DIR / "test_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_plots"


def get_model_names(df: pd.DataFrame) -> list[str]:
    return sorted(col.removesuffix("_fake_prob") for col in df.columns if col.endswith("_fake_prob"))


DISPLAY_NAMES = {
    "logistic_regression": "Logistic\nRegression",
    "naive_bayes": "Naive\nBayes",
    "xgboost": "XGBoost",
    "punjabi_bert": "Punjabi\nBERT",
    "muril": "MuRIL",
    "stacking_ensemble": "Stacking\nEnsemble",
}

MODEL_COLORS = {
    "logistic_regression": "#6366f1",
    "naive_bayes": "#f59e0b",
    "xgboost": "#10b981",
    "punjabi_bert": "#3b82f6",
    "muril": "#8b5cf6",
    "stacking_ensemble": "#ef4444",
}


def plot_confusion_matrices(df: pd.DataFrame, models: list[str]) -> None:
    """Plot a grid of confusion matrix heatmaps for all models."""
    y_true = df["label"].astype(int).to_numpy()
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    fig.suptitle("Confusion Matrices — All Models on Test Set (n=2,901)",
                 fontsize=20, fontweight="bold", y=1.02, color="#1e293b")
    
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, model in enumerate(models):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        
        prob_col = f"{model}_fake_prob"
        y_pred = (df[prob_col] >= 0.5).astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        # Normalized confusion matrix for color
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        # Custom colormap
        cmap = plt.cm.Blues
        
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        
        # Text annotations
        labels_text = [
            [f"TN\n{tn:,}\n({cm_norm[0,0]:.1%})", f"FP\n{fp:,}\n({cm_norm[0,1]:.1%})"],
            [f"FN\n{fn:,}\n({cm_norm[1,0]:.1%})", f"TP\n{tp:,}\n({cm_norm[1,1]:.1%})"],
        ]
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "#1e293b"
                ax.text(j, i, labels_text[i][j], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        display = DISPLAY_NAMES.get(model, model).replace("\n", " ")
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{display}\nAccuracy: {acc:.4f}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["REAL", "FAKE"], fontsize=11)
        ax.set_yticklabels(["REAL", "FAKE"], fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")


def plot_metrics_comparison(df: pd.DataFrame, models: list[str]) -> None:
    """Bar chart comparing key metrics across all models."""
    y_true = df["label"].astype(int).to_numpy()
    
    metrics_data = {}
    for model in models:
        prob_col = f"{model}_fake_prob"
        y_probs = df[prob_col].to_numpy()
        y_pred = (y_probs >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            auc = 0.0
        metrics_data[model] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "MCC": mcc,
            "Cohen's κ": kappa,
            "ROC-AUC": auc,
        }

    metric_names = list(next(iter(metrics_data.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.12
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(18, 8))
    
    for i, model in enumerate(models):
        values = [metrics_data[model][m] for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=DISPLAY_NAMES.get(model, model).replace("\n", " "),
                      color=MODEL_COLORS.get(model, "#64748b"), alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0.88, 1.005)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Model Performance Comparison — All Metrics",
                 fontsize=18, fontweight="bold", pad=20, color="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9, ncol=3)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'metrics_comparison.png'}")


def plot_roc_curves(df: pd.DataFrame, models: list[str]) -> None:
    """ROC curves for all models on the same plot."""
    y_true = df["label"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    
    for model in models:
        prob_col = f"{model}_fake_prob"
        y_probs = df[prob_col].to_numpy()
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            auc = roc_auc_score(y_true, y_probs)
            display = DISPLAY_NAMES.get(model, model).replace("\n", " ")
            ax.plot(fpr, tpr, linewidth=2.5, color=MODEL_COLORS.get(model, "#64748b"),
                    label=f"{display} (AUC = {auc:.4f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("ROC Curves — All Models", fontsize=18, fontweight="bold", pad=15, color="#1e293b")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'roc_curves.png'}")


def plot_error_analysis(df: pd.DataFrame, models: list[str]) -> None:
    """Stacked bar chart showing FP vs FN breakdown per model."""
    y_true = df["label"].astype(int).to_numpy()
    
    fps = []
    fns = []
    for model in models:
        prob_col = f"{model}_fake_prob"
        y_pred = (df[prob_col] >= 0.5).astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fps.append(fp)
        fns.append(fn)

    x = np.arange(len(models))
    display_labels = [DISPLAY_NAMES.get(m, m).replace("\n", " ") for m in models]

    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars_fp = ax.bar(x, fps, 0.5, label="False Positives (REAL → FAKE)", color="#ef4444", alpha=0.85)
    bars_fn = ax.bar(x, fns, 0.5, bottom=fps, label="False Negatives (FAKE → REAL)", color="#3b82f6", alpha=0.85)

    # Add value labels
    for i, (fp, fn) in enumerate(zip(fps, fns)):
        total = fp + fn
        if total > 0:
            ax.text(i, total + 1, f"{total}", ha="center", va="bottom", fontsize=12, fontweight="bold")
            if fp > 0:
                ax.text(i, fp / 2, f"FP={fp}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            if fn > 0:
                ax.text(i, fp + fn / 2, f"FN={fn}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(i, 0.5, "0 errors ✓", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#10b981")

    ax.set_ylabel("Error Count", fontsize=13, fontweight="bold")
    ax.set_title("Error Analysis — False Positives & False Negatives per Model",
                 fontsize=16, fontweight="bold", pad=15, color="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'error_analysis.png'}")


def plot_per_class_f1(df: pd.DataFrame, models: list[str]) -> None:
    """Per-class F1 scores comparison."""
    y_true = df["label"].astype(int).to_numpy()

    real_f1s = []
    fake_f1s = []
    for model in models:
        prob_col = f"{model}_fake_prob"
        y_pred = (df[prob_col] >= 0.5).astype(int).to_numpy()
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        real_f1s.append(f1[0])
        fake_f1s.append(f1[1])

    x = np.arange(len(models))
    width = 0.35
    display_labels = [DISPLAY_NAMES.get(m, m).replace("\n", " ") for m in models]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, real_f1s, width, label="REAL (class 0)", color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x + width/2, fake_f1s, width, label="FAKE (class 1)", color="#ef4444", alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0.95, 1.008)
    ax.set_ylabel("F1-Score", fontsize=13, fontweight="bold")
    ax.set_title("Per-Class F1 Scores — REAL vs FAKE",
                 fontsize=16, fontweight="bold", pad=15, color="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_f1.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_class_f1.png'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_PREDICTIONS.exists():
        print(f"Error: {TEST_PREDICTIONS} not found")
        return

    df = pd.read_csv(TEST_PREDICTIONS)
    models = get_model_names(df)
    print(f"Test set: {len(df):,} samples, {len(models)} models: {', '.join(models)}\n")

    plot_confusion_matrices(df, models)
    plot_metrics_comparison(df, models)
    plot_roc_curves(df, models)
    plot_error_analysis(df, models)
    plot_per_class_f1(df, models)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
