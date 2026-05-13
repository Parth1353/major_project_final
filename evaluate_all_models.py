"""Comprehensive test-set evaluation for all WhatsApp fake-news models.

Computes per-model:
  - Confusion matrix (TN, FP, FN, TP)
  - Accuracy, Precision, Recall, F1 (per-class AND macro/weighted)
  - Specificity, MCC, Cohen's kappa
  - ROC-AUC, Log-loss
  - Error counts (FP, FN)

Usage:
    python evaluate_all_models.py
    python evaluate_all_models.py --model xgboost
    python evaluate_all_models.py --save-json evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "artifacts"
TEST_PREDICTIONS = ARTIFACT_DIR / "test_predictions.csv"
LABEL_NAMES = ["REAL (0)", "FAKE (1)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full evaluation of WhatsApp fake-news models on test set.")
    parser.add_argument("--predictions", type=Path, default=TEST_PREDICTIONS, help="Path to test_predictions.csv")
    parser.add_argument("--model", help="Evaluate a single model only, e.g. xgboost, muril, stacking_ensemble")
    parser.add_argument("--save-json", type=Path, help="Save all metrics to a JSON file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default 0.5)")
    return parser.parse_args()


def get_model_names(df: pd.DataFrame) -> list[str]:
    """Extract model names from columns ending in _fake_prob."""
    return sorted(col.removesuffix("_fake_prob") for col in df.columns if col.endswith("_fake_prob"))


def compute_all_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute every classification metric for a single model."""
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class precision, recall, f1
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # Macro / Weighted averages
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Matthews Correlation Coefficient
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Cohen's Kappa
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # ROC-AUC (uses probabilities)
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = None  # single class edge case

    # Log loss
    try:
        eps = 1e-15
        y_probs_clipped = np.clip(y_probs, eps, 1 - eps)
        logloss = float(log_loss(y_true, y_probs_clipped))
    except ValueError:
        logloss = None

    return {
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            "REAL": {
                "precision": float(prec_per_class[0]),
                "recall": float(rec_per_class[0]),
                "f1_score": float(f1_per_class[0]),
                "support": int(support[0]),
            },
            "FAKE": {
                "precision": float(prec_per_class[1]),
                "recall": float(rec_per_class[1]),
                "f1_score": float(f1_per_class[1]),
                "support": int(support[1]),
            },
        },
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1_score": float(f1_weighted),
        },
        "specificity": specificity,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "roc_auc": auc,
        "log_loss": logloss,
        "error_counts": {
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_errors": int(fp + fn),
        },
        "threshold": threshold,
        "total_samples": int(len(y_true)),
    }


def print_separator(char: str = "═", width: int = 80) -> None:
    print(char * width)


def print_dataset_overview(df: pd.DataFrame) -> None:
    print_separator()
    print("  DATASET OVERVIEW")
    print_separator()
    print(f"  Total test samples:  {len(df):,}")
    label_counts = df["label"].value_counts().sort_index()
    print(f"  REAL (0):            {label_counts.get(0, 0):,}")
    print(f"  FAKE (1):            {label_counts.get(1, 0):,}")
    if "language" in df.columns:
        print(f"  Languages:           {dict(df['language'].value_counts())}")
    if "source_type" in df.columns:
        print(f"  Source types:        {dict(df['source_type'].value_counts())}")
    print()


def print_confusion_matrix_visual(cm: list[list[int]], model_name: str) -> None:
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │         CONFUSION MATRIX: {model_name:<18s}│")
    print(f"  ├─────────────────┬─────────────┬─────────────┤")
    print(f"  │                 │ Pred: REAL  │ Pred: FAKE  │")
    print(f"  ├─────────────────┼─────────────┼─────────────┤")
    print(f"  │ Actual: REAL    │  TN={tn:<6d} │  FP={fp:<6d} │")
    print(f"  ├─────────────────┼─────────────┼─────────────┤")
    print(f"  │ Actual: FAKE    │  FN={fn:<6d} │  TP={tp:<6d} │")
    print(f"  └─────────────────┴─────────────┴─────────────┘")
    print(f"  Total: {total:,}")


def print_model_report(model_name: str, metrics: dict[str, Any]) -> None:
    print_separator("─")
    print(f"  MODEL: {model_name.upper()}")
    print_separator("─")

    # Confusion matrix visual
    print_confusion_matrix_visual(metrics["confusion_matrix"]["matrix"], model_name)

    # Per-class metrics
    print(f"\n  ┌───────────────────────────────────────────────────────────────┐")
    print(f"  │                    CLASSIFICATION REPORT                      │")
    print(f"  ├─────────────┬───────────┬───────────┬───────────┬─────────────┤")
    print(f"  │ Class       │ Precision │   Recall  │  F1-Score │   Support   │")
    print(f"  ├─────────────┼───────────┼───────────┼───────────┼─────────────┤")
    for cls_name in ["REAL", "FAKE"]:
        cls = metrics["per_class"][cls_name]
        print(f"  │ {cls_name:<11s} │  {cls['precision']:.4f}   │  {cls['recall']:.4f}   │  {cls['f1_score']:.4f}   │   {cls['support']:<9d} │")
    print(f"  ├─────────────┼───────────┼───────────┼───────────┼─────────────┤")
    m = metrics["macro_avg"]
    print(f"  │ Macro Avg   │  {m['precision']:.4f}   │  {m['recall']:.4f}   │  {m['f1_score']:.4f}   │   {metrics['total_samples']:<9d} │")
    w = metrics["weighted_avg"]
    print(f"  │ Weighted    │  {w['precision']:.4f}   │  {w['recall']:.4f}   │  {w['f1_score']:.4f}   │   {metrics['total_samples']:<9d} │")
    print(f"  └─────────────┴───────────┴───────────┴───────────┴─────────────┘")

    # Summary metrics
    print(f"\n  Summary Metrics:")
    print(f"    Accuracy:            {metrics['accuracy']:.6f}")
    print(f"    Specificity:         {metrics['specificity']:.6f}")
    print(f"    MCC:                 {metrics['mcc']:.6f}")
    print(f"    Cohen's Kappa:       {metrics['cohen_kappa']:.6f}")
    if metrics["roc_auc"] is not None:
        print(f"    ROC-AUC:             {metrics['roc_auc']:.6f}")
    if metrics["log_loss"] is not None:
        print(f"    Log Loss:            {metrics['log_loss']:.6f}")
    print(f"    False Positives:     {metrics['error_counts']['false_positives']}")
    print(f"    False Negatives:     {metrics['error_counts']['false_negatives']}")
    print(f"    Total Errors:        {metrics['error_counts']['total_errors']}")
    print(f"    Threshold:           {metrics['threshold']}")
    print()


def print_comparison_table(all_metrics: dict[str, dict[str, Any]]) -> None:
    print_separator("═")
    print("  MODEL COMPARISON SUMMARY")
    print_separator("═")
    header = (
        f"  {'Model':<22s} │ {'Acc':>7s} │ {'Prec(M)':>8s} │ {'Rec(M)':>7s} │ "
        f"{'F1(M)':>7s} │ {'MCC':>7s} │ {'Kappa':>7s} │ {'AUC':>7s} │ {'LogLoss':>8s} │ {'Errors':>6s}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for name, m in all_metrics.items():
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "  N/A "
        ll_str = f"{m['log_loss']:.5f}" if m['log_loss'] is not None else "   N/A  "
        print(
            f"  {name:<22s} │ {m['accuracy']:>7.4f} │ {m['macro_avg']['precision']:>8.4f} │ "
            f"{m['macro_avg']['recall']:>7.4f} │ {m['macro_avg']['f1_score']:>7.4f} │ "
            f"{m['mcc']:>7.4f} │ {m['cohen_kappa']:>7.4f} │ {auc_str:>7s} │ {ll_str:>8s} │ "
            f"{m['error_counts']['total_errors']:>6d}"
        )
    print()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        print(f"Error: predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.predictions)
    print_dataset_overview(df)

    y_true = df["label"].astype(int).to_numpy()
    models = get_model_names(df)
    if args.model:
        if args.model not in models:
            print(f"Error: model '{args.model}' not found. Available: {', '.join(models)}", file=sys.stderr)
            sys.exit(1)
        models = [args.model]

    print(f"  Evaluating {len(models)} model(s): {', '.join(models)}\n")

    all_metrics: dict[str, dict[str, Any]] = {}
    for model_name in models:
        prob_col = f"{model_name}_fake_prob"
        y_probs = df[prob_col].to_numpy()
        metrics = compute_all_metrics(y_true, y_probs, threshold=args.threshold)
        all_metrics[model_name] = metrics
        print_model_report(model_name, metrics)

    if len(all_metrics) > 1:
        print_comparison_table(all_metrics)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Results saved to: {args.save_json}")

    print_separator("═")
    print("  EVALUATION COMPLETE")
    print_separator("═")


if __name__ == "__main__":
    main()
