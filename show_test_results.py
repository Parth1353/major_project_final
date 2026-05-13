"""Show model results on the saved test dataset predictions.

Run:
  python3 show_test_results.py
  python3 show_test_results.py --errors 10
  python3 show_test_results.py --model xgboost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "artifacts"
TEST_PREDICTIONS = ARTIFACT_DIR / "test_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show test-dataset results from saved predictions.")
    parser.add_argument("--predictions", type=Path, default=TEST_PREDICTIONS)
    parser.add_argument("--model", help="Optional single model, e.g. xgboost, muril, stacking_ensemble.")
    parser.add_argument("--errors", type=int, default=5, help="Number of false positives/negatives to print per model.")
    return parser.parse_args()


def prediction_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column.removesuffix("_fake_prob") for column in df.columns if column.endswith("_fake_prob"))


def prediction_labels(fake_probs: pd.Series) -> pd.Series:
    return (fake_probs >= 0.5).astype(int)


def print_distribution(df: pd.DataFrame) -> None:
    print("=== Test Dataset ===")
    print(f"Rows: {len(df):,}")
    print("Labels:", df["label"].value_counts().sort_index().to_dict(), "(0=REAL, 1=FAKE)")
    if "language" in df:
        print("Language:", df["language"].value_counts().to_dict())
    if "source_type" in df:
        print("Source type:", df["source_type"].value_counts().to_dict())


def print_model_metrics(df: pd.DataFrame, model: str, errors: int) -> None:
    prob_column = f"{model}_fake_prob"
    if prob_column not in df.columns:
        print(f"\nSkipping {model}: missing {prob_column}")
        return

    y_true = df["label"].astype(int)
    y_pred = prediction_labels(df[prob_column])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n=== {model} ===")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision macro: {precision:.4f}")
    print(f"Recall macro:    {recall:.4f}")
    print(f"F1 macro:        {f1:.4f}")
    print(f"Confusion matrix [[real_ok, real_as_fake], [fake_as_real, fake_ok]]: {matrix.tolist()}")

    false_positive = df[(y_true == 0) & (y_pred == 1)].copy()
    false_negative = df[(y_true == 1) & (y_pred == 0)].copy()
    print(f"False positives: {len(false_positive)}")
    print(f"False negatives: {len(false_negative)}")

    if errors > 0 and not false_positive.empty:
        print("\nFalse positive examples:")
        for _, row in false_positive.head(errors).iterrows():
            print(f"- prob={row[prob_column]:.4f} lang={row.get('language', '')} text={str(row['text'])[:180].replace(chr(10), ' ')}")

    if errors > 0 and not false_negative.empty:
        print("\nFalse negative examples:")
        for _, row in false_negative.head(errors).iterrows():
            print(f"- prob={row[prob_column]:.4f} lang={row.get('language', '')} text={str(row['text'])[:180].replace(chr(10), ' ')}")


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Missing predictions file: {args.predictions}")

    df = pd.read_csv(args.predictions)
    print_distribution(df)

    models = prediction_columns(df)
    if args.model:
        models = [args.model]

    print("\nAvailable models:", ", ".join(prediction_columns(df)))
    for model in models:
        print_model_metrics(df, model, args.errors)


if __name__ == "__main__":
    main()

