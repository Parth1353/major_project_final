"""Train WhatsApp fake-news models from prepared CSV splits.

This is the script equivalent of the Kaggle notebook. It is safe to run on
Kaggle with GPU for transformers, or locally with ``--skip-transformers`` for a
quick classical-model refresh.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference_utils import METADATA_FEATURES, preprocess_whatsapp  # noqa: E402


REQUIRED_COLUMNS = ["text", "label", "language", "source_type"]
DEFAULT_TRANSFORMER_SPECS = {
    "indic_bert": ["ai4bharat/indic-bert"],
    "punjabi_bert": ["neuralspace-reverie/indic-transformers-pa-bert", "l3cube-pune/punjabi-bert"],
    "muril": ["google/muril-base-cased"],
}


def parse_args() -> argparse.Namespace:
    default_data_dir = PROJECT_ROOT / "data" / "whatsapp_dataset"
    default_output_dir = PROJECT_ROOT / "models" / "artifacts"
    parser = argparse.ArgumentParser(description="Train WhatsApp fake-news detection models.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional per-split sample for smoke tests.")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from transformers import set_seed

        torch.manual_seed(seed)
        set_seed(seed)
    except Exception:
        pass


def validate_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["label"] = out["label"].astype(int)
    bad_labels = sorted(set(out["label"].unique()) - {0, 1})
    if bad_labels:
        raise ValueError(f"{split_name} has labels outside {{0, 1}}: {bad_labels}")
    return out


def load_splits(data_dir: Path, sample_size: int = 0, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for split in ["train", "valid", "test"]:
        path = data_dir / f"whatsapp_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split: {path}")
        df = validate_frame(pd.read_csv(path), split)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)
        frames.append(preprocess_dataframe(df))
    return tuple(frames)  # type: ignore[return-value]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = [preprocess_whatsapp(text) for text in df["text"]]
    out = df.copy()
    out["clean_text"] = [row.clean_text for row in rows]
    out["detected_language"] = [row.language_detected for row in rows]
    out["transformer_text"] = [row.transformer_text for row in rows]
    for feature in METADATA_FEATURES:
        out[feature] = [row.features[feature] for row in rows]
    return out


def fake_probability_from_estimator(estimator: Any, x_matrix: Any) -> np.ndarray:
    probabilities = estimator.predict_proba(x_matrix)
    classes = list(estimator.classes_)
    return probabilities[:, classes.index(1)]


def metrics_from_probabilities(y_true: np.ndarray, fake_probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (fake_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def register_result(
    model_name: str,
    y_true: np.ndarray,
    fake_probs: np.ndarray,
    model_comparison: dict[str, Any],
) -> dict[str, Any]:
    metrics = metrics_from_probabilities(y_true, fake_probs)
    model_comparison[model_name] = metrics
    print(f"\n=== {model_name} ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def train_classical(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    model_comparison: dict[str, Any],
    valid_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
) -> None:
    from xgboost import XGBClassifier

    classical_dir = output_dir / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    x_train_tfidf = vectorizer.fit_transform(train_df["clean_text"])
    x_valid_tfidf = vectorizer.transform(valid_df["clean_text"])
    x_test_tfidf = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"].to_numpy()
    y_valid = valid_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    logistic_regression = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    logistic_regression.fit(x_train_tfidf, y_train)
    valid_probabilities["logistic_regression"] = fake_probability_from_estimator(logistic_regression, x_valid_tfidf)
    test_probabilities["logistic_regression"] = fake_probability_from_estimator(logistic_regression, x_test_tfidf)
    register_result("logistic_regression", y_test, test_probabilities["logistic_regression"], model_comparison)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_tfidf, y_train)
    valid_probabilities["naive_bayes"] = fake_probability_from_estimator(naive_bayes, x_valid_tfidf)
    test_probabilities["naive_bayes"] = fake_probability_from_estimator(naive_bayes, x_test_tfidf)
    register_result("naive_bayes", y_test, test_probabilities["naive_bayes"], model_comparison)

    scaler = StandardScaler()
    train_meta = scaler.fit_transform(train_df[METADATA_FEATURES].astype(float))
    valid_meta = scaler.transform(valid_df[METADATA_FEATURES].astype(float))
    test_meta = scaler.transform(test_df[METADATA_FEATURES].astype(float))

    x_train_xgb = sparse.hstack([x_train_tfidf, sparse.csr_matrix(train_meta)], format="csr")
    x_valid_xgb = sparse.hstack([x_valid_tfidf, sparse.csr_matrix(valid_meta)], format="csr")
    x_test_xgb = sparse.hstack([x_test_tfidf, sparse.csr_matrix(test_meta)], format="csr")

    xgboost = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgboost.fit(x_train_xgb, y_train, eval_set=[(x_valid_xgb, y_valid)], verbose=False)
    valid_probabilities["xgboost"] = fake_probability_from_estimator(xgboost, x_valid_xgb)
    test_probabilities["xgboost"] = fake_probability_from_estimator(xgboost, x_test_xgb)
    register_result("xgboost", y_test, test_probabilities["xgboost"], model_comparison)

    joblib.dump(vectorizer, classical_dir / "tfidf_vectorizer.joblib")
    joblib.dump(logistic_regression, classical_dir / "logistic_regression.joblib")
    joblib.dump(naive_bayes, classical_dir / "naive_bayes.joblib")
    joblib.dump(scaler, classical_dir / "metadata_scaler.joblib")
    joblib.dump(xgboost, classical_dir / "xgboost.joblib")
    save_json(classical_dir / "feature_config.json", {"metadata_features": METADATA_FEATURES})


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_transformers(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    model_comparison: dict[str, Any],
    valid_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    transformer_dir = output_dir / "transformers"
    trainer_runs_dir = output_dir / "trainer_runs"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    def tokenized_dataset(df: pd.DataFrame, tokenizer: Any) -> Any:
        dataset = Dataset.from_pandas(
            df[["transformer_text", "label"]].rename(columns={"label": "labels"}),
            preserve_index=False,
        )

        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(
                batch["transformer_text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["transformer_text"])
        tokenized.set_format(type="torch")
        return tokenized

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    def training_arguments(alias: str) -> Any:
        kwargs: dict[str, Any] = {
            "output_dir": str(trainer_runs_dir / alias),
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "num_train_epochs": args.epochs,
            "weight_decay": 0.01,
            "logging_steps": 100,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "report_to": "none",
            "seed": args.seed,
            "fp16": bool(torch.cuda.is_available()),
            "save_strategy": "epoch",
        }
        signature = inspect.signature(TrainingArguments.__init__)
        kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "epoch"
        return TrainingArguments(**kwargs)

    for alias, candidates in DEFAULT_TRANSFORMER_SPECS.items():
        last_error = None
        for model_id in candidates:
            try:
                print(f"\nTraining {alias} from {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    num_labels=2,
                    id2label={0: "REAL", 1: "FAKE"},
                    label2id={"REAL": 0, "FAKE": 1},
                    ignore_mismatched_sizes=True,
                )
                tokenized_train = tokenized_dataset(train_df, tokenizer)
                tokenized_valid = tokenized_dataset(valid_df, tokenizer)
                tokenized_test = tokenized_dataset(test_df, tokenizer)
                trainer_kwargs = {
                    "model": model,
                    "args": training_arguments(alias),
                    "train_dataset": tokenized_train,
                    "eval_dataset": tokenized_valid,
                    "compute_metrics": compute_metrics,
                }
                trainer_signature = inspect.signature(Trainer.__init__)
                trainer_kwargs["processing_class" if "processing_class" in trainer_signature.parameters else "tokenizer"] = tokenizer
                trainer = Trainer(**trainer_kwargs)
                trainer.train()
                save_dir = transformer_dir / alias
                trainer.save_model(save_dir)
                tokenizer.save_pretrained(save_dir)
                valid_probabilities[alias] = softmax_numpy(trainer.predict(tokenized_valid).predictions)[:, 1]
                test_probabilities[alias] = softmax_numpy(trainer.predict(tokenized_test).predictions)[:, 1]
                metrics = register_result(alias, test_df["label"].to_numpy(), test_probabilities[alias], model_comparison)
                metrics["source_model_id"] = model_id
                save_json(
                    save_dir / "training_metadata.json",
                    {
                        "alias": alias,
                        "source_model_id": model_id,
                        "max_length": args.max_length,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                    },
                )
                del trainer, model, tokenizer, tokenized_train, tokenized_valid, tokenized_test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            except Exception as exc:
                last_error = repr(exc)
                print(f"Failed {alias} candidate {model_id}: {last_error}")
                model_comparison[f"{alias}__{model_id}__error"] = {"error": last_error}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if last_error and alias not in test_probabilities:
            print(f"All candidates failed for {alias}.")


def train_stacking(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    model_comparison: dict[str, Any],
    valid_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
) -> None:
    from xgboost import XGBClassifier

    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    stack_feature_names: list[str] = []
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for model_name in sorted(valid_probabilities):
        if model_name not in test_probabilities:
            continue
        stack_feature_names.append(f"{model_name}_fake_prob")
        valid_parts.append(valid_probabilities[model_name].reshape(-1, 1))
        test_parts.append(test_probabilities[model_name].reshape(-1, 1))

    if not valid_parts:
        print("Skipping stacking ensemble because no base model probabilities are available.")
        return

    scaler_path = output_dir / "classical" / "metadata_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else StandardScaler().fit(valid_df[METADATA_FEATURES].astype(float))
    valid_meta = scaler.transform(valid_df[METADATA_FEATURES].astype(float))
    test_meta = scaler.transform(test_df[METADATA_FEATURES].astype(float))
    stack_feature_names.extend(METADATA_FEATURES)
    valid_parts.append(valid_meta)
    test_parts.append(test_meta)

    stacker = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    x_valid = np.column_stack(valid_parts)
    x_test = np.column_stack(test_parts)
    stacker.fit(x_valid, valid_df["label"].to_numpy(), verbose=False)
    test_probabilities["stacking_ensemble"] = fake_probability_from_estimator(stacker, x_test)
    register_result("stacking_ensemble", test_df["label"].to_numpy(), test_probabilities["stacking_ensemble"], model_comparison)
    joblib.dump(stacker, ensemble_dir / "stacking_xgboost.joblib")
    save_json(ensemble_dir / "stacking_feature_config.json", {"feature_names": stack_feature_names})


def write_predictions(test_df: pd.DataFrame, output_dir: Path, test_probabilities: dict[str, np.ndarray]) -> None:
    prediction_frame = test_df[["text", "label", "language", "source_type", "detected_language"]].copy()
    for model_name, probabilities in test_probabilities.items():
        probability_column = f"{model_name}_fake_prob"
        prediction_column = f"{model_name}_prediction"
        prediction_frame[probability_column] = probabilities
        prediction_frame[prediction_column] = np.where(probabilities >= 0.5, "FAKE", "REAL")
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_comparison: dict[str, Any] = {}
    valid_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    train_df, valid_df, test_df = load_splits(args.data_dir, sample_size=args.sample_size, seed=args.seed)
    print(f"Train/valid/test sizes: {len(train_df):,}/{len(valid_df):,}/{len(test_df):,}")

    if not args.skip_classical:
        train_classical(train_df, valid_df, test_df, args.output_dir, model_comparison, valid_probabilities, test_probabilities)
    if not args.skip_transformers:
        train_transformers(train_df, valid_df, test_df, args.output_dir, args, model_comparison, valid_probabilities, test_probabilities)
    if not args.skip_stacking:
        train_stacking(valid_df, test_df, args.output_dir, model_comparison, valid_probabilities, test_probabilities)

    write_predictions(test_df, args.output_dir, test_probabilities)
    save_json(args.output_dir / "model_comparison.json", model_comparison)
    save_json(
        args.output_dir / "training_config.json",
        {
            "dataset_dir": str(args.data_dir),
            "seed": args.seed,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "classical_models": ["logistic_regression", "naive_bayes", "xgboost"],
            "transformer_specs": DEFAULT_TRANSFORMER_SPECS,
            "metadata_features": METADATA_FEATURES,
            "label_map": {"0": "REAL", "1": "FAKE"},
            "created_at_unix": int(time.time()),
        },
    )
    archive_path = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(f"Saved model bundle: {archive_path}")


if __name__ == "__main__":
    main()
