"""Manual checker for the WhatsApp fake-news backend.

Run this while the FastAPI backend is running on http://127.0.0.1:8000.

Examples:
  python3 check_model_results.py --sample
  python3 check_model_results.py --text "Forwarded many times..."
  python3 check_model_results.py --file message.txt --model ensemble
  python3 check_model_results.py --metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API = "http://127.0.0.1:8000"
SAMPLE_TEXT = """Forwarded many times
🚨 जरूर शेयर करें! सरकार दे रही है ₹50,000 सीधे बैंक खाते में।
अभी लिंक खोलें और सभी को भेजें।"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local WhatsApp fake-news model results.")
    parser.add_argument("--api", default=DEFAULT_API, help="Backend base URL.")
    parser.add_argument("--model", default="ensemble", help="ensemble, muril, punjabi_bert, xgboost, logistic_regression, naive_bayes")
    parser.add_argument("--text", help="Message text to check.")
    parser.add_argument("--file", type=Path, help="Text file containing the message to check.")
    parser.add_argument("--sample", action="store_true", help="Run a built-in fake-forward style sample.")
    parser.add_argument("--metrics", action="store_true", help="Print model comparison metrics.")
    parser.add_argument("--health", action="store_true", help="Print backend health.")
    parser.add_argument("--comment", action="append", default=[], help="Optional comment/reply. Can be passed multiple times.")
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach backend at {url}: {exc}") from exc


def get_message(args: argparse.Namespace) -> str | None:
    if args.sample:
        return SAMPLE_TEXT
    if args.text:
        return args.text
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        return text or None
    return None


def print_health(api: str) -> None:
    health = request_json("GET", f"{api}/health")
    print("\n=== Backend Health ===")
    print(f"Status: {health.get('status')}")
    print(f"Artifact dir: {health.get('artifact_dir')}")
    print(f"Stance model: {health.get('stance_model')}")
    print("Loaded models:")
    for name, loaded in health.get("loaded_models", {}).items():
        print(f"  - {name}: {'YES' if loaded else 'NO'}")
    errors = health.get("load_errors") or {}
    if errors:
        print("Load warnings:")
        for name, error in errors.items():
            first_line = str(error).splitlines()[0]
            print(f"  - {name}: {first_line[:140]}")


def print_metrics(api: str) -> None:
    comparison = request_json("GET", f"{api}/model/compare")
    print("\n=== Saved Test Metrics ===")
    rows = []
    for name, metrics in comparison.items():
        if not isinstance(metrics, dict) or "accuracy" not in metrics:
            continue
        rows.append(
            (
                name,
                metrics["accuracy"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
                metrics.get("confusion_matrix"),
            )
        )
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'F1':>9}  Confusion Matrix")
    for name, accuracy, precision, recall, f1, matrix in rows:
        print(f"{name:<22} {accuracy:>9.4f} {precision:>10.4f} {recall:>9.4f} {f1:>9.4f}  {matrix}")


def print_prediction(api: str, text: str, model: str, comments: list[str]) -> None:
    result = request_json(
        "POST",
        f"{api}/predict/whatsapp",
        {"text": text, "model": model, "comments": comments},
    )
    print("\n=== Prediction ===")
    print(f"Model requested: {model}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Fake probability: {result['fake_probability']:.4f}")
    print(f"Language detected: {result.get('language_detected')}")
    print(f"Threshold: {result.get('threshold_used')}")
    print(f"Stance score: {result.get('stance_score')}")
    print("\nModel scores:")
    for name, score in result.get("model_scores", {}).items():
        print(f"  - {name}: {score:.4f}")
    print("\nRed flags:")
    red_flags = result.get("red_flags") or []
    if red_flags:
        for flag in red_flags:
            print(f"  - {flag}")
    else:
        print("  - none")


def main() -> None:
    args = parse_args()
    api = args.api.rstrip("/")

    if args.health or not (args.metrics or args.text or args.file or args.sample or not sys.stdin.isatty()):
        print_health(api)
    if args.metrics:
        print_metrics(api)

    message = get_message(args)
    if message:
        print_prediction(api, message, args.model, args.comment)
    elif not args.metrics and not args.health:
        print("\nNo message provided. Use --sample, --text, --file, or pipe text through stdin.")


if __name__ == "__main__":
    main()

