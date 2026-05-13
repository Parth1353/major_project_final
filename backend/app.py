from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference_utils import METADATA_FEATURES, feature_vector, heuristic_fake_probability, preprocess_whatsapp
from models.stance_module import StanceAnalyzer, combine_model_and_stance_scores


ModelName = Literal["ensemble", "muril", "indic_bert", "punjabi_bert", "xgboost", "logistic_regression", "naive_bayes"]
ARTIFACT_DIR = Path(os.getenv("WHATSAPP_MODEL_DIR", PROJECT_ROOT / "models" / "artifacts"))
LOAD_TRANSFORMERS = os.getenv("WHATSAPP_LOAD_TRANSFORMERS", "0") == "1"
ENABLE_STANCE_MODEL = os.getenv("WHATSAPP_ENABLE_STANCE_MODEL", "0") == "1"
DEFAULT_THRESHOLD = float(os.getenv("WHATSAPP_FAKE_THRESHOLD", "0.5"))
MAX_BATCH_SIZE = int(os.getenv("WHATSAPP_MAX_BATCH_SIZE", "100"))


class WhatsAppPredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    comments: list[str] | None = None
    model: ModelName = "ensemble"


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    model: ModelName = "ensemble"


class FakeNewsModelService:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.classical: dict[str, Any] = {}
        self.transformers: dict[str, dict[str, Any]] = {}
        self.ensemble: Any | None = None
        self.ensemble_feature_names: list[str] = []
        self.model_comparison: dict[str, Any] = {}
        self.training_config: dict[str, Any] = {}
        self.load_errors: dict[str, str] = {}
        self.stance_analyzer = StanceAnalyzer(enable_model=ENABLE_STANCE_MODEL)
        self._load_all()

    def _load_all(self) -> None:
        if not self.artifact_dir.exists():
            self.load_errors["artifacts"] = f"Artifact directory not found: {self.artifact_dir}"
            return
        self._load_json_files()
        self._load_classical()
        self._load_ensemble()
        if LOAD_TRANSFORMERS:
            self._load_transformers()

    def _load_json_files(self) -> None:
        for name, attr in [("model_comparison.json", "model_comparison"), ("training_config.json", "training_config")]:
            path = self.artifact_dir / name
            if path.exists():
                setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))

    def _load_classical(self) -> None:
        classical_dir = self.artifact_dir / "classical"
        paths = {
            "tfidf_vectorizer": classical_dir / "tfidf_vectorizer.joblib",
            "metadata_scaler": classical_dir / "metadata_scaler.joblib",
            "logistic_regression": classical_dir / "logistic_regression.joblib",
            "naive_bayes": classical_dir / "naive_bayes.joblib",
            "xgboost": classical_dir / "xgboost.joblib",
        }
        for name, path in paths.items():
            if not path.exists():
                self.load_errors[name] = f"Missing {path}"
                continue
            try:
                self.classical[name] = joblib.load(path)
            except Exception as exc:
                self.load_errors[name] = repr(exc)

    def _load_ensemble(self) -> None:
        ensemble_path = self.artifact_dir / "ensemble" / "stacking_xgboost.joblib"
        feature_path = self.artifact_dir / "ensemble" / "stacking_feature_config.json"
        if ensemble_path.exists():
            try:
                self.ensemble = joblib.load(ensemble_path)
            except Exception as exc:
                self.load_errors["stacking_ensemble"] = repr(exc)
        else:
            self.load_errors["stacking_ensemble"] = f"Missing {ensemble_path}"
        if feature_path.exists():
            self.ensemble_feature_names = json.loads(feature_path.read_text(encoding="utf-8")).get("feature_names", [])

    def _load_transformers(self) -> None:
        for alias in ["muril", "punjabi_bert", "indic_bert"]:
            self._load_transformer_alias(alias)

    def _load_transformer_alias(self, alias: str) -> bool:
        if alias in self.transformers:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            self.load_errors["transformers"] = f"transformers/torch unavailable: {exc!r}"
            return False

        transformer_dir = self.artifact_dir / "transformers"
        path = transformer_dir / alias
        if not path.exists():
            self.load_errors[alias] = f"Missing {path}"
            return False
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to("cpu")
            model.eval()
            self.transformers[alias] = {"tokenizer": tokenizer, "model": model, "torch": torch}
            self.load_errors.pop(alias, None)
            return True
        except Exception as exc:
            self.load_errors[alias] = repr(exc)
            return False

    def health(self) -> dict[str, Any]:
        loaded = {
            "logistic_regression": "logistic_regression" in self.classical,
            "naive_bayes": "naive_bayes" in self.classical,
            "xgboost": "xgboost" in self.classical,
            "ensemble": self.ensemble is not None,
            "muril": "muril" in self.transformers,
            "punjabi_bert": "punjabi_bert" in self.transformers,
            "indic_bert": "indic_bert" in self.transformers,
        }
        return {
            "status": "ok" if any(loaded.values()) else "degraded",
            "artifact_dir": str(self.artifact_dir),
            "load_transformers": LOAD_TRANSFORMERS,
            "stance_model": self.stance_analyzer.used_model,
            "loaded_models": loaded,
            "load_errors": self.load_errors,
        }

    def predict(self, text: str, comments: list[str] | None = None, requested_model: str = "ensemble") -> dict[str, Any]:
        if requested_model not in {"ensemble", "muril", "indic_bert", "punjabi_bert", "xgboost", "logistic_regression", "naive_bayes"}:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {requested_model}")
        if requested_model in {"muril", "indic_bert", "punjabi_bert"}:
            self._load_transformer_alias(requested_model)
        processed = preprocess_whatsapp(text)
        model_scores = self._score_models(processed)
        selected_score = self._select_score(model_scores, requested_model, text)
        stance = self.stance_analyzer.analyze(text, comments)
        final_score = combine_model_and_stance_scores(selected_score, stance.stance_score) if comments else selected_score
        prediction = "FAKE" if final_score >= DEFAULT_THRESHOLD else "REAL"
        confidence = final_score if prediction == "FAKE" else 1.0 - final_score
        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "fake_probability": round(float(final_score), 4),
            "model_scores": {name: round(float(score), 4) for name, score in model_scores.items()},
            "selected_model": requested_model,
            "stance_score": round(float(stance.stance_score), 4),
            "stance_counts": stance.stance_counts,
            "red_flags": processed.red_flags,
            "language_detected": processed.language_detected,
            "threshold_used": DEFAULT_THRESHOLD,
        }

    def _score_models(self, processed: Any) -> dict[str, float]:
        scores: dict[str, float] = {}
        if {"tfidf_vectorizer", "logistic_regression", "naive_bayes"}.issubset(self.classical):
            tfidf = self.classical["tfidf_vectorizer"].transform([processed.clean_text])
            for name in ["logistic_regression", "naive_bayes"]:
                scores[name] = self._fake_probability(self.classical[name], tfidf)
        if {"tfidf_vectorizer", "metadata_scaler", "xgboost"}.issubset(self.classical):
            tfidf = self.classical["tfidf_vectorizer"].transform([processed.clean_text])
            meta = self._scaled_metadata(processed)
            x_matrix = sparse.hstack([tfidf, sparse.csr_matrix(meta)], format="csr")
            scores["xgboost"] = self._fake_probability(self.classical["xgboost"], x_matrix)
        for alias in ["muril", "punjabi_bert", "indic_bert"]:
            if alias in self.transformers:
                scores[alias] = self._score_transformer(alias, processed.transformer_text)
        if self.ensemble is not None and self.ensemble_feature_names:
            scores["ensemble"] = self._score_ensemble(processed, scores)
        return scores

    def _scaled_metadata(self, processed: Any) -> np.ndarray:
        raw = np.array([feature_vector(processed.features)], dtype=float)
        scaler = self.classical.get("metadata_scaler")
        return scaler.transform(raw) if scaler is not None else raw

    def _fake_probability(self, estimator: Any, x_matrix: Any) -> float:
        probabilities = estimator.predict_proba(x_matrix)
        classes = list(estimator.classes_)
        return float(probabilities[0, classes.index(1)])

    def _score_transformer(self, alias: str, transformer_text: str) -> float:
        bundle = self.transformers[alias]
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        torch = bundle["torch"]
        inputs = tokenizer(
            transformer_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(self.training_config.get("max_length", 128)),
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(probs[1])

    def _score_ensemble(self, processed: Any, base_scores: dict[str, float]) -> float:
        scaled_meta = dict(zip(METADATA_FEATURES, self._scaled_metadata(processed)[0]))
        fallback_score = base_scores.get("xgboost", base_scores.get("logistic_regression", heuristic_fake_probability(processed.original_text)))
        values = []
        for name in self.ensemble_feature_names:
            if name.endswith("_fake_prob"):
                base_name = name.removesuffix("_fake_prob")
                values.append(float(base_scores.get(base_name, fallback_score)))
            else:
                values.append(float(scaled_meta.get(name, processed.features.get(name, 0.0))))
        return self._fake_probability(self.ensemble, np.array([values], dtype=float))

    def _select_score(self, scores: dict[str, float], requested_model: str, text: str) -> float:
        if requested_model in scores:
            return scores[requested_model]
        if requested_model == "ensemble" and scores:
            return float(np.mean(list(scores.values())))
        for candidate in ["ensemble", "xgboost", "logistic_regression", "naive_bayes"]:
            if candidate in scores:
                return scores[candidate]
        if scores:
            return float(np.mean(list(scores.values())))
        return heuristic_fake_probability(text)


service = FakeNewsModelService(ARTIFACT_DIR)
app = FastAPI(title="Hindi/Punjabi WhatsApp Fake News Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WHATSAPP_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return service.health()


@app.post("/predict/whatsapp")
def predict_whatsapp(request: WhatsAppPredictRequest) -> dict[str, Any]:
    return service.predict(request.text, comments=request.comments, requested_model=request.model)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    return {
        "count": len(request.texts),
        "results": [service.predict(text, requested_model=request.model) for text in request.texts],
    }


@app.get("/model/compare")
def model_compare() -> dict[str, Any]:
    if not service.model_comparison:
        raise HTTPException(status_code=404, detail="model_comparison.json not found")
    return service.model_comparison
