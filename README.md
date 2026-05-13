# Hindi/Punjabi WhatsApp Fake News Detection

End-to-end fake-news detection for WhatsApp-style forwarded messages in Hindi, Punjabi, and mixed-script text.

The system includes:

- Prepared WhatsApp dataset splits in `data/whatsapp_dataset/`
- Kaggle training notebook in `notebooks/train_whatsapp_models_kaggle.ipynb`
- Scriptable training pipeline in `models/train_whatsapp_models.py`
- Optional stance analysis in `models/stance_module.py`
- FastAPI backend in `backend/app.py`
- Mobile-friendly single-file frontend in `frontend/index.html`
- Docker Compose deployment for backend + nginx frontend

## Current Model Bundle

The trained Kaggle bundle has been unpacked into:

```text
models/artifacts/
```

The original Kaggle ZIP is at:

```text
Output/whatsapp_model_artifacts.zip
```

Only deployable artifacts were extracted. The large `trainer_runs/` checkpoints were intentionally skipped.
For GitHub, the very large transformer `.safetensors` files and the original multi-GB Kaggle ZIP are intentionally ignored. Keep those in Kaggle, local storage, or another model-artifact store, then place them back under `models/artifacts/transformers/...` when full transformer inference is needed.

Verified test metrics from `models/artifacts/model_comparison.json`:

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9931 | 0.9931 | 0.9931 | 0.9931 |
| Naive Bayes | 0.9648 | 0.9665 | 0.9642 | 0.9648 |
| XGBoost | 0.9997 | 0.9997 | 0.9996 | 0.9997 |
| Punjabi BERT | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MuRIL | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

IndicBERT is not present because `ai4bharat/indic-bert` was gated during Kaggle training.

## Local Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

On macOS, XGBoost may require OpenMP:

```bash
brew install libomp
```

Open the frontend:

```bash
python3 -m http.server 3000 -d frontend
```

Then visit:

```text
http://localhost:3000
```

For transformer-backed model choices, enable transformer loading:

```bash
WHATSAPP_LOAD_TRANSFORMERS=1 uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The default API mode keeps transformer loading off for faster CPU startup. Selecting `muril` or `punjabi_bert` can still load that specific model on demand.

## Docker Compose

```bash
docker compose up
```

Services:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

The Compose file mounts the project directory into the backend container and serves `frontend/index.html` through nginx.

## Public Deployment

The app can deploy as one Docker web service. FastAPI serves both the API and the frontend at the same public URL.

Render setup:

1. Push this repo to GitHub.
2. Open Render and create a new Web Service from `Parth1353/major_project_final`.
3. Choose Docker runtime.
4. Keep the default Dockerfile.
5. Set health check path to `/health`.
6. Deploy.

Runtime defaults:

```text
WHATSAPP_LOAD_TRANSFORMERS=0
WHATSAPP_ENABLE_STANCE_MODEL=0
WHATSAPP_FAKE_THRESHOLD=0.5
```

The deployment uses the lightweight classical artifacts committed to GitHub. Full MuRIL/Punjabi transformer `.safetensors` files are not committed because they are too large for normal GitHub deployment.
The Dockerfile uses `requirements-deploy.txt`, which excludes training/GPU dependencies for a smaller production image.

## API

### `GET /health`

Returns model loading status and missing artifacts.

### `POST /predict/whatsapp`

Request:

```json
{
  "text": "जरूर शेयर करें! सरकार दे रही है...",
  "comments": ["यह सच है", "मुझे नहीं लगता"],
  "model": "ensemble"
}
```

Model choices:

```text
ensemble, muril, punjabi_bert, indic_bert, xgboost, logistic_regression, naive_bayes
```

Response:

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "fake_probability": 0.87,
  "model_scores": {
    "xgboost": 0.91,
    "muril": 0.82,
    "punjabi_bert": 0.88,
    "ensemble": 0.87
  },
  "selected_model": "ensemble",
  "stance_score": 0.3,
  "stance_counts": {
    "AGREE": 1,
    "DISAGREE": 1,
    "DISCUSS": 0,
    "UNRELATED": 0
  },
  "red_flags": ["forwarded_message", "share_urgency_words", "no_source"],
  "language_detected": "hi",
  "threshold_used": 0.5
}
```

### `POST /predict/batch`

Request:

```json
{
  "texts": ["message one", "message two"],
  "model": "ensemble"
}
```

Returns a `results` list with one prediction per text.

### `GET /model/compare`

Returns the saved `model_comparison.json`.

## Kaggle Training

Upload `whatsapp_dataset_for_kaggle.zip` to Kaggle as a dataset. It contains:

```text
whatsapp_dataset/whatsapp_train.csv
whatsapp_dataset/whatsapp_valid.csv
whatsapp_dataset/whatsapp_test.csv
```

Run:

```text
notebooks/train_whatsapp_models_kaggle.ipynb
```

The notebook writes:

```text
/kaggle/working/whatsapp_model_artifacts/
/kaggle/working/whatsapp_model_artifacts.zip
```

To retrain from the command line:

```bash
python models/train_whatsapp_models.py --data-dir data/whatsapp_dataset --output-dir models/artifacts
```

For a quick local smoke run:

```bash
python models/train_whatsapp_models.py --skip-transformers --sample-size 500
```

## Reliability Note

The current metrics are excellent on the prepared transformed dataset, but the dataset is synthetic/transformed. Before production use, validate on real WhatsApp forwards collected independently from the training generation process.
