# South African Language Prediction API

A REST API that predicts which of South Africa's 11 official languages a piece of
text is written in, using a scikit-learn NLP model (CountVectorizer +
MultinomialNB). This is the service/software companion to the notebook model in
[LANGUAGE-PREDICTION](https://github.com/muzi7-b/LANGUAGE-PREDICTION) and the
SageMaker deployment in
[LANGUAGE-PREDICTION-SAGEMAKER](https://github.com/muzi7-b/LANGUAGE-PREDICTION-SAGEMAKER).

Languages: Afrikaans (afr), English (eng), isiNdebele (nbl), Sepedi (nso),
Sesotho (sot), siSwati (ssw), Setswana (tsn), Xitsonga (tso), Tshivenda (ven),
isiXhosa (xho), isiZulu (zul).

## Stack

FastAPI + Pydantic (validation) + scikit-learn, with pytest tests, a Dockerfile,
and GitHub Actions CI.

## Project layout

```
app/
  config.py     configuration from env (loopback-safe defaults)
  model.py      model loading + preprocessing + prediction
  schemas.py    request/response models with input validation
  main.py       FastAPI app: /health and /predict
scripts/
  train.py      train the model from a lang_id,text CSV
tests/          pytest suite (builds a tiny real model, no network)
Dockerfile
.github/workflows/ci.yml
```

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Train the model (CSV needs columns: lang_id,text)
python scripts/train.py --csv path/to/languages.csv   # writes model/model.joblib

# 2. Run the API (binds 127.0.0.1:8000 by default)
python -m app.main

# 3. Call it
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":["die weer is warm vandag","the meeting is next week"]}'
```

Response:

```json
{
  "predictions": [
    {"text": "die weer is warm vandag", "lang_id": "afr", "language": "Afrikaans"},
    {"text": "the meeting is next week", "lang_id": "eng", "language": "English"}
  ]
}
```

Interactive docs are at `http://127.0.0.1:8000/docs` once the server is running.

## Endpoints

| Method | Path       | Purpose |
|--------|------------|---------|
| GET    | `/health`  | Liveness + whether the model is loaded. |
| POST   | `/predict` | Classify one or more texts. Body: `{"text": ["...", "..."]}`. |

Validation: the batch must be a non-empty list (max `LANGID_MAX_BATCH_SIZE`,
default 100) of non-empty strings, each at most `LANGID_MAX_TEXT_LENGTH`
characters (default 5000). Invalid input returns HTTP 422; a missing model
returns HTTP 503.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `LANGID_HOST` | `127.0.0.1` | Bind address. Loopback by default. |
| `LANGID_PORT` | `8000` | Bind port. |
| `LANGID_MODEL_PATH` | `model/model.joblib` | Path to the trained model. |
| `LANGID_MAX_TEXT_LENGTH` | `5000` | Max characters per input string. |
| `LANGID_MAX_BATCH_SIZE` | `100` | Max texts per request. |

## Tests

```bash
pytest -q
```

The suite builds a small real sklearn Pipeline at runtime, so it needs no
network and no committed model artifact. CI runs it on Python 3.11 and 3.12.

## Docker

```bash
docker build -t language-prediction-api .
# Publish to loopback on the host (reachable only from this machine):
docker run -p 127.0.0.1:8000:8000 language-prediction-api
```

## A note on network exposure

This service binds to **127.0.0.1 by default** and is **unauthenticated by
design**: it is a public, read-only language-identification demo with no
restricted audience, no user data, and no secrets, so gating it would add no
security value. It is *not* intended to be reachable from other machines as
shipped.

If you want it reachable beyond your machine, that is a deliberate decision:
- Inside Docker the app binds `0.0.0.0` (correct — the container is the
  boundary), but **publish the port to `127.0.0.1` on the host** as shown above.
  A bare `-p 8000:8000` publishes on all host interfaces — avoid that unless you
  intend wide reachability.
- If you expose it to real users, put authentication in front of it (and, if the
  callers are a restricted audience, do not leave it open).

## Model

The model is trained from the NCHLT / EDSA "South African Language
Identification" dataset (`lang_id`, `text`). The repository does not ship the
dataset or the trained artifact; generate the model with `scripts/train.py`.
