"""FastAPI application: local REST service for South African language ID.

Security posture (see app/config.py):
- Public, unauthenticated demo by design — no restricted audience, no user data,
  no secrets — so it is intentionally not gated.
- Binds to 127.0.0.1 by default; override LANGID_HOST only when you intend wider
  reachability (e.g. 0.0.0.0 inside a container whose published port is mapped to
  127.0.0.1 on the host).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from . import config
from .model import load_model
from .schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger("language-prediction-api")

app = FastAPI(
    title="South African Language Prediction API",
    description="Predicts which of South Africa's 11 official languages a text is written in.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = load_model()
    except FileNotFoundError as e:
        # Model missing is an operational condition, not a bad request.
        raise HTTPException(status_code=503, detail=str(e))
    predictions = model.predict(request.text)
    return PredictResponse(predictions=predictions)


def run() -> None:
    """Entry point for `python -m app.main`. Binds per config (loopback default)."""
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    run()
