"""Request/response schemas with input validation (Pydantic v2)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

from . import config


class PredictRequest(BaseModel):
    # Accepts a non-empty list of non-empty strings, bounded in count and length.
    text: List[str] = Field(..., min_length=1, max_length=config.MAX_BATCH_SIZE,
                            description="One or more texts to classify.")

    @field_validator("text")
    @classmethod
    def _validate_items(cls, values: List[str]) -> List[str]:
        cleaned = []
        for i, v in enumerate(values):
            if not isinstance(v, str):
                raise ValueError(f"text[{i}] must be a string")
            stripped = v.strip()
            if not stripped:
                raise ValueError(f"text[{i}] must not be empty or whitespace")
            if len(v) > config.MAX_TEXT_LENGTH:
                raise ValueError(
                    f"text[{i}] exceeds max length {config.MAX_TEXT_LENGTH}"
                )
            cleaned.append(v)
        return cleaned


class Prediction(BaseModel):
    text: str
    lang_id: str
    language: str


class PredictResponse(BaseModel):
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
