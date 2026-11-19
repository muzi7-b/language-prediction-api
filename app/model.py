"""Model loading and prediction.

The model is a scikit-learn Pipeline (CountVectorizer + MultinomialNB) trained to
classify text into one of South Africa's 11 official languages. The same artifact
is used by the SageMaker deployment companion project; here it is served locally.

If the model file is absent, load_model raises FileNotFoundError with guidance,
and the API reports 503 (model not loaded) rather than returning wrong answers.
"""
from __future__ import annotations

import os
import string
from functools import lru_cache
from typing import Optional

import joblib

from . import config

LANG_NAMES = {
    "afr": "Afrikaans", "eng": "English", "nbl": "isiNdebele",
    "nso": "Sepedi", "sot": "Sesotho", "ssw": "siSwati",
    "tsn": "Setswana", "tso": "Xitsonga", "ven": "Tshivenda",
    "xho": "isiXhosa", "zul": "isiZulu",
}


def _preprocess(text: str) -> str:
    """Match the training-time preprocessing: lowercase, strip punctuation."""
    lowered = text.lower()
    return "".join(ch for ch in lowered if ch not in string.punctuation)


class LanguageModel:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def predict(self, texts: list[str]) -> list[dict]:
        cleaned = [_preprocess(t) for t in texts]
        preds = self._pipeline.predict(cleaned)
        results = []
        for original, code in zip(texts, preds):
            code = str(code)
            results.append({
                "text": original,
                "lang_id": code,
                "language": LANG_NAMES.get(code, code),
            })
        return results


@lru_cache(maxsize=1)
def load_model(path: Optional[str] = None) -> LanguageModel:
    path = path or config.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Train it first "
            "(see README: scripts/train.py) or set LANGID_MODEL_PATH."
        )
    pipeline = joblib.load(path)
    return LanguageModel(pipeline)
