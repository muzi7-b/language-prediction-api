"""Shared test fixtures.

Builds a tiny real model at test time (no network, no committed artifact) so the
API and model layers are tested against an actual sklearn Pipeline, and points
the app's model loader at it.
"""
import joblib
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# A handful of clearly-separable samples per language so predictions are stable.
_SAMPLES = [
    ("eng", "the department of transport will hold a public meeting next week"),
    ("eng", "please submit the completed application before the closing date"),
    ("afr", "ek het gister na die winkel toe gegaan om brood te koop"),
    ("afr", "die weer is vandag baie warm en sonnig"),
    ("zul", "ngiyabonga kakhulu ngosizo lwakho namhlanje"),
    ("zul", "sawubona unjani namhlanje ngiyajabula ukukubona"),
    ("sot", "re tla kopana hosane mosong ka nako ya borobong"),
    ("sot", "ke a leboha haholo ka thuso ya hao"),
]


def _build_pipeline() -> Pipeline:
    texts = [t for _, t in _SAMPLES]
    labels = [c for c, _ in _SAMPLES]
    pipe = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    pipe.fit(texts, labels)
    return pipe


@pytest.fixture()
def model_path(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(_build_pipeline(), path)
    return str(path)


@pytest.fixture()
def client(model_path, monkeypatch):
    # Point config + the lru_cache-backed loader at the temp model.
    from app import config, model as model_module

    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    model_module.load_model.cache_clear()

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c

    model_module.load_model.cache_clear()
