"""API behavior tests: health, prediction, and input validation."""


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single(client):
    r = client.post("/predict", json={"text": ["the meeting is next week"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["lang_id"] == "eng"
    assert preds[0]["language"] == "English"
    assert preds[0]["text"] == "the meeting is next week"


def test_predict_batch_multiple_languages(client):
    r = client.post("/predict", json={"text": [
        "the department will hold a meeting",
        "ek is baie honger vandag",
        "re tla kopana hosane mosong",
    ]})
    assert r.status_code == 200
    codes = [p["lang_id"] for p in r.json()["predictions"]]
    assert codes == ["eng", "afr", "sot"]


def test_predict_empty_list_rejected(client):
    r = client.post("/predict", json={"text": []})
    assert r.status_code == 422  # fails min_length=1


def test_predict_empty_string_rejected(client):
    r = client.post("/predict", json={"text": ["   "]})
    assert r.status_code == 422


def test_predict_missing_field_rejected(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_overlong_text_rejected(client):
    from app import config
    r = client.post("/predict", json={"text": ["a" * (config.MAX_TEXT_LENGTH + 1)]})
    assert r.status_code == 422


def test_predict_batch_too_large_rejected(client):
    from app import config
    r = client.post("/predict", json={"text": ["hello"] * (config.MAX_BATCH_SIZE + 1)})
    assert r.status_code == 422
