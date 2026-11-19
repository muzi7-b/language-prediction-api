"""Runtime configuration, read from environment variables.

Security note: HOST defaults to 127.0.0.1 (loopback). This service is a public,
unauthenticated language-identification demo with no restricted audience and no
secrets, so it is not gated — but it still binds to loopback by default so it is
not reachable beyond this machine unless the operator explicitly opts in by
setting LANGID_HOST (e.g. to 0.0.0.0 inside a container whose published port is
mapped to 127.0.0.1 on the host).
"""
import os

# Bind address. Default loopback-only. Override with LANGID_HOST.
HOST = os.environ.get("LANGID_HOST", "127.0.0.1")
PORT = int(os.environ.get("LANGID_PORT", "8000"))

# Path to the trained model artifact (joblib Pipeline: CountVectorizer + MultinomialNB).
MODEL_PATH = os.environ.get("LANGID_MODEL_PATH", "model/model.joblib")

# Max characters accepted per input string (defensive input bound).
MAX_TEXT_LENGTH = int(os.environ.get("LANGID_MAX_TEXT_LENGTH", "5000"))
# Max number of texts per batch request.
MAX_BATCH_SIZE = int(os.environ.get("LANGID_MAX_BATCH_SIZE", "100"))
