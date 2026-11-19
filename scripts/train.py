"""Train the language-ID model and save it as model/model.joblib.

Usage:
    python scripts/train.py --csv path/to/languages.csv

The CSV must have columns: lang_id, text. This produces the same
CountVectorizer + MultinomialNB Pipeline the API serves.
"""
import argparse
import os
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in string.punctuation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns lang_id,text")
    parser.add_argument("--out", default="model/model.joblib")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if not {"lang_id", "text"}.issubset(df.columns):
        raise ValueError("CSV must have columns: lang_id, text")

    df["text"] = df["text"].str.lower().apply(remove_punctuation)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["lang_id"],
        test_size=args.test_size, random_state=args.random_state,
        stratify=df["lang_id"],
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    model.fit(X_train, y_train)
    print(classification_report(y_val, model.predict(X_val)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
