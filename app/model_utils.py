import os
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load labeled feedback data from CSV."""
    return pd.read_csv(path)


def train_and_save(path: str = DATA_PATH,
                   model_path: str = MODEL_PATH) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train TF-IDF + Logistic Regression pipeline, evaluate it,
    then retrain on the full dataset and save the final model.

    Returns:
        pipeline: trained sklearn Pipeline
        stats: dict containing accuracy, confusion matrix, report, counts, etc.
    """
    df = load_data(path)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # ------------------------
    # 1) Train / test split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # ------------------------
    # 2) Build pipeline
    # ------------------------
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # ------------------------
    # 3) Train on train split
    # ------------------------
    pipeline.fit(X_train, y_train)

    # ------------------------
    # 4) Evaluation on test split
    # ------------------------
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ------------------------
    # 5) Retrain on full data for production model
    # ------------------------
    pipeline.fit(X, y)

    # ------------------------
    # 6) Save model
    # ------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(pipeline, model_path)

    # ------------------------
    # 7) Stats dict
    # ------------------------
    stats: Dict[str, Any] = {
        "total_samples": int(len(y)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "class_counts": dict(pd.Series(y).value_counts().to_dict()),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),          # 2D list for easy printing / JSON
        "classification_report": report,          # string table
    }

    return pipeline, stats


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """Load the saved model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run training first.")
    return load(model_path)


def predict_text(text: str, model: Pipeline) -> dict:
    """
    Predict label and probabilities for a single feedback text.
    Returns:
        {
            'label': predicted label (in Arabic),
            'probability': highest probability,
            'all': {label: prob, ...}
        }
    """
    proba = model.predict_proba([text])[0]
    labels = model.classes_.tolist()
    idx = int(proba.argmax())
    
    # Map numeric labels to Arabic
    label_map = {'0': 'رضا', '1': 'شكوى'}
    predicted_label = label_map.get(str(labels[idx]), str(labels[idx]))
    all_probs = {label_map.get(str(l), str(l)): float(p) for l, p in zip(labels, proba.tolist())}
    
    return {
        "label": predicted_label,
        "probability": float(proba[idx]),
        "all": all_probs,
    }
