import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from .model_utils import (
    MODEL_PATH,
    DATA_PATH,
    train_and_save,
    load_model,
    predict_text,
    load_data
)

import csv
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SUBMISSIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'submissions.csv'
)


# ===============================
# Submissions helpers
# ===============================

def ensure_submissions_file():
    os.makedirs(os.path.dirname(SUBMISSIONS_PATH), exist_ok=True)
    if not os.path.exists(SUBMISSIONS_PATH):
        with open(SUBMISSIONS_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'text', 'label', 'probability'])


def append_submission(text: str, label: str, probability: float):
    ensure_submissions_file()
    with open(SUBMISSIONS_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            text,
            label,
            float(probability)
        ])


def read_submissions(limit: int = 200):
    ensure_submissions_file()
    rows = []
    with open(SUBMISSIONS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return list(reversed(rows))[:limit]


def delete_submissions_by_timestamps(timestamps: list) -> int:
    ensure_submissions_file()
    timestamps_set = set(timestamps)
    kept = []
    removed = 0

    with open(SUBMISSIONS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('timestamp') in timestamps_set:
                removed += 1
            else:
                kept.append(r)

    with open(SUBMISSIONS_PATH, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['timestamp', 'text', 'label', 'probability']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)

    return removed


def clear_submissions() -> int:
    ensure_submissions_file()
    with open(SUBMISSIONS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    removed = len(rows)

    with open(SUBMISSIONS_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'text', 'label', 'probability'])

    return removed


# ===============================
# FastAPI app
# ===============================

app = FastAPI(title='Arabic Feedback Analyzer')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')


class PredictRequest(BaseModel):
    text: str
    save: bool = False


@app.on_event('startup')
def startup_event():
    try:
        load_model()
    except Exception:
        train_and_save()


# ===============================
# Prediction
# ===============================

@app.post('/predict')
def predict(req: PredictRequest) -> Any:
    try:
        model = load_model()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail='Model not found. Train first.')

    res = predict_text(req.text, model)

    if req.save:
        try:
            append_submission(req.text, res['label'], res['probability'])
        except Exception:
            pass

    return {'ok': True, 'input': req.text, 'result': res}


# ===============================
# Dataset stats
# ===============================

@app.get('/stats')
def stats():
    df = load_data()
    counts = df['label'].value_counts().to_dict()

    label_map = {'0': 'رضا', '1': 'شكوى'}
    counts_arabic = {
        label_map.get(str(k), str(k)): int(v)
        for k, v in counts.items()
    }

    total = int(df.shape[0])
    return {'ok': True, 'total': total, 'counts': counts_arabic}


# ===============================
# Model accuracy (NEW)
# ===============================

@app.get('/accuracy')
def get_model_accuracy():
    df = load_data()
    X = df['text']
    y = df['label'].astype(str)  # تحويل إلى نص

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = load_model()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    
    # طباعة الدقة في الـ terminal
    print(f"دقة النموذج: {round(acc * 100, 2)}%")

    return {
        'ok': True,
        'accuracy': round(acc * 100, 2)
    }


# ===============================
# Training
# ===============================

@app.post('/train')
def train():
    pipeline, stats = train_and_save()
    return {'ok': True, 'stats': stats}


# ===============================
# Submissions APIs
# ===============================

@app.get('/submissions')
def submissions(limit: int = 200):
    try:
        rows = read_submissions(limit)
        return {'ok': True, 'total': len(rows), 'rows': rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/submissions/delete')
def delete_submissions(payload: dict):
    timestamps = payload.get('timestamps')
    if not timestamps or not isinstance(timestamps, list):
        raise HTTPException(
            status_code=400,
            detail='Provide JSON with timestamps: [..]'
        )

    try:
        removed = delete_submissions_by_timestamps(timestamps)
        return {'ok': True, 'removed': removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/submissions/clear')
def clear_all_submissions():
    try:
        removed = clear_submissions()
        return {'ok': True, 'removed': removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/submissions/download')
def download_submissions():
    try:
        ensure_submissions_file()
        return FileResponse(
            SUBMISSIONS_PATH,
            media_type='text/csv',
            filename='submissions.csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
