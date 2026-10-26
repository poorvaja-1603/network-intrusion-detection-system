"""FastAPI inference service and in-memory alert feed for NSL-KDD flows."""
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .demo_model import DemoModel, LABELS
from .schemas import PredictRequest, PredictResponse, Prediction
from src.data import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final.joblib"
DUMMY_PATH = ROOT / "models" / "dummy.joblib"
ALERT_LIMIT = 10000


def load_model():
    """Prefer the production pipeline, then saved dummy, then an in-memory stub."""
    for path in (MODEL_PATH, DUMMY_PATH):
        if path.is_file():
            try:
                return joblib.load(path), "v1" if path == MODEL_PATH else "dummy"
            except Exception as exc:
                if path == MODEL_PATH:
                    raise RuntimeError(f"Could not load model at {path}: {exc}") from exc
    return DemoModel(), "dummy"


model, model_version = load_model()
alerts: deque[dict[str, Any]] = deque(maxlen=ALERT_LIMIT)
label_counts: Counter[str] = Counter()
total_flows = 0
store_lock = Lock()
app = FastAPI(title="Network Intrusion Detection API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    global total_flows
    if not request.flows:
        return PredictResponse(predictions=[], latency_ms=0.0)
    frame = pd.DataFrame([flow.model_dump() for flow in request.flows], columns=FEATURE_COLUMNS)
    started = perf_counter()
    try:
        labels = model.predict(frame)
        probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model could not process input: {exc}") from exc
    elapsed = (perf_counter() - started) * 1000
    outputs: list[Prediction] = []
    now = datetime.now(timezone.utc).isoformat()
    with store_lock:
        for index, raw_label in enumerate(labels):
            label = str(raw_label).lower()
            is_attack = label != "normal"
            confidence = float(np.max(probabilities[index])) if probabilities is not None else 1.0
            outputs.append(Prediction(label=label, is_attack=is_attack, confidence=confidence))
            label_counts[label] += 1
            total_flows += 1
            if is_attack:
                alerts.appendleft({
                    "timestamp": now,
                    "label": label,
                    "confidence": round(confidence, 6),
                    "flow": request.flows[index].model_dump(),
                })
    return PredictResponse(predictions=outputs, latency_ms=round(elapsed, 3))


@app.get("/alerts")
def get_alerts(limit: int = Query(default=100, ge=1, le=5000)):
    with store_lock:
        return {"alerts": list(alerts)[:limit], "total_alerts": len(alerts)}


@app.post("/reset")
def reset_stats():
    global total_flows
    with store_lock:
        alerts.clear()
        label_counts.clear()
        total_flows = 0
    return {"status": "ok", "message": "All alerts and traffic statistics reset successfully"}


@app.get("/timeline")
def get_timeline(
    minutes: int = Query(default=10, ge=1, le=60),
    seconds: int = Query(default=60, ge=5, le=300),
    granularity: str = Query(default="seconds")
):
    now = datetime.now(timezone.utc)
    if granularity == "seconds":
        now_sec = now.replace(microsecond=0)
        counts = {
            (now_sec - timedelta(seconds=i)).isoformat(): 0
            for i in reversed(range(seconds))
        }
        with store_lock:
            alert_snapshot = list(alerts)
        for alert in alert_snapshot:
            try:
                ts = datetime.fromisoformat(alert["timestamp"]).astimezone(timezone.utc).replace(microsecond=0).isoformat()
                if ts in counts:
                    counts[ts] += 1
            except Exception:
                pass
        timeline_list = [{"timestamp": k, "count": v} for k, v in counts.items()]
        return {"minutes": timeline_list, "timeline": timeline_list, "granularity": "seconds"}
    else:
        first_minute = now.replace(second=0, microsecond=0) - timedelta(minutes=minutes - 1)
        counts = {
            (first_minute + timedelta(minutes=index)).isoformat(): 0
            for index in range(minutes)
        }
        with store_lock:
            alert_snapshot = list(alerts)
        for alert in alert_snapshot:
            try:
                timestamp = datetime.fromisoformat(alert["timestamp"]).astimezone(timezone.utc)
                bucket = timestamp.replace(second=0, microsecond=0).isoformat()
                if bucket in counts:
                    counts[bucket] += 1
            except Exception:
                pass
        timeline_list = [{"timestamp": timestamp, "count": count} for timestamp, count in counts.items()]
        return {"minutes": timeline_list, "timeline": timeline_list, "granularity": "minutes"}


@app.get("/stats")
def get_stats():
    with store_lock:
        snapshot = dict(label_counts)
        count = total_flows
    attacks = count - snapshot.get("normal", 0)
    return {
        "counts_per_label": {label: snapshot.get(label, 0) for label in LABELS},
        "total_flows": count,
        "alert_rate": round(attacks / count, 6) if count else 0.0,
    }
