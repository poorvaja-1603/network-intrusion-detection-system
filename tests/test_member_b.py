"""Member B integration checks for data splits, API, and final model.

The data checks run independently. API and model checks are skipped with an
explicit reason until Members C and A provide their contract implementations.
"""

import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import ALL_COLUMNS, DATA_DIR, FEATURE_COLUMNS, load_nsl_kdd


ROOT = Path(__file__).resolve().parents[1]
CLEAN_TEST_PATH = DATA_DIR / "KDDTest+_no_train_feature_overlap.txt"
AUDIT_PATH = DATA_DIR / "split_overlap_audit.json"


def test_data_shapes_and_feature_contract():
    X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat = load_nsl_kdd()

    assert X_train.shape == (125_973, 41)
    assert X_test.shape == (22_544, 41)
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS
    assert len(y_train_bin) == len(y_train_cat) == len(X_train)
    assert len(y_test_bin) == len(y_test_cat) == len(X_test)
    assert set(np.unique(y_train_bin)) <= {0, 1}
    assert set(np.unique(y_test_bin)) <= {0, 1}
    assert set(np.unique(y_train_cat)) == {"normal", "dos", "probe", "r2l", "u2r"}
    assert set(np.unique(y_test_cat)) == {"normal", "dos", "probe", "r2l", "u2r"}


def test_leakage_reduced_test_has_no_train_feature_overlap():
    assert CLEAN_TEST_PATH.exists(), "Run `python -m src.deduplicate_splits` first"
    train = pd.read_csv(DATA_DIR / "KDDTrain+.txt", header=None, names=ALL_COLUMNS)
    clean_test = pd.read_csv(CLEAN_TEST_PATH, header=None, names=ALL_COLUMNS)

    train_features = pd.MultiIndex.from_frame(train[FEATURE_COLUMNS].drop_duplicates())
    test_features = pd.MultiIndex.from_frame(clean_test[FEATURE_COLUMNS])
    assert len(clean_test) > 0
    assert not test_features.isin(train_features).any()

    audit = json.loads(AUDIT_PATH.read_text())
    assert audit["official_files_modified"] is False
    assert audit["remaining_feature_overlap_rows"] == 0
    assert audit["clean_test_rows"] == len(clean_test)


def _api_app_or_skip():
    try:
        module = importlib.import_module("api.main")
    except Exception as exc:  # pragma: no cover - depends on Member C's implementation
        pytest.skip(f"Member C API is not importable yet: {exc}")
    app = getattr(module, "app", None)
    if app is None:
        pytest.skip("Member C has not implemented the FastAPI app yet")
    return app


def test_api_health_and_prediction_contract():
    from fastapi.testclient import TestClient

    app = _api_app_or_skip()
    _, _, _, X_test, _, _ = load_nsl_kdd()
    flows = X_test.iloc[:5].to_dict(orient="records")
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    health = health_response.json()
    assert health.get("status") == "ok"
    assert "model_version" in health

    response = client.post("/predict", json={"flows": flows})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 5
    assert payload["latency_ms"] >= 0
    for prediction in payload["predictions"]:
        assert isinstance(prediction["label"], str)
        assert isinstance(prediction["is_attack"], bool)
        assert 0 <= prediction["confidence"] <= 1


def test_final_model_loads_and_predicts_on_five_raw_rows():
    model_path = ROOT / "models" / "final.joblib"
    if not model_path.exists():
        pytest.skip("Member A has not supplied models/final.joblib yet")

    _, _, _, X_test, _, _ = load_nsl_kdd()
    model = joblib.load(model_path)
    sample = X_test.iloc[:5]
    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert len(predictions) == 5
    assert probabilities.shape[0] == 5
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
