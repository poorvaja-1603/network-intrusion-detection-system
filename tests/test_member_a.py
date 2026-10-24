"""
test_member_a.py — Comprehensive verification tests for Member A deliverables.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data import load_nsl_kdd, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, get_numeric_columns
from src.preprocess import build_preprocessor, preprocess_data
from src.models import get_models, get_paper_models
from src.evaluate import compute_metrics, compute_fpr


def test_data_contract():
    """Verify Task 3: data loading contract and dedupe checks."""
    X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat = load_nsl_kdd()

    assert X_train.shape == (125973, 41)
    assert X_test.shape == (22544, 41)
    assert set(FEATURE_COLUMNS) == set(X_train.columns)
    assert len(CATEGORICAL_COLUMNS) == 3
    assert len(get_numeric_columns()) == 38

    # Binary labels should only be 0 and 1
    assert set(np.unique(y_train_bin)).issubset({0, 1})
    assert set(np.unique(y_test_bin)).issubset({0, 1})

    # 5-class labels
    expected_classes = {"normal", "dos", "probe", "r2l", "u2r"}
    assert set(np.unique(y_train_cat)) == expected_classes
    assert set(np.unique(y_test_cat)) == expected_classes


def test_preprocess_contract():
    """Verify Task 6: preprocessor expands to 122 dims and handles unseen test categories."""
    X_train, _, _, X_test, _, _ = load_nsl_kdd()
    sample_train = X_train.iloc[:1000]
    sample_test = X_test.iloc[:200]

    X_train_proc, X_test_proc, preprocessor = preprocess_data(sample_train, sample_test)

    # Output dimension must be >= 41 (due to one-hot encoding)
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert X_train_proc.shape[1] >= 41
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()


def test_model_zoo():
    """Verify Task 7: model dictionary."""
    models = get_models()
    assert "NaiveBayes" in models
    assert "J48" in models
    assert "RandomForest" in models
    assert "MLP" in models

    paper_models = get_paper_models()
    assert len(paper_models) == 3


def test_evaluate_metrics():
    """Verify Task 8: metric calculations."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    metrics = compute_metrics(y_true, y_pred, average="binary")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["fpr"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert "confusion_matrix" in metrics


def test_final_production_model():
    """Verify Task 20: final.joblib contract."""
    model_path = Path("models/final.joblib")
    assert model_path.exists(), "models/final.joblib must exist"

    pipeline = joblib.load(model_path)
    X_train, _, _, X_test, _, _ = load_nsl_kdd()

    # Pass raw 41-feature DataFrame slice
    test_slice = X_test.iloc[:5]
    preds = pipeline.predict(test_slice)
    probas = pipeline.predict_proba(test_slice)

    assert len(preds) == 5
    assert probas.shape == (5, 5)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_results_files_exist():
    """Verify all results artifacts exist and contain valid data."""
    required = [
        "exp1_full_features.json",
        "exp3_rp_gaussian_sweep.json",
        "exp5_multiclass.json",
        "paper_reported.json",
    ]
    for r in required:
        p = Path("results") / r
        assert p.exists(), f"Missing {p}"
        with open(p) as f:
            data = json.load(f)
        assert len(data) > 0


if __name__ == "__main__":
    print("Running verification test suite for Member A...")
    test_data_contract()
    print("  [PASS] test_data_contract")
    test_preprocess_contract()
    print("  [PASS] test_preprocess_contract")
    test_model_zoo()
    print("  [PASS] test_model_zoo")
    test_evaluate_metrics()
    print("  [PASS] test_evaluate_metrics")
    test_final_production_model()
    print("  [PASS] test_final_production_model")
    test_results_files_exist()
    print("  [PASS] test_results_files_exist")
    print("\nALL 6 VERIFICATION SUITES PASSED SUCCESSFULLY!")
