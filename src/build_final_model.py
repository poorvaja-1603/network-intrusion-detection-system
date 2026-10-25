"""
build_final_model.py — Task 20: Train and save production pipeline

Creates:
  models/final.joblib  — End-to-end sklearn Pipeline accepting raw 41-feature DataFrame
  models/metrics.json  — Performance summary metrics on test set
  models/dummy.joblib  — Task 24 stub model for early testing
"""

from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from api.demo_model import DemoModel

from src.data import load_nsl_kdd
from src.preprocess import build_preprocessor
from src.evaluate import compute_metrics, Timer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_final_model():
    print("Building final production pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build and save dummy model (Task 24)
    dummy_pipe = DemoModel()
    dummy_path = MODELS_DIR / "dummy.joblib"
    joblib.dump(dummy_pipe, dummy_path)
    print(f"  [OK] Saved dummy model: {dummy_path.name}")

    # 2. Load dataset
    X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat = load_nsl_kdd()

    # 3. Create full end-to-end Pipeline
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("Fitting production pipeline on raw 41 features...")
    with Timer() as t:
        pipeline.fit(X_train, y_train_cat)
    print(f"  Fit completed in {t.elapsed:.2f}s")

    # 4. Evaluate on raw test set
    with Timer() as pt:
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
    print(f"  Test prediction completed in {pt.elapsed:.2f}s")

    metrics = compute_metrics(y_test_cat, y_pred, average="macro")
    metrics["fit_seconds"] = round(t.elapsed, 3)
    metrics["predict_seconds"] = round(pt.elapsed, 3)
    metrics["classes"] = list(pipeline.classes_)

    # 5. Save artifacts
    final_path = MODELS_DIR / "final.joblib"
    joblib.dump(pipeline, final_path)
    print(f"  [OK] Saved production model: {final_path.name}")

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Saved metrics summary: {metrics_path.name}")

    # 6. Verify pipeline contract
    print("\nVerifying Pipeline contract with a test row:")
    sample = X_test.iloc[:2]
    pred = pipeline.predict(sample)
    proba = pipeline.predict_proba(sample)
    print(f"  Sample inputs shape: {sample.shape}")
    print(f"  Predicted categories: {pred}")
    print(f"  Confidence max: {proba.max(axis=1)}")
    print("Contract verification passed!")


if __name__ == "__main__":
    build_final_model()

