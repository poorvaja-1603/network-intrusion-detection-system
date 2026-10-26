"""Task 21: export global feature importance from the saved final pipeline."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data import load_nsl_kdd
from src.preprocess import build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final.joblib"
FIGURE_PATH = ROOT / "reports" / "figures" / "importance.png"
DATA_PATH = ROOT / "results" / "feature_importance.json"


def export_feature_importance(top_n: int = 20) -> dict:
    model_source = "models/final.joblib"
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        # The checkpoint was written by scikit-learn 1.6.1; newer releases may
        # not unpickle its private ColumnTransformer internals. Recreate the
        # documented pipeline configuration in memory without replacing it.
        print(f"Could not load checkpoint ({exc}); recreating its documented pipeline in memory.")
        X_train, _, y_train_cat, _, _, _ = load_nsl_kdd()
        model = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=100, criterion="entropy", random_state=42, n_jobs=-1
            )),
        ])
        model.fit(X_train, y_train_cat)
        model_source = "retrained in memory from src/build_final_model.py configuration"
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    values = np.asarray(classifier.feature_importances_, dtype=float)
    if len(names) != len(values):
        raise ValueError(f"Feature names ({len(names)}) do not match importances ({len(values)})")

    order = np.argsort(values)[::-1][:top_n]
    ranked = [{"feature": names[i], "importance": round(float(values[i]), 8)} for i in order]
    result = {
        "model_path": "models/final.joblib",
        "importance_model_source": model_source,
        "method": "Random Forest mean decrease in impurity (global, not class-specific)",
        "classes": [str(value) for value in classifier.classes_],
        "features_ranked": ranked,
        "caveat": (
            "Impurity importance is a model-specific ranking, not evidence of causation. "
            "One-hot encoded categories appear as separate inputs."
        ),
    }

    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(result, indent=2) + "\n")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["feature"].replace("num__", "").replace("cat__", "") for row in ranked]
    chart_values = [row["importance"] for row in ranked]
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, chart_values, color="#31688e")
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Final Random Forest: top 20 global feature importances")
    ax.grid(axis="x", color="#d9e1e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_PATH} and {DATA_PATH}")
    return result


if __name__ == "__main__":
    export_feature_importance()
