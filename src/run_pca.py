"""Task 10: evaluate PCA with and without scaling on NSL-KDD.

PCA and preprocessing are fitted on KDDTrain+ only. KDDTest+ is used once as
the fixed evaluation split. The unscaled branch one-hot encodes categorical
features but leaves the 38 numeric features in their original units.

Run with ``python -m src.run_pca``.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from src.data import CATEGORICAL_COLUMNS, get_numeric_columns, load_nsl_kdd
from src.models import get_paper_models
from src.preprocess import build_preprocessor


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "results" / "exp2_pca.json"
FIGURE_DIR = ROOT / "reports" / "figures"
SEED = 42
DIMS = (1, 2, 3, 4, 5, 10, 15, 20, 30)


def _unscaled_encode(X_train, X_test):
    """One-hot encode categoricals while preserving numeric feature units."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    test_cat = encoder.transform(X_test[CATEGORICAL_COLUMNS])
    train_num = X_train[get_numeric_columns()].to_numpy(dtype=float)
    test_num = X_test[get_numeric_columns()].to_numpy(dtype=float)
    return np.hstack((train_num, train_cat)), np.hstack((test_num, test_cat))


def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def run_pca_experiment(output_path: Path = RESULT_PATH) -> dict:
    X_train, y_train, _, X_test, y_test, _ = load_nsl_kdd()
    scaled_preprocessor = build_preprocessor()
    train_scaled = scaled_preprocessor.fit_transform(X_train)
    test_scaled = scaled_preprocessor.transform(X_test)
    train_unscaled, test_unscaled = _unscaled_encode(X_train, X_test)

    datasets = {
        "scaled": (train_scaled, test_scaled),
        "unscaled": (train_unscaled, test_unscaled),
    }
    models = get_paper_models()
    rows = []
    variance = {}

    for preprocessing, (train_features, test_features) in datasets.items():
        max_dims = min(max(DIMS), train_features.shape[1])
        pca = PCA(n_components=max_dims, svd_solver="full", random_state=SEED)
        started = time.perf_counter()
        train_projected = pca.fit_transform(train_features)
        pca_fit_seconds = time.perf_counter() - started
        test_projected = pca.transform(test_features)
        variance[preprocessing] = {
            "input_features": int(train_features.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "first_component_variance": float(pca.explained_variance_ratio_[0]),
            "pca_fit_seconds": pca_fit_seconds,
        }

        for dims in DIMS:
            if dims > train_projected.shape[1]:
                continue
            for model_name, prototype in models.items():
                model = prototype
                started = time.perf_counter()
                model.fit(train_projected[:, :dims], y_train)
                fit_seconds = time.perf_counter() - started
                predicted = model.predict(test_projected[:, :dims])
                rows.append({
                    "experiment": "pca_sweep",
                    "model": model_name,
                    "setting": "pca",
                    "preprocessing": preprocessing,
                    "n_dims": dims,
                    "seed": SEED,
                    "fit_seconds": fit_seconds,
                    **_metrics(y_test, predicted),
                })
                print(f"{preprocessing:8} PCA {dims:2} {model_name:12} accuracy={rows[-1]['accuracy']:.4f}")

    result = {
        "experiment": "pca_scaled_vs_unscaled",
        "seed": SEED,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_split": "official KDDTest+; fixed evaluation only",
        "preprocessing_note": (
            "Scaled: numeric columns standardized and categoricals one-hot encoded. "
            "Unscaled: numeric columns left in original units and categoricals one-hot encoded. "
            "Both preprocessors and PCA transforms are fitted on KDDTrain+ only."
        ),
        "first_component_claim": {
            "paper_reported_percent": 92.0,
            "scaled_percent": variance["scaled"]["first_component_variance"] * 100,
            "unscaled_percent": variance["unscaled"]["first_component_variance"] * 100,
            "interpretation": "Variance explained is descriptive and does not establish classification quality.",
        },
        "variance": variance,
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    _plot_variance(variance)
    _plot_accuracy(rows)
    return result


def _plot_variance(variance):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in (("scaled", "Scaled numeric features", "#355C7D"),
                              ("unscaled", "Unscaled numeric features", "#F2A65A")):
        vals = np.asarray(variance[key]["cumulative_explained_variance"])
        ax.plot(np.arange(1, len(vals) + 1), vals, marker=".", label=label, color=color)
    ax.axhline(0.92, color="#B56576", linestyle="--", label="Paper claim: 92% in PC1")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative variance explained", ylim=(0, 1.02),
           title="PCA variance: scaled vs unscaled features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_explained_variance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy(rows):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, preprocessing in zip(axes, ("scaled", "unscaled")):
        for model, color in (("NaiveBayes", "#355C7D"), ("J48", "#C06C84"), ("RandomForest", "#6C9A8B")):
            selected = sorted((r for r in rows if r["preprocessing"] == preprocessing and r["model"] == model),
                              key=lambda r: r["n_dims"])
            ax.plot([r["n_dims"] for r in selected], [r["accuracy"] for r in selected],
                    marker="o", label=model, color=color)
        ax.set(title=f"{preprocessing.title()} numeric features", xlabel="PCA components", xticks=list(DIMS))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("KDDTest+ accuracy")
    axes[1].legend()
    fig.suptitle("Classification accuracy across PCA dimensions")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pca_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    result = run_pca_experiment()
    claim = result["first_component_claim"]
    print(f"PC1 variance: scaled={claim['scaled_percent']:.2f}%, unscaled={claim['unscaled_percent']:.2f}%")
    print(f"Saved {RESULT_PATH}")
