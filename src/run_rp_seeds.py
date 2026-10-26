"""Task 14: measure random-projection seed variability at sweep-best dimensions.

For each model and projection type, use the dimension with the highest accuracy
in the corresponding single-seed sweep, then repeat that setting with 10 seeds.
The overlap-clean test file removes exact train/test feature duplicates, but
the same evaluation population informed dimension selection in the sweeps, so
these repeated scores are descriptive and not an unbiased final test estimate.

Run with ``python -m src.run_rp_seeds --repeats 10``.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from src.data import load_nsl_kdd
from src.models import get_paper_models
from src.preprocess import build_preprocessor


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "results" / "exp3_rp_seeds.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "rp_seed_variance.png"
PROJECTIONS = {
    "rp_gaussian": GaussianRandomProjection,
    "rp_sparse": SparseRandomProjection,
}
SWEEP_PATHS = {
    "rp_gaussian": ROOT / "results" / "exp3_rp_gaussian_sweep.json",
    "rp_sparse": ROOT / "results" / "exp3_rp_sparse_sweep.json",
}


def _best_dimensions(models: dict) -> dict[str, dict[str, int]]:
    """Read each model's best dimension from its completed one-seed sweep."""
    best = {}
    for setting, path in SWEEP_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required RP sweep is missing: {path}")
        rows = json.loads(path.read_text())
        best[setting] = {}
        for model_name in models:
            candidates = [
                row for row in rows
                if row.get("model") == model_name and row.get("setting") == setting
            ]
            if not candidates:
                raise ValueError(f"No sweep rows for {model_name}/{setting} in {path}")
            # Resolve ties toward fewer dimensions for a simpler representation.
            winner = max(candidates, key=lambda row: (row["accuracy"], -row["n_dims"]))
            best[setting][model_name] = int(winner["n_dims"])
    return best


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


def run_seed_experiment(repeats: int = 10, base_seed: int = 42) -> dict:
    if repeats < 2:
        raise ValueError("At least two repeats are required to estimate variability")
    X_train, y_train, _, X_test, y_test, _ = load_nsl_kdd(
        test_file="KDDTest+_no_train_feature_overlap.txt"
    )
    preprocessor = build_preprocessor()
    train = preprocessor.fit_transform(X_train)
    test = preprocessor.transform(X_test)
    models = get_paper_models()
    best_dims = _best_dimensions(models)
    runs = []

    for setting, projection_type in PROJECTIONS.items():
        for repeat in range(repeats):
            seed = base_seed + repeat
            for model_name, prototype in models.items():
                dims = best_dims[setting][model_name]
                projection = projection_type(n_components=dims, random_state=seed)
                started = time.perf_counter()
                train_reduced = projection.fit_transform(train)
                test_reduced = projection.transform(test)
                projection_seconds = time.perf_counter() - started
                classifier = clone(prototype)
                started = time.perf_counter()
                classifier.fit(train_reduced, y_train)
                fit_seconds = time.perf_counter() - started
                prediction = classifier.predict(test_reduced)
                runs.append({
                    "experiment": "rp_seed_stability",
                    "model": model_name,
                    "setting": setting,
                    "n_dims": dims,
                    "seed": seed,
                    "projection_fit_transform_seconds": projection_seconds,
                    "classifier_fit_seconds": fit_seconds,
                    **_metrics(y_test, prediction),
                })
                print(
                    f"{setting:12} seed={seed} {model_name:12} "
                    f"accuracy={runs[-1]['accuracy']:.4f}"
                )

    metrics = ("accuracy", "fpr", "precision", "recall", "f1", "macro_f1")
    summary = {}
    for setting in PROJECTIONS:
        for model_name in models:
            dims = best_dims[setting][model_name]
            selected = [r for r in runs if r["setting"] == setting and r["model"] == model_name]
            summary[f"{setting}/{model_name}"] = {
                "n_dims": dims,
                **{
                    metric: {
                        "mean": round(statistics.mean(r[metric] for r in selected), 6),
                        "std": round(statistics.stdev(r[metric] for r in selected), 6),
                    }
                    for metric in metrics
                },
            }

    result = {
        "experiment": "rp_seed_stability",
        "best_dimensions": best_dims,
        "base_seed": base_seed,
        "seeds": list(range(base_seed, base_seed + repeats)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_file": "KDDTest+_no_train_feature_overlap.txt",
        "test_selection_note": (
            "For each projection and model, dimensions were selected as the highest-accuracy "
            "choice in its single-seed sweep on official KDDTest+. Repeated results use the "
            "overlap-clean subset of that same test population, so selection can make scores "
            "optimistic; treat this as a seed-variability analysis, not an unbiased estimate. "
            "The overlap-clean file excludes exact feature rows also present in KDDTrain+."
        ),
        "summary": summary,
        "runs": runs,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    _plot(summary)
    print(f"Saved {RESULT_PATH} and {FIGURE_PATH}")
    return result


def _plot(summary):
    labels = []
    means = []
    stds = []
    colors = []
    palette = {"rp_gaussian": "#355C7D", "rp_sparse": "#F2A65A"}
    for setting in PROJECTIONS:
        for model in get_paper_models():
            values = summary[f"{setting}/{model}"]["accuracy"]
            dims = summary[f"{setting}/{model}"]["n_dims"]
            labels.append(f"{setting.removeprefix('rp_')} {dims}D\n{model}")
            means.append(values["mean"])
            stds.append(values["std"])
            colors.append(palette[setting])
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Accuracy on overlap-clean test split")
    ax.set_title("Random Projection seed variability at sweep-selected dimensions")
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP seed variability (Task 14)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_seed_experiment(args.repeats, args.seed)
