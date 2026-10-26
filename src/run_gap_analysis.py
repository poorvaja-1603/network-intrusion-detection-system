"""Task 19: compare stratified train-only cross-validation with KDDTest+.

The preprocessing pipeline is cloned and fitted inside each fold to prevent
information from a validation fold leaking into its own transforms.
Run with ``python -m src.run_gap_analysis``.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data import load_nsl_kdd
from src.evaluate import compute_metrics
from src.preprocess import build_preprocessor


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "results" / "exp8_gap.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "generalisation_gap.png"
SEED = 42
FOLDS = 5


def _models():
    return {
        "NaiveBayes": GaussianNB(),
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, criterion="entropy", random_state=SEED, n_jobs=-1
        ),
    }


def run_gap_analysis() -> dict:
    X_train, y_train, _, X_test, y_test, _ = load_nsl_kdd()
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    output = {
        "experiment": "train_cv_vs_kddtest",
        "seed": SEED,
        "cv_folds": FOLDS,
        "training_rows": len(X_train),
        "official_test_rows": len(X_test),
        "official_test_note": (
            "The official KDDTest+ split is unchanged. It includes repeated feature "
            "rows found in training; see data/split_overlap_audit.json."
        ),
        "models": [],
    }

    for name, estimator in _models().items():
        oof = np.empty(len(y_train), dtype=np.int8)
        fold_fit_seconds = []
        for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X_train, y_train), 1):
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", clone(estimator)),
            ])
            started = time.perf_counter()
            pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            fold_fit_seconds.append(time.perf_counter() - started)
            oof[valid_idx] = pipeline.predict(X_train.iloc[valid_idx])
            print(f"{name}: fold {fold}/{FOLDS} complete")

        test_pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clone(estimator)),
        ])
        started = time.perf_counter()
        test_pipeline.fit(X_train, y_train)
        test_fit_seconds = time.perf_counter() - started
        test_prediction = test_pipeline.predict(X_test)
        cv_metrics = compute_metrics(y_train, oof, average="binary")
        test_metrics = compute_metrics(y_test, test_prediction, average="binary")
        output["models"].append({
            "model": name,
            "cv_metrics": cv_metrics,
            "cv_fold_fit_seconds": [round(x, 4) for x in fold_fit_seconds],
            "cv_fit_seconds_mean": round(float(np.mean(fold_fit_seconds)), 4),
            "test_metrics": test_metrics,
            "test_fit_seconds": round(test_fit_seconds, 4),
            "accuracy_gap_cv_minus_test": round(
                cv_metrics["accuracy"] - test_metrics["accuracy"], 5
            ),
            "attack_recall_gap_cv_minus_test": round(
                cv_metrics["recall"] - test_metrics["recall"], 5
            ),
        })
        print(
            f"{name}: CV accuracy={cv_metrics['accuracy']:.4f}, "
            f"test accuracy={test_metrics['accuracy']:.4f}, "
            f"CV recall={cv_metrics['recall']:.4f}, "
            f"test recall={test_metrics['recall']:.4f}"
        )

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    _plot_gap(output)
    print(f"Saved {RESULT_PATH}")
    return output


def _plot_gap(output: dict) -> None:
    names = [row["model"] for row in output["models"]]
    cv_accuracy = [row["cv_metrics"]["accuracy"] for row in output["models"]]
    test_accuracy = [row["test_metrics"]["accuracy"] for row in output["models"]]
    cv_recall = [row["cv_metrics"]["recall"] for row in output["models"]]
    test_recall = [row["test_metrics"]["recall"] for row in output["models"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    positions = np.arange(len(names))
    width = 0.36
    for ax, first, second, title in (
        (axes[0], cv_accuracy, test_accuracy, "Accuracy"),
        (axes[1], cv_recall, test_recall, "Attack recall"),
    ):
        ax.bar(positions - width / 2, first, width, label="5-fold train CV", color="#31688e")
        ax.bar(positions + width / 2, second, width, label="Official KDDTest+", color="#f28e2b")
        ax.set_xticks(positions, names, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(axis="y", color="#d9e1e8", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Train cross-validation vs fixed test split", fontsize=15)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_PATH}")


if __name__ == "__main__":
    run_gap_analysis()
