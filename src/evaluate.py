"""
evaluate.py — Evaluation utilities and results writer.

Computes: accuracy, FPR, per-class precision/recall/F1, macro F1,
           confusion matrix, and timing.
Writes standardised JSON to results/.
"""

import json
import time
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

def compute_fpr(y_true, y_pred, positive_label=1) -> float:
    """
    False Positive Rate = FP / (FP + TN).
    'positive_label' is the attack class (1 for binary).
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        # cm layout: [[TN, FP], [FN, TP]]  when labels=[0,1]
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        # Multiclass: macro-average FPR
        fprs = []
        for i in range(cm.shape[0]):
            fp_i = cm[:, i].sum() - cm[i, i]
            tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fprs.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0)
        return float(np.mean(fprs))


def compute_metrics(y_true, y_pred, average: str = "binary") -> dict:
    """
    Compute all standard metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
    average : str
        "binary" for binary task, "macro" for multiclass.

    Returns
    -------
    dict with keys: accuracy, fpr, precision, recall, f1, macro_f1,
                    confusion_matrix, classification_report
    """
    pos_label = 1 if average == "binary" else None
    avg = "binary" if average == "binary" else "macro"

    acc = accuracy_score(y_true, y_pred)
    fpr = compute_fpr(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 5),
        "fpr": round(fpr, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "macro_f1": round(macro, 5),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ------------------------------------------------------------------ #
#  Timing wrapper                                                     #
# ------------------------------------------------------------------ #

class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


# ------------------------------------------------------------------ #
#  Train + evaluate pipeline                                          #
# ------------------------------------------------------------------ #

def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str = "",
    setting: str = "full",
    n_dims: int | None = None,
    seed: int | None = None,
    average: str = "binary",
) -> dict:
    """
    Fit a model, predict on test, compute metrics and timing.

    Returns a result dict ready to be saved as JSON.
    """
    # --- Fit ---
    with Timer() as fit_timer:
        model.fit(X_train, y_train)

    # --- Predict ---
    with Timer() as pred_timer:
        y_pred = model.predict(X_test)

    # --- Metrics ---
    metrics = compute_metrics(y_test, y_pred, average=average)

    result = {
        "model": model_name or type(model).__name__,
        "setting": setting,
        "n_dims": n_dims if n_dims is not None else X_train.shape[1],
        "seed": seed,
        "fit_seconds": round(fit_timer.elapsed, 4),
        "predict_seconds": round(pred_timer.elapsed, 4),
        **metrics,
    }
    return result


# ------------------------------------------------------------------ #
#  Results I/O                                                        #
# ------------------------------------------------------------------ #

def save_result(result: dict, filename: str) -> Path:
    """Save a single result dict to results/<filename>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [OK] Saved: {filepath.name}")
    return filepath


def save_results(results: list[dict], filename: str) -> Path:
    """Save a list of result dicts to results/<filename>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Saved: {filepath.name}  ({len(results)} entries)")
    return filepath


def load_results(filename: str) -> list[dict] | dict:
    """Load results from a JSON file."""
    filepath = RESULTS_DIR / f"{filename}.json"
    with open(filepath) as f:
        return json.load(f)


def print_comparison_table(our_results: list[dict], paper_results: list[dict] | None = None):
    """Pretty-print a comparison table: our results vs paper."""
    header = f"{'Model':<15} {'Setting':<15} {'Dims':>5} {'Acc%':>7} {'FPR%':>7} {'F1':>7} {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in our_results:
        print(
            f"{r['model']:<15} {r['setting']:<15} {r['n_dims']:>5} "
            f"{r['accuracy']*100:>6.1f}% {r['fpr']*100:>6.1f}% "
            f"{r['f1']:>6.3f} {r['fit_seconds']:>7.2f}s"
        )
    print("=" * len(header))
