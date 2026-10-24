"""
run_experiments.py - Master experiment runner for Member A's tasks.

Experiments:
  1. Full features, all models (reproduce paper Table 3)
  2. Gaussian RP sweep (dims 1-5, 10, 15, 20, 30) - Task 12
  3. 5-class classification - Task 16

Usage:
    python -m src.run_experiments --exp 1
    python -m src.run_experiments --exp 2
    python -m src.run_experiments --exp 3
    python -m src.run_experiments --exp all
"""

import argparse
import sys
import numpy as np

from src.data import load_nsl_kdd
from src.preprocess import preprocess_data
from src.models import get_models, get_paper_models
from src.evaluate import (
    train_and_evaluate,
    save_results,
    load_results,
    print_comparison_table,
)


# ================================================================== #
#  EXPERIMENT 1 - Full features (Task 9)                              #
# ================================================================== #

def run_exp1_full_features():
    """
    Reproduce paper Table 3: all models on full 41 features (after encoding).
    Binary classification (normal vs attack).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Full Features - Binary Classification")
    print("=" * 60)

    X_train, y_train_bin, _, X_test, y_test_bin, _ = load_nsl_kdd()
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    models = get_models(task="binary")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        result = train_and_evaluate(
            model=model,
            X_train=X_train_proc,
            y_train=y_train_bin,
            X_test=X_test_proc,
            y_test=y_test_bin,
            model_name=name,
            setting="full",
        )
        results.append(result)

    save_results(results, "exp1_full_features")
    print_comparison_table(results)

    # --- Compare with paper ---
    try:
        paper = load_results("paper_reported")
        paper_exp1 = next(e for e in paper if e["experiment"] == "exp1_full")
        print("\n--- Paper Table 3 (for comparison) ---")
        for p in paper_exp1["results"]:
            print(f"  {p['model']:<15} Acc: {p['accuracy']*100:.1f}%  FPR: {p['fpr']*100:.1f}%")
    except Exception:
        print("  (paper_reported.json not found, skipping comparison)")

    return results


# ================================================================== #
#  EXPERIMENT 2 - Gaussian RP Sweep (Task 12)                        #
# ================================================================== #

def run_exp2_rp_gaussian_sweep():
    """
    Gaussian Random Projection: sweep dims 1-5, 10, 15, 20, 30.
    Binary classification.
    """
    from sklearn.random_projection import GaussianRandomProjection

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Gaussian Random Projection Sweep")
    print("=" * 60)

    X_train, y_train_bin, _, X_test, y_test_bin, _ = load_nsl_kdd()
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    dims = [1, 2, 3, 4, 5, 10, 15, 20, 30]
    models = get_paper_models()
    seed = 42
    all_results = []

    for n_dim in dims:
        print(f"\n{'='*40}")
        print(f"  Gaussian RP -> {n_dim} dimensions (seed={seed})")
        print(f"{'='*40}")

        rp = GaussianRandomProjection(n_components=n_dim, random_state=seed)
        X_train_rp = rp.fit_transform(X_train_proc)
        X_test_rp = rp.transform(X_test_proc)

        for name, model in models.items():
            print(f"  {name}...", end=" ")
            result = train_and_evaluate(
                model=model,
                X_train=X_train_rp,
                y_train=y_train_bin,
                X_test=X_test_rp,
                y_test=y_test_bin,
                model_name=name,
                setting="rp_gaussian",
                n_dims=n_dim,
                seed=seed,
            )
            all_results.append(result)
            print(f"Acc: {result['accuracy']*100:.1f}%  FPR: {result['fpr']*100:.1f}%")

    save_results(all_results, "exp3_rp_gaussian_sweep")
    print_comparison_table(all_results)
    return all_results


# ================================================================== #
#  EXPERIMENT 3 - 5-Class Classification (Task 16)                   #
# ================================================================== #

def run_exp3_multiclass():
    """
    5-class classification: normal, dos, probe, r2l, u2r.
    Full features + best RP setting.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.random_projection import GaussianRandomProjection

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: 5-Class Classification")
    print("=" * 60)

    X_train, _, y_train_cat, X_test, _, y_test_cat = load_nsl_kdd()
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_cat)
    y_test_enc = le.transform(y_test_cat)
    print(f"  Classes: {list(le.classes_)}")

    models = get_models(task="multiclass")
    all_results = []

    # --- Full features ---
    print("\n--- Full Features (5-class) ---")
    for name, model in models.items():
        print(f"  {name}...", end=" ")
        result = train_and_evaluate(
            model=model,
            X_train=X_train_proc,
            y_train=y_train_enc,
            X_test=X_test_proc,
            y_test=y_test_enc,
            model_name=name,
            setting="full",
            average="macro",
        )
        # Store class names for readability
        result["classes"] = list(le.classes_)
        result["task"] = "multiclass"
        all_results.append(result)
        print(f"Acc: {result['accuracy']*100:.1f}%  Macro-F1: {result['macro_f1']:.3f}")

    # --- Best RP setting (5 dims, from paper) ---
    print("\n--- Gaussian RP 5 dims (5-class) ---")
    rp = GaussianRandomProjection(n_components=5, random_state=42)
    X_train_rp = rp.fit_transform(X_train_proc)
    X_test_rp = rp.transform(X_test_proc)

    models_rp = get_models(task="multiclass")
    for name, model in models_rp.items():
        print(f"  {name}...", end=" ")
        result = train_and_evaluate(
            model=model,
            X_train=X_train_rp,
            y_train=y_train_enc,
            X_test=X_test_rp,
            y_test=y_test_enc,
            model_name=name,
            setting="rp_gaussian",
            n_dims=5,
            seed=42,
            average="macro",
        )
        result["classes"] = list(le.classes_)
        result["task"] = "multiclass"
        all_results.append(result)
        print(f"Acc: {result['accuracy']*100:.1f}%  Macro-F1: {result['macro_f1']:.3f}")

    save_results(all_results, "exp5_multiclass")
    print_comparison_table(all_results)
    return all_results


# ================================================================== #
#  CLI                                                                #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(description="NIDS Experiment Runner")
    parser.add_argument(
        "--exp",
        type=str,
        default="1",
        choices=["1", "2", "3", "all"],
        help="Which experiment to run: 1=full features, 2=RP sweep, 3=multiclass, all=everything",
    )
    args = parser.parse_args()

    if args.exp in ("1", "all"):
        run_exp1_full_features()
    if args.exp in ("2", "all"):
        run_exp2_rp_gaussian_sweep()
    if args.exp in ("3", "all"):
        run_exp3_multiclass()

    print("\n[DONE] All experiments completed.")


if __name__ == "__main__":
    main()
