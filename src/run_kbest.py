"""Task 18: compare mutual-information feature selection with PCA and RP.

All preprocessing and dimensionality selection are fitted on KDDTrain+ only.
The same three paper classifiers and dimensions are used for each setting so
the comparison is paired by model and dimensionality.

Run with::

    python -m src.run_kbest
"""

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.random_projection import GaussianRandomProjection

from src.data import load_nsl_kdd
from src.evaluate import save_results, train_and_evaluate
from src.models import get_paper_models
from src.preprocess import preprocess_data


SEED = 42
DIMENSIONS = (5, 10, 20)


def run_kbest_comparison() -> list[dict]:
    """Evaluate K-best, PCA, and Gaussian RP for each model and dimension."""
    X_train, y_train_bin, _, X_test, y_test_bin, _ = load_nsl_kdd()
    X_train_proc, X_test_proc, preprocessor = preprocess_data(X_train, X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Score once, then take nested top-k subsets; avoids recomputing the same
    # expensive mutual-information estimates for k=5, 10, and 20.
    selection_started = time.perf_counter()
    scores = mutual_info_classif(X_train_proc, y_train_bin, random_state=SEED)
    selection_fit_seconds = time.perf_counter() - selection_started
    ranking = np.argsort(scores)[::-1]

    models = get_paper_models()
    results: list[dict] = []
    for n_dims in DIMENSIONS:
        selected = ranking[:n_dims]
        X_train_kbest = X_train_proc[:, selected]
        X_test_kbest = X_test_proc[:, selected]

        pca = PCA(n_components=n_dims, random_state=SEED)
        pca_started = time.perf_counter()
        X_train_pca = pca.fit_transform(X_train_proc)
        X_test_pca = pca.transform(X_test_proc)
        pca_fit_seconds = time.perf_counter() - pca_started

        rp = GaussianRandomProjection(n_components=n_dims, random_state=SEED)
        rp_started = time.perf_counter()
        X_train_rp = rp.fit_transform(X_train_proc)
        X_test_rp = rp.transform(X_test_proc)
        rp_fit_seconds = time.perf_counter() - rp_started

        settings = (
            ("kbest", X_train_kbest, X_test_kbest, selection_fit_seconds),
            ("pca", X_train_pca, X_test_pca, pca_fit_seconds),
            ("rp_gaussian", X_train_rp, X_test_rp, rp_fit_seconds),
        )
        for setting, X_reduced_train, X_reduced_test, reduction_seconds in settings:
            for model_name, model in models.items():
                result = train_and_evaluate(
                    model=model,
                    X_train=X_reduced_train,
                    y_train=y_train_bin,
                    X_test=X_reduced_test,
                    y_test=y_test_bin,
                    model_name=model_name,
                    setting=setting,
                    n_dims=n_dims,
                    seed=SEED,
                )
                result["reduction_fit_seconds"] = round(reduction_seconds, 4)
                if setting == "kbest":
                    result["selected_features"] = [str(feature_names[i]) for i in selected]
                results.append(result)
                print(
                    f"{setting:<12} k={n_dims:<2} {model_name:<13} "
                    f"accuracy={result['accuracy']:.4f} f1={result['f1']:.4f}"
                )

    save_results(results, "exp7_kbest")
    return results


if __name__ == "__main__":
    run_kbest_comparison()
