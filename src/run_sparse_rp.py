"""Task 13: sweep Sparse Random Projection dimensions on KDDTest+.

Run with ``python -m src.run_sparse_rp``. Preprocessing and projection fit
only on KDDTrain+; the official test split is transformed with those objects.
"""
from sklearn.random_projection import SparseRandomProjection

from src.data import load_nsl_kdd
from src.evaluate import print_comparison_table, save_results, train_and_evaluate
from src.models import get_paper_models
from src.preprocess import preprocess_data

SEED = 42
DIMENSIONS = (1, 2, 3, 4, 5, 10, 15, 20, 30)


def run_sparse_rp_sweep():
    X_train, y_train, _, X_test, y_test, _ = load_nsl_kdd()
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)
    results = []

    for dimensions in DIMENSIONS:
        print(f"\nSparse RP: {dimensions} dimensions (seed={SEED})")
        projection = SparseRandomProjection(
            n_components=dimensions,
            density="auto",
            dense_output=True,
            random_state=SEED,
        )
        train_reduced = projection.fit_transform(X_train_proc)
        test_reduced = projection.transform(X_test_proc)

        for name, estimator in get_paper_models().items():
            result = train_and_evaluate(
                model=estimator,
                X_train=train_reduced,
                y_train=y_train,
                X_test=test_reduced,
                y_test=y_test,
                model_name=name,
                setting="rp_sparse",
                n_dims=dimensions,
                seed=SEED,
            )
            results.append(result)
            print(f"  {name}: acc={result['accuracy']:.4f}, fpr={result['fpr']:.4f}")

    save_results(results, "exp3_rp_sparse_sweep")
    print_comparison_table(results)
    return results


if __name__ == "__main__":
    run_sparse_rp_sweep()
