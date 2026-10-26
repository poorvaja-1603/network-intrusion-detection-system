"""Task 17: benchmark modern classifiers on full and Gaussian RP features."""
from sklearn.random_projection import GaussianRandomProjection

from src.data import load_nsl_kdd
from src.evaluate import print_comparison_table, save_results, train_and_evaluate
from src.models import get_models
from src.preprocess import preprocess_data

SEED = 42
RP_DIMENSIONS = 5


def run_modern_models():
    X_train, y_train, _, X_test, y_test, _ = load_nsl_kdd()
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    model_zoo = get_models(task="binary")
    selected = [name for name in ("XGBoost", "MLP") if name in model_zoo]
    if not selected:
        raise RuntimeError("Neither XGBoost nor MLP is available in this environment")
    unavailable = [name for name in ("XGBoost",) if name not in model_zoo]
    if unavailable:
        print("Skipped unavailable modern models: " + ", ".join(unavailable))

    projection = GaussianRandomProjection(n_components=RP_DIMENSIONS, random_state=SEED)
    train_rp = projection.fit_transform(X_train_proc)
    test_rp = projection.transform(X_test_proc)
    settings = (
        ("full", X_train_proc, X_test_proc, X_train_proc.shape[1]),
        ("rp_gaussian", train_rp, test_rp, RP_DIMENSIONS),
    )
    results = []
    for setting, train_features, test_features, dimensions in settings:
        for name in selected:
            # Fresh deterministic estimator for each representation.
            estimator = get_models(task="binary")[name]
            result = train_and_evaluate(
                model=estimator,
                X_train=train_features,
                y_train=y_train,
                X_test=test_features,
                y_test=y_test,
                model_name=name,
                setting=setting,
                n_dims=dimensions,
                seed=SEED,
            )
            result["task"] = "modern_binary"
            results.append(result)
            print(
                f"{name} / {setting}: accuracy={result['accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}, fpr={result['fpr']:.4f}"
            )

    save_results(results, "exp6_modern")
    print_comparison_table(results)
    return results


if __name__ == "__main__":
    run_modern_models()
