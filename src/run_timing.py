"""Timing benchmark for the preprocessing and dimensionality settings.

Task 15: measure preprocessing/projection fit time separately from classifier
fit time for full features, PCA, Gaussian RP, and mutual-information K-best.
All transforms are fitted on the training split only. The KDDTest+ split is
not used because this benchmark measures training cost.

Run with::

    python -m src.run_timing
    python -m src.run_timing --repeats 5 --dims 20 --model J48
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.random_projection import GaussianRandomProjection

from src.data import load_nsl_kdd
from src.models import get_paper_models
from src.preprocess import build_preprocessor


RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "timing.json"
SETTINGS = ("full", "pca", "rp_gaussian", "kbest")


def _new_transformer(setting: str, dims: int, seed: int):
    """Build a fresh dimensionality reducer for one timing repetition."""
    if setting == "full":
        return None
    if setting == "pca":
        return PCA(n_components=dims, random_state=seed)
    if setting == "rp_gaussian":
        return GaussianRandomProjection(n_components=dims, random_state=seed)
    if setting == "kbest":
        # mutual_info_classif uses randomness to break ties on continuous data.
        score_func: Callable = lambda x, y: mutual_info_classif(
            x, y, random_state=seed
        )
        return SelectKBest(score_func=score_func, k=dims)
    raise ValueError(f"Unknown setting: {setting}")


def _mean_std(values: list[float]) -> dict:
    return {
        "mean_seconds": round(statistics.mean(values), 6),
        "std_seconds": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def run_timing_benchmark(
    repeats: int = 5,
    dims: int = 20,
    model_name: str = "J48",
    seed: int = 42,
    output_path: Path = RESULTS_PATH,
) -> dict:
    """Run repeated fit-time measurements and save the aggregate JSON."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if dims < 1:
        raise ValueError("dims must be at least 1")

    models = get_paper_models()
    if model_name not in models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {', '.join(models)}"
        )

    X_train, y_train_bin, _, _, _, _ = load_nsl_kdd()
    # This is the encoded/scaled dimension count after fitting on train only.
    probe_preprocessor = build_preprocessor()
    probe_preprocessor.fit(X_train)
    encoded_dims = len(probe_preprocessor.get_feature_names_out())
    if dims > encoded_dims:
        raise ValueError(f"dims={dims} exceeds {encoded_dims} preprocessed features")

    samples: dict[str, list[dict]] = {setting: [] for setting in SETTINGS}
    for repetition in range(repeats):
        repetition_seed = seed + repetition
        for setting in SETTINGS:
            # Include the common preprocessing cost but report it separately.
            preprocessor = build_preprocessor()
            start = time.perf_counter()
            X_encoded = preprocessor.fit_transform(X_train)
            preprocessing_seconds = time.perf_counter() - start

            reducer = _new_transformer(setting, dims, repetition_seed)
            if reducer is None:
                X_model = X_encoded
                projection_seconds = 0.0
                effective_dims = X_model.shape[1]
            else:
                start = time.perf_counter()
                if setting == "kbest":
                    X_model = reducer.fit_transform(X_encoded, y_train_bin)
                else:
                    X_model = reducer.fit_transform(X_encoded)
                projection_seconds = time.perf_counter() - start
                effective_dims = X_model.shape[1]

            classifier = clone(models[model_name])
            start = time.perf_counter()
            classifier.fit(X_model, y_train_bin)
            classifier_seconds = time.perf_counter() - start

            samples[setting].append({
                "repetition": repetition + 1,
                "seed": repetition_seed,
                "preprocessing_fit_seconds": preprocessing_seconds,
                "projection_fit_seconds": projection_seconds,
                "classifier_fit_seconds": classifier_seconds,
                "total_fit_seconds": preprocessing_seconds + projection_seconds + classifier_seconds,
                "n_dims": int(effective_dims),
            })

    summaries = {}
    for setting, rows in samples.items():
        summaries[setting] = {
            "n_dims": rows[0]["n_dims"],
            "preprocessing_fit": _mean_std([r["preprocessing_fit_seconds"] for r in rows]),
            "projection_fit": _mean_std([r["projection_fit_seconds"] for r in rows]),
            "classifier_fit": _mean_std([r["classifier_fit_seconds"] for r in rows]),
            "total_fit": _mean_std([r["total_fit_seconds"] for r in rows]),
        }

    result = {
        "experiment": "timing_benchmark",
        "model": model_name,
        "repeats": repeats,
        "requested_reduced_dims": dims,
        "base_seed": seed,
        "train_rows": len(X_train),
        "preprocessed_features": encoded_dims,
        "timing_scope": "fit only; transforms are fit on KDDTrain+ only",
        "summary": summaries,
        "runs": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved timing benchmark to {output_path}")
    for setting, values in summaries.items():
        print(
            f"{setting:<12} projection={values['projection_fit']['mean_seconds']:.4f}s  "
            f"classifier={values['classifier_fit']['mean_seconds']:.4f}s  "
            f"total={values['total_fit']['mean_seconds']:.4f}s"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="NIDS fit-time benchmark (Task 15)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dims", type=int, default=20)
    parser.add_argument("--model", default="J48", choices=sorted(get_paper_models()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    run_timing_benchmark(args.repeats, args.dims, args.model, args.seed, args.output)


if __name__ == "__main__":
    main()
