"""Task 11: run Weka PART and BayesNet on full and Gaussian-RP features.

This reproducible runner exports train/test CSVs to a temporary directory and
invokes Weka's command-line classifier interface. The class remains binary
(normal/attack), and preprocessing and RP are fitted on KDDTrain+ only.

Run with ``WEKA_JAR=/path/to/weka.jar python -m src.run_weka``.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

from src.data import load_nsl_kdd
from src.preprocess import preprocess_data


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEKA_JAR = Path("/Applications/weka-3-8-7/weka.jar")
RESULTS_PATH = ROOT / "results" / "weka_results.json"
EXPORT_DIR = ROOT / "data" / "weka_exports"
SEED = 42


def _confusion_matrix(output: str) -> tuple[list[list[int]], list[str]]:
    """Extract Weka's confusion matrix and class labels from classifier output."""
    lines = output.splitlines()
    starts = [i for i, line in enumerate(lines) if "<-- classified as" in line]
    if not starts:
        raise ValueError("Weka output did not contain a confusion matrix:\n" + output[:3000])
    # Weka prints one matrix for the training data and another for the supplied
    # test set; the final one is the KDDTest+ result we report.
    start = starts[-1]

    matrix: list[list[int]] = []
    labels: list[str] = []
    row_pattern = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*\|\s*[a-z]\s*=\s*(\S+)")
    for line in lines[start + 1 :]:
        match = row_pattern.match(line)
        if not match:
            if matrix:
                break
            continue
        matrix.append([int(value) for value in match.group(1).split()])
        labels.append(match.group(2))

    if len(matrix) != 2 or any(len(row) != 2 for row in matrix) or len(labels) != 2:
        raise ValueError(
            "Expected a binary Weka confusion matrix; got: " + repr(matrix)
            + "\n" + "\n".join(lines[start : start + 10])
        )

    # Normalize to [[TN, FP], [FN, TP]] with the positive class = attack.
    order = [labels.index("normal"), labels.index("attack")]
    normalized = [[matrix[i][j] for j in order] for i in order]
    return normalized, labels


def _metrics(cm: list[list[int]]) -> dict:
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = (
        2 * normal_precision * normal_recall / (normal_precision + normal_recall)
        if normal_precision + normal_recall
        else 0.0
    )
    return {
        "accuracy": (tn + tp) / total if total else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + normal_f1) / 2,
    }


def _run_classifier(weka_jar: Path, classifier: str, train_data: Path, test_data: Path) -> tuple[str, float]:
    class_name = {
        "PART": "weka.classifiers.rules.PART",
        "BayesNet": "weka.classifiers.bayes.BayesNet",
    }[classifier]
    classpath = str(weka_jar)
    libs = weka_jar.parent / "lib" / "*"
    if libs.parent.exists():
        classpath += os.pathsep + str(libs)
    weka_home = Path(tempfile.gettempdir()) / "nids-weka-home"
    weka_home.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    classifier_options = ["-D"] if classifier == "BayesNet" else []
    try:
        completed = subprocess.run(
            [
                "java", "--add-opens=java.base/java.lang=ALL-UNNAMED",
                f"-DWEKA_HOME={weka_home}", "-cp", classpath, class_name,
                "-t", str(train_data), "-T", str(test_data), "-c", "last",
                *classifier_options,
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "WEKA_HOME": str(weka_home)},
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stdout + "\n" + error.stderr) from error
    return completed.stdout + "\n" + completed.stderr, time.perf_counter() - started


def _write_arff(
    frame: pd.DataFrame,
    path: Path,
    relation: str,
    feature_scale: float = 1.0,
) -> None:
    """Write CSV data as ARFF with a shared binary class domain."""
    with path.open("w") as stream:
        stream.write(f"@relation {relation}\n\n")
        for column in frame.columns[:-1]:
            stream.write(f"@attribute {column} numeric\n")
        stream.write("@attribute label {attack,normal}\n\n@data\n")
        scaled = frame.copy()
        scaled.iloc[:, :-1] = scaled.iloc[:, :-1] * feature_scale
        scaled.to_csv(stream, index=False, header=False)


def run_weka_benchmarks(weka_jar: Path = DEFAULT_WEKA_JAR) -> list[dict]:
    if not weka_jar.exists():
        raise FileNotFoundError(
            f"Weka JAR not found at {weka_jar}; set WEKA_JAR to the installed weka.jar"
        )

    X_train, y_train_bin, _, X_test, y_test_bin, _ = load_nsl_kdd()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    X_train_full, X_test_full, _ = preprocess_data(X_train, X_test)
    all_feature_names = [f"feature_{i + 1}" for i in range(X_train_full.shape[1])]
    variable_mask = np.var(X_train_full, axis=0) > 0
    dropped_constant_features = [
        name for name, keep in zip(all_feature_names, variable_mask) if not keep
    ]
    X_train_full = X_train_full[:, variable_mask]
    X_test_full = X_test_full[:, variable_mask]
    feature_names = [name for name, keep in zip(all_feature_names, variable_mask) if keep]

    rp = GaussianRandomProjection(n_components=5, random_state=SEED)
    X_train_rp = rp.fit_transform(X_train_full)
    X_test_rp = rp.transform(X_test_full)

    results = []
    with tempfile.TemporaryDirectory(prefix="nids-weka-") as tmpdir:
        temp_dir = Path(tmpdir)
        for setting, train_values, test_values, names in (
            ("full", X_train_full, X_test_full, feature_names),
            (
                "rp_gaussian",
                X_train_rp,
                X_test_rp,
                [f"rp_{i + 1}" for i in range(X_train_rp.shape[1])],
            ),
        ):
            train_frame = pd.DataFrame(train_values, columns=names)
            test_frame = pd.DataFrame(test_values, columns=names)
            train_frame["label"] = np.where(np.asarray(y_train_bin) == 1, "attack", "normal")
            test_frame["label"] = np.where(np.asarray(y_test_bin) == 1, "attack", "normal")
            # Weka's CSV loader infers nominal value order from first occurrence.
            # Use a stable shared ordering so train/test class declarations match.
            train_frame = train_frame.sort_values("label", kind="stable")
            test_frame = test_frame.sort_values("label", kind="stable")
            train_csv = temp_dir / f"{setting}_train.csv"
            test_csv = temp_dir / f"{setting}_test.csv"
            train_arff = temp_dir / f"{setting}_train.arff"
            test_arff = temp_dir / f"{setting}_test.arff"
            train_bayesnet_arff = temp_dir / f"{setting}_train_bayesnet.arff"
            test_bayesnet_arff = temp_dir / f"{setting}_test_bayesnet.arff"
            train_frame.to_csv(train_csv, index=False)
            test_frame.to_csv(test_csv, index=False)
            train_frame.to_csv(
                EXPORT_DIR / f"{train_csv.name}.gz", index=False, compression="gzip"
            )
            test_frame.to_csv(
                EXPORT_DIR / f"{test_csv.name}.gz", index=False, compression="gzip"
            )
            _write_arff(train_frame, train_arff, f"nids_{setting}_train")
            _write_arff(test_frame, test_arff, f"nids_{setting}_test")
            # Weka BayesNet's automatic discretizer needs extra numeric range
            # precision for the scaled one-hot data. Multiplying all features
            # by a constant preserves their ordering and resolves duplicate
            # printed bin boundaries; ADTree is disabled to keep memory bounded.
            _write_arff(
                train_frame,
                train_bayesnet_arff,
                f"nids_{setting}_bayesnet_train",
                feature_scale=1_000_000,
            )
            _write_arff(
                test_frame,
                test_bayesnet_arff,
                f"nids_{setting}_bayesnet_test",
                feature_scale=1_000_000,
            )

            for classifier in ("PART", "BayesNet"):
                print(f"Running Weka {classifier} on {setting}...")
                train_input, test_input = (
                    (train_bayesnet_arff, test_bayesnet_arff)
                    if classifier == "BayesNet"
                    else (train_arff, test_arff)
                )
                output, elapsed = _run_classifier(weka_jar, classifier, train_input, test_input)
                cm, labels = _confusion_matrix(output)
                metrics = _metrics(cm)
                result = {
                    "model": classifier,
                    "setting": setting,
                    "n_dims": int(train_values.shape[1]),
                    "seed": SEED if setting == "rp_gaussian" else None,
                    **{key: round(value, 5) for key, value in metrics.items()},
                    "confusion_matrix": cm,
                    "class_order_from_weka": labels,
                    "weka_elapsed_seconds": round(elapsed, 4),
                    "dropped_zero_variance_features": dropped_constant_features,
                    "bayesnet_feature_scale": 1_000_000 if classifier == "BayesNet" else None,
                }
                results.append(result)
                print(
                    f"  accuracy={result['accuracy']:.4f} "
                    f"FPR={result['fpr']:.4f} time={elapsed:.1f}s"
                )

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved Weka results: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    configured_jar = Path(os.environ.get("WEKA_JAR", str(DEFAULT_WEKA_JAR)))
    run_weka_benchmarks(configured_jar)
