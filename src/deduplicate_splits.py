"""Create a test split with no feature-vector overlap with NSL-KDD training.

The official KDDTest+ file is preserved unchanged. This script writes a
separate, leakage-reduced test file and an audit manifest so results on the
cleaned split are not confused with results on the published benchmark.
"""

import json
from pathlib import Path

import pandas as pd

from src.data import DATA_DIR, FEATURE_COLUMNS, _load_raw


OUTPUT_PATH = DATA_DIR / "KDDTest+_no_train_feature_overlap.txt"
AUDIT_PATH = DATA_DIR / "split_overlap_audit.json"


def create_clean_test_split() -> dict:
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"
    train = _load_raw(train_path)
    test = _load_raw(test_path)

    train_keys = pd.MultiIndex.from_frame(train[FEATURE_COLUMNS].drop_duplicates())
    test_keys = pd.MultiIndex.from_frame(test[FEATURE_COLUMNS])
    overlap_mask = test_keys.isin(train_keys)
    clean_test = test.loc[~overlap_mask].copy()

    # Keep the standard 43-column, headerless NSL-KDD format.
    clean_test.to_csv(OUTPUT_PATH, index=False, header=False)

    overlap = test.loc[overlap_mask]
    shared_vectors = len(
        pd.MultiIndex.from_frame(test.loc[overlap_mask, FEATURE_COLUMNS].drop_duplicates())
    )
    audit = {
        "source_train": train_path.name,
        "source_test": test_path.name,
        "clean_test": OUTPUT_PATH.name,
        "overlap_definition": "Exact match on all 41 input feature columns; labels and difficulty are excluded.",
        "train_rows": int(len(train)),
        "official_test_rows": int(len(test)),
        "removed_test_rows": int(overlap_mask.sum()),
        "shared_distinct_feature_vectors": int(shared_vectors),
        "clean_test_rows": int(len(clean_test)),
        "removed_rows_by_test_label": {
            str(label): int(count)
            for label, count in overlap["attack_type"].value_counts().sort_index().items()
        },
        "clean_test_rows_by_label": {
            str(label): int(count)
            for label, count in clean_test["attack_type"].value_counts().sort_index().items()
        },
        "official_files_modified": False,
    }

    clean_keys = pd.MultiIndex.from_frame(clean_test[FEATURE_COLUMNS])
    remaining_overlap = int(clean_keys.isin(train_keys).sum())
    audit["remaining_feature_overlap_rows"] = remaining_overlap
    if remaining_overlap:
        raise AssertionError(f"Clean test still has {remaining_overlap} overlapping rows")
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    print(json.dumps(create_clean_test_split(), indent=2))
