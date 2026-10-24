"""
data.py — NSL-KDD data loading contract.

load_nsl_kdd() -> X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat

X_*        : pandas DataFrame, 41 raw columns, no label, no difficulty column
y_*_bin    : Series, 0 = normal, 1 = attack
y_*_cat    : Series, one of "normal", "dos", "probe", "r2l", "u2r"
"""

import pandas as pd
from pathlib import Path

from src.attack_map import get_binary_label, get_category

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 41 NSL-KDD feature names (standard ordering)
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Categorical columns (will be one-hot encoded)
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# All 43 columns in the raw file (41 features + label + difficulty)
ALL_COLUMNS = FEATURE_COLUMNS + ["attack_type", "difficulty"]


def _load_raw(filepath: Path) -> pd.DataFrame:
    """Load a raw NSL-KDD text file (no header, comma-separated, 43 cols)."""
    df = pd.read_csv(filepath, header=None, names=ALL_COLUMNS)
    # Strip whitespace from string columns
    for col in CATEGORICAL_COLUMNS + ["attack_type"]:
        df[col] = df[col].str.strip()
    return df


def load_nsl_kdd(
    train_file: str = "KDDTrain+.txt",
    test_file: str = "KDDTest+.txt",
) -> tuple:
    """
    Load NSL-KDD and return (X_train, y_train_bin, y_train_cat,
                              X_test,  y_test_bin,  y_test_cat).

    Raises FileNotFoundError if data files are missing.
    """
    train_path = DATA_DIR / train_file
    test_path = DATA_DIR / test_file

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            f"Download from https://www.unb.ca/cic/datasets/nsl.html"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            f"Download from https://www.unb.ca/cic/datasets/nsl.html"
        )

    # --- Load ---
    train_df = _load_raw(train_path)
    test_df = _load_raw(test_path)

    # --- Features (drop label + difficulty) ---
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

    # --- Labels ---
    y_train_bin = train_df["attack_type"].apply(get_binary_label)
    y_test_bin = test_df["attack_type"].apply(get_binary_label)

    y_train_cat = train_df["attack_type"].apply(get_category)
    y_test_cat = test_df["attack_type"].apply(get_category)

    # --- Sanity & Deduplication checks (Task 3 requirement) ---
    assert X_train.shape[1] == 41, f"Expected 41 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 41, f"Expected 41 features, got {X_test.shape[1]}"
    assert len(X_train) == len(y_train_bin)
    assert len(X_test) == len(y_test_bin)

    train_dups = int(X_train.duplicated().sum())
    test_dups = int(X_test.duplicated().sum())
    overlap_dups = int(pd.merge(X_train, X_test, how="inner").shape[0])

    # --- Summary ---
    print(f"Loaded NSL-KDD:")
    print(f"  Train: {len(X_train):,} rows  |  Normal: {(y_train_bin==0).sum():,}  Attack: {(y_train_bin==1).sum():,}")
    print(f"  Test:  {len(X_test):,} rows  |  Normal: {(y_test_bin==0).sum():,}  Attack: {(y_test_bin==1).sum():,}")
    print(f"  Dedupe Check: Train duplicate feature rows: {train_dups}, Test duplicates: {test_dups}, Exact overlap: {overlap_dups}")
    print(f"  Categories (train): {y_train_cat.value_counts().to_dict()}")
    print(f"  Categories (test):  {y_test_cat.value_counts().to_dict()}")

    return X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat


def get_feature_columns() -> list[str]:
    """Return the 41 feature column names."""
    return FEATURE_COLUMNS.copy()


def get_categorical_columns() -> list[str]:
    """Return the 3 categorical column names."""
    return CATEGORICAL_COLUMNS.copy()


def get_numeric_columns() -> list[str]:
    """Return the 38 numeric column names."""
    return [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
