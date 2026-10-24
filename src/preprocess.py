"""
preprocess.py — Preprocessing pipeline for NIDS.

Provides a ColumnTransformer that:
  - OneHotEncodes the 3 categorical columns (protocol_type, service, flag)
  - StandardScales all numeric columns

IMPORTANT: Fit on train only. Transform test with the fitted object.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data import CATEGORICAL_COLUMNS, get_numeric_columns


def build_preprocessor(X_train=None) -> ColumnTransformer:
    """
    Build the ColumnTransformer.

    If X_train is provided, the preprocessor is fit on it before returning.

    Returns
    -------
    ColumnTransformer (fitted if X_train was given, unfitted otherwise)
    """
    numeric_cols = get_numeric_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    if X_train is not None:
        preprocessor.fit(X_train)

    return preprocessor


def preprocess_data(X_train, X_test) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Fit preprocessor on X_train, transform both train and test.

    Parameters
    ----------
    X_train : DataFrame with 41 raw features
    X_test  : DataFrame with 41 raw features

    Returns
    -------
    X_train_proc : np.ndarray (scaled + one-hot encoded)
    X_test_proc  : np.ndarray (scaled + one-hot encoded)
    preprocessor : fitted ColumnTransformer
    """
    preprocessor = build_preprocessor()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Preprocessing:")
    print(f"  Raw features:         {X_train.shape[1]}")
    print(f"  After encoding:       {X_train_proc.shape[1]}")
    print(f"  Numeric columns:      {len(get_numeric_columns())}")
    print(f"  Categorical columns:  {len(CATEGORICAL_COLUMNS)} -> {X_train_proc.shape[1] - len(get_numeric_columns())} (one-hot)")

    return X_train_proc, X_test_proc, preprocessor


def build_full_pipeline(model, X_train=None) -> Pipeline:
    """
    Build a full Pipeline: preprocessor → model.

    Accepts raw 41-column DataFrames directly.
    If X_train is given, nothing is fit yet — caller must call pipeline.fit().

    Returns
    -------
    sklearn Pipeline
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline
