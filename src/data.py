"""
data.py — NSL-KDD data loading contract

Returns:
    load_nsl_kdd() -> X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat
    
    X_*        : pandas DataFrame, 41 raw columns, no label, no difficulty column
    y_*_bin    : 0 = normal, 1 = attack
    y_*_cat    : one of "normal", "dos", "probe", "r2l", "u2r"
"""

