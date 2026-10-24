"""
models.py — Model zoo for NIDS experiments.

get_models(task="binary") returns a dict of classifiers with fixed seeds.
Supports both binary and multiclass tasks.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

SEED = 42


def get_models(task: str = "binary") -> dict:
    """
    Return a dictionary of model_name -> sklearn estimator.

    Parameters
    ----------
    task : str
        "binary" or "multiclass" — some models may adjust params.

    Returns
    -------
    dict[str, estimator]
    """
    models = {
        # --- Paper models ---
        "NaiveBayes": GaussianNB(),
        "J48": DecisionTreeClassifier(
            criterion="entropy",       # C4.5 uses information gain (entropy)
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=SEED,
            n_jobs=-1,
        ),
        # --- Extension models ---
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        ),
    }

    # XGBoost — optional dependency
    try:
        from xgboost import XGBClassifier

        xgb_params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )
        if task == "multiclass":
            xgb_params["objective"] = "multi:softprob"
        else:
            xgb_params["objective"] = "binary:logistic"

        models["XGBoost"] = XGBClassifier(**xgb_params)
    except ImportError:
        pass

    return models


def get_paper_models() -> dict:
    """Return only the models the paper used (for reproduction experiments)."""
    return {
        "NaiveBayes": GaussianNB(),
        "J48": DecisionTreeClassifier(
            criterion="entropy",
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
