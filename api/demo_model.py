"""Deterministic random baseline used to exercise the demo API."""
import numpy as np

LABELS = ("normal", "dos", "probe", "r2l", "u2r")


class DemoModel:
    classes_ = np.asarray(LABELS)

    def predict_proba(self, X):
        # Fixed seed keeps demo screenshots and rehearsals repeatable.
        rng = np.random.default_rng(42)
        values = rng.uniform(0.01, 1.0, size=(len(X), len(LABELS)))
        return values / values.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

