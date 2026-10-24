"""
evaluate.py — Evaluation utilities

Computes: accuracy, FPR, per-class precision/recall/F1, macro F1,
           confusion matrix, timing wrapper.

Writes results to results/<experiment_name>.json with standard keys:
  model, setting, n_dims, seed, accuracy, fpr, precision, recall, f1,
  macro_f1, fit_seconds, confusion_matrix
"""

