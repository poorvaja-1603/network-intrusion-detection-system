# Results Directory

All experiment outputs are saved here as JSON files with a standard schema:

```json
{
  "model": "RandomForest",
  "setting": "full | pca | rp_gaussian | rp_sparse | kbest",
  "n_dims": 41,
  "seed": 42,
  "accuracy": 0.778,
  "fpr": 0.201,
  "precision": 0.80,
  "recall": 0.75,
  "f1": 0.77,
  "macro_f1": 0.76,
  "fit_seconds": 12.3,
  "confusion_matrix": [[9240, 471], [4231, 8602]]
}
```

## Expected Files
- `exp1_full_*.json` — Full features (Experiment 1)
- `exp2_pca_*.json` — PCA reduction (Experiment 2)
- `exp3_rp_gauss_*.json` — Gaussian RP (Experiment 3)
- `exp3_rp_sparse_*.json` — Sparse RP (Experiment 3)
- `exp3_rp_seeds.json` — Multi-seed RP stability
- `exp5_multiclass_*.json` — 5-class classification
- `exp6_modern_*.json` — XGBoost / MLP
- `exp7_kbest_*.json` — Feature selection baselines
- `exp8_gap.json` — Train CV vs test gap
- `timing.json` — Timing benchmarks
- `paper_reported.json` — Paper's reported numbers for comparison

