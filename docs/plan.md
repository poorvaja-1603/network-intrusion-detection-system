# NIDS Mini Project Plan (v2, paper specific)

Paper: Nabi and Zhou, "Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security", Cyber Security and Applications 2 (2024) 100033.

Team: 4 members (B and C currently available) | Today: Sat 24 Oct 2026 | Submit: Tue 27 Oct 2026
Code freeze: Monday night. Tuesday is buffer, rehearsal, submission.

Team members are referred to as **A**, **B**, **C**, **D** below. Swap in real names.

---

## 1. What the paper does (the part we must reproduce)

- Dataset: NSL-KDD. 125,973 training rows (KDDTrain+), 22,544 test rows (KDDTest+), 41 features.
- Task: **binary** only (normal vs anomaly).
- Classifiers (all in Weka): BayesNet, Naive Bayes, J48, PART, Random Forest.
- Three feature settings: full 41 features, PCA, Random Projection (Gaussian and Sparse matrices).
- Reported results to compare against:

| Setting | Best model | Accuracy | FPR |
|---|---|---|---|
| Full features | J48 | 79.1% | 18.5% |
| PCA (1 component) | Naive Bayes | 78.8% | 19.0% |
| Gaussian RP (5 dims) | PART | 82.0% | 16.2% |
| Sparse RP (5 dims) | Random Forest | 77.3% | 20.5% |

Full per model tables are in the paper (Tables 3, 5, 7, 10). Copy them into `results/paper_reported.json` (task 4 area) so our tables can sit next to them.

---

## 2. Weak spots in the paper (this is where our value comes from)

1. **Binary only.** Accuracy of about 80% hides that anomaly recall is only about 67 to 70% (roughly 30% of attacks are missed). No attack category analysis.
2. **No timing numbers.** The paper claims RP is faster than PCA but reports no measured times. We will measure.
3. **Random Projection is random, but results are from one run.** No seeds, no mean or std. We will run many seeds.
4. **Only 1 to 5 dimensions tried.** We will sweep further (5, 10, 15, 20, 30, full).
5. **PCA oddity.** They report 92% variance in the first component and best accuracy at 1 dimension. That is suspicious and often means unscaled features dominate. We will redo PCA with correct scaling and check.
6. **Table 4 typo.** J48 confusion matrix shows 34,231 where it must be 4,231 (anomaly row must sum to 12,833).
7. **Unsupported claims.** Conclusion says PCA "prevents overfitting" and gives "noise reduction" with no experiment backing it.
8. **KDDTest+ is a hard test set** (contains attack types not in training). Accuracy near 80% is normal for simple models here, so do not treat it as a bug. Cross validation on the training file gives about 99%, and that gap is itself a finding.

---

## 3. Project scope

**Reproduce:** the paper's 3 settings (full, PCA, RP) in Python with the paper's metrics (accuracy, FPR, confusion matrix, precision, recall, F1).

**Extend (our contribution):**
- E1. Multi seed Random Projection (mean and std) and extended dimension sweep
- E2. Measured timing: full vs PCA vs RP vs feature selection
- E3. Modern models (XGBoost or LightGBM, MLP) on the same settings
- E4. 5 class classification (Normal, DoS, Probe, R2L, U2R) with macro F1 and per attack recall
- E5. Feature selection baseline (mutual information top k) vs PCA vs RP
- E6. Train CV vs KDDTest+ gap analysis and error analysis by attack type

**System:** FastAPI inference service + replay of KDDTest+ flows + Streamlit dashboard.

**Not doing:** real attack generation, live packet capture, deep learning research (only a small MLP).

### Honest deviations to state in the report
- Weka's J48 maps to sklearn `DecisionTreeClassifier(criterion="entropy")` (close to C4.5, not identical).
- sklearn has no PART or BayesNet. Options: (a) run those two in the Weka GUI on our exported CSV (task 11, easiest and exact), or (b) substitute and say so.
- Categorical columns (protocol_type, service, flag) are one hot encoded before scaling and projection. This raises the input dimension above 41. Document it.

---

## 4. Contracts (agree on these Saturday morning so everyone can work alone)

**Data contract (`src/data.py`, now maintained by C, task 3)**
```
FEATURE_COLUMNS: list of the 41 raw feature names (in dataset order)
load_nsl_kdd() -> X_train, y_train_bin, y_train_cat, X_test, y_test_bin, y_test_cat
   X_*        : pandas DataFrame, 41 raw columns, no label, no difficulty column
   y_*_bin    : 0 = normal, 1 = attack
   y_*_cat    : one of "normal", "dos", "probe", "r2l", "u2r"
```

**Model contract (`models/final.joblib`, now maintained by C, task 20)**
A single sklearn Pipeline that takes the **raw** 41 column DataFrame and supports `predict` and `predict_proba`. Preprocessing and any projection live inside the pipeline. Nobody downstream should ever re preprocess.

**API contract (owned by C, task 23)**
```
POST /predict   {"flows": [ {<41 raw features>}, ... ]}
  ->            {"predictions": [{"label": "dos", "is_attack": true, "confidence": 0.97}], "latency_ms": 4.2}
GET  /health    {"status": "ok", "model_version": "v1"}
GET  /alerts?limit=50   latest attack predictions
GET  /stats     counts per label, total flows, alert rate
```

**Results contract (all experiment code writes here)**
`results/<experiment_name>.json` with keys: `model`, `setting` (full, pca, rp_gaussian, rp_sparse, kbest), `n_dims`, `seed`, `accuracy`, `fpr`, `precision`, `recall`, `f1`, `macro_f1`, `fit_seconds`, `confusion_matrix`. B and C read from here for plots and the dashboard.

---

## 5. Master task list (numbered)

Slots: **Sat**, **Sun**, **Mon**. "Needs" means what must exist before you can start. Until it exists, use the stand in noted.

### Setup and data
| # | Task | Output | Slot | Needs |
|---|---|---|---|---|
| 1 | Create repo, folder structure, `requirements.txt`, `.gitignore`, README skeleton, branch rules | Repo link shared with all | Sat early | none |
| 2 | Download NSL-KDD (KDDTrain+ and KDDTest+), place in `data/`, write `data/README.md` with source (UNB CIC page cited as ref 16 in the paper) | Raw files + README | Sat early | none |
| 3 | Write `src/data.py` per the data contract (column names, drop difficulty column, binary and category labels, dedupe check) | `data.py` | Sat | 2, 4 |
| 4 | Write `src/attack_map.py`: dict mapping every attack name (including test only ones) to dos, probe, r2l, u2r. Also type the paper's reported tables into `results/paper_reported.json` | `attack_map.py`, `paper_reported.json` | Sat early | 2 |
| 5 | EDA notebook: class balance train vs test, attack types present only in test, categorical cardinality, zero variance columns, feature scale ranges (src_bytes vs rates) | `01_eda.ipynb` | Sat | 2 |
| 6 | Write `src/preprocess.py`: ColumnTransformer with OneHot (3 categorical cols) and scaler (fit on train only) | `preprocess.py` | Sat | 3 |
| 7 | Write `src/models.py`: `get_models()` returning the model zoo (GaussianNB, entropy DecisionTree as J48, RandomForest, plus extras) with fixed seeds | `models.py` | Sat | none |
| 8 | Write `src/evaluate.py`: accuracy, FPR, per class P/R/F1, macro F1, confusion matrix, timing wrapper, writes the results JSON | `evaluate.py` | Sat | none |
| 9 | Reproduce Experiment 1 (full features, all models) and compare to paper Table 3 | `results/exp1_*.json`, comparison table | Sat night | 3, 6, 7, 8 |

### Reproduction experiments
| # | Task | Output | Slot | Needs |
|---|---|---|---|---|
| 10 | PCA experiment: explained variance curve, accuracy for 1 to 5 components (and more), check the paper's "92% in first PC" claim with and without scaling | `results/exp2_pca_*.json`, variance plot | Sun | 6, 7, 8 |
| 11 | Export the preprocessed train and test CSV and run **PART** and **BayesNet** in Weka GUI (full features and Gaussian RP 5 dims). Record accuracy and FPR. Skip if Weka is not installed by Sunday noon | `results/weka_*.json` | Sun | 6 |
| 12 | Gaussian Random Projection sweep: dims 1 to 5 as in the paper, then 10, 15, 20, 30 | `results/exp3_rp_gauss_*.json` | Sun | 6, 7, 8 |
| 13 | Sparse Random Projection sweep with the same dims | `results/exp3_rp_sparse_*.json` | Sun | 6, 7, 8 |
| 14 | Multi seed RP: repeat best dims over 10 seeds, report mean and std, plot variance | `results/exp3_rp_seeds.json`, plot | Sun | 12, 13 |
| 15 | Timing benchmark: fit time of projection and of classifier, for full, PCA, RP, kbest. Repeat 5 times, report mean | `results/timing.json` | Sun | 6, 7 |

### Extensions
| # | Task | Output | Slot | Needs |
|---|---|---|---|---|
| 16 | 5 class experiment (normal, dos, probe, r2l, u2r) on full features and best reduction. Macro F1 and per class recall | `results/exp5_multiclass_*.json` | Sun | 6, 7, 8 |
| 17 | Modern models: XGBoost or LightGBM and a small MLP on full and RP settings | `results/exp6_modern_*.json` | Sun | 6, 8 |
| 18 | Feature selection baseline: mutual information top k (k = 5, 10, 20) vs PCA vs RP | `results/exp7_kbest_*.json` | Sun | 6, 8 |
| 19 | Generalisation gap: 5 fold CV on train vs KDDTest+ score, for 3 models. Also score on the hard subset if time allows | `results/exp8_gap.json`, plot | Sun | 6, 7, 8 |
| 20 | Pick the final model from results, fit the full pipeline on raw features, save `models/final.joblib` with `predict` and `predict_proba`. **Deadline Sunday 6 pm** | `final.joblib`, `metrics.json` | Sun 6 pm | 9 to 19 |
| 21 | Feature importance or SHAP for the final model (top features per attack class if 5 class) | `reports/figures/importance.png` | Mon am | 20 |
| 22 | Error analysis: per attack type recall, which attacks are missed (expect R2L and U2R), 10 example false negatives and why | `reports/error_analysis.md` | Mon am | 20 |

### System and demo
| # | Task | Output | Slot | Needs |
|---|---|---|---|---|
| 23 | FastAPI app per API contract with pydantic schema on the 41 raw columns | `api/main.py`, `schemas.py` | Sat | 1 |
| 24 | Dummy model stub (`models/dummy.joblib`, random labels) so API and dashboard can start before the real model exists | `dummy.joblib` | Sat | 23 |
| 25 | `app/replay.py`: reads KDDTest+ rows, posts batches to the API at a configurable rate, option to filter by attack type | `replay.py` | Sun | 23, 24 |
| 26 | Alert store (SQLite or in memory) plus `/alerts` and `/stats` endpoints | updated `main.py` | Sun | 23 |
| 27 | Streamlit dashboard: live alert feed, counts per class, timeline, confidence, and a "Model metrics" page that reads `results/*.json` | `app/dashboard.py` | Sun to Mon | 24, results contract |
| 28 | One command run script (`run.sh` or Makefile) and a fresh clone test on a clean environment | `run.sh`, tested | Mon | 20, 25, 27 |
| 29 | Tests: data shapes, no train/test overlap, API schema, model loads and predicts on 5 sample rows | `tests/` | Mon | 3, 23 |

### Report, slides, demo
| # | Task | Output | Slot | Needs |
|---|---|---|---|---|
| 30 | Figures and tables: before vs after bar charts, confusion matrices, accuracy vs dimension, timing chart, paper vs ours table (all from the results JSONs) | `reports/figures/*.png`, tables | Mon | results JSONs |
| 31 | Report: methodology, results, discussion, limitations (our analysis of the paper's weak spots) | report sections | Mon | 9 to 22 |
| 32 | Report: abstract, introduction, related work (use the paper's references), dataset description | report sections | Mon | 5 |
| 33 | Report: system design and demo section, architecture diagram | report section, diagram | Mon | 23 to 28 |
| 34 | Slide deck, 10 to 12 slides (deferred; prepare later) | `slides.pptx` or PDF | Later | 30 |
| 35 | Demo script (3 minute flow) and a 2 minute backup screen recording | script + video file | Mon night | 28 |
| 36 | Viva prep document: 20 likely questions with short answers (see section 9) and a one page glossary | `viva_prep.md` | Mon | 31 |

---

## 6. Current task owners after A and D became unavailable

Members A and D are unavailable. Completed files and results from their work stay in the project and retain their contribution history; they are not being redone. Remaining responsibilities are assigned to B and C below. Task 34 (slides) stays deferred for later.

| Person | Current task numbers | Count |
|---|---|---|
| **B** | **2, 4, 5, 10, 11, 14, 15, 18, 19, 21, 22, 29, 30, 31, 32** | 15 |
| **C** | **1, 3, 6, 7, 8, 9, 12, 13, 16, 17, 20, 23, 24, 25, 26, 27, 28, 33, 35, 36** | 20 |
| **Deferred** | **34** | — |

Tasks 1–33 and 35–36 are assigned exactly once. Completed A outputs for tasks 3, 6–9, 12, 16, 20, 22, and 31 remain complete; C maintains the shared ML pipeline and model artifacts, while B maintains the error-analysis and report work. D outputs already completed by B/C (19, 21, 27) remain complete.

### Handoffs and remaining work

| Task | Owner | Status / handoff |
|---|---|---|
| 10 — PCA experiment | B | Pending; use C's shared preprocessing and save the variance curve and scores. |
| 13 — Sparse RP sweep | C | Pending; follow task 12's format, use fixed seeds, and save results in the shared JSON schema. |
| 14 — Multi-seed RP | B | Pending; depends on the RP sweep results from tasks 12 and 13; report mean and standard deviation. |
| 17 — Modern models | C | Pending; run supported models with shared preprocessing/results format and note any model the environment cannot support. |
| 19 — Generalisation gap | B | Completed; rerun only if the shared pipeline or data contract changes. |
| 20 — Final model | C | Existing model code and checkpoint are present, but the checkpoint currently fails to load under scikit-learn 1.7.2 because it was saved with 1.6.1. Regenerate it with the agreed environment and verify the API integration. |
| 21 — Feature importance | B | Initial figure/results are complete from the documented configuration retrained in memory. Regenerate from C's compatible checkpoint when task 20 is fixed. |
| 27 — Streamlit dashboard | C | Existing implementation; continue integration with API/replay and available results. |
| 29 — Integration tests | B | Data/overlap checks pass. Rerun API/model checks after C resolves task 20 and the app imports successfully. |
| 34 — Slide deck | Deferred | Prepare later when the team is ready to make slides. |

Notes on ownership:
- C takes maintenance of shared `data.py`, `preprocess.py`, `models.py`, `evaluate.py`, and final-model code so B's remaining experiments can use stable interfaces.
- B keeps the bounded experiment, analysis, testing, figures, and report tasks that build on B's existing project contributions.
- C owns API/dashboard integration and the shared model-version repair. B reruns the dependent checks after that handoff.
- Keep changes scoped to each owner's files and coordinate shared-file edits before making them.

---

## 7. Timeline

**Saturday (today)**
- Morning (first 1 hr, all): read the paper together, confirm contracts in section 4, C creates the repo (1).
- Early: 2, 4 (B), 7, 8 (C), 23 (C).
- Midday: 3, 6 (C), 5 (B), 24 (C).
- Night: 9 done (C). Checkpoint: data loads, Experiment 1 table exists, API returns dummy predictions.

**Sunday**
- Morning to afternoon: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 all in parallel. 25, 26, 27 for the system side.
- Hard stop for experiments at 6 pm. C owns task 20 (`final.joblib`) and swaps the dummy model for the compatible final model.
- Night checkpoint: replay streams real KDDTest+ flows through the API and the dashboard shows real predictions.

**Monday**
- Morning: 21, 22 (analysis), 30 (figures), 29 (tests), 27 polish.
- Afternoon: 31, 32, 33 (report), 36 (viva doc). Task 34 (slides) is deferred for later.
- Evening: 28 fresh clone test, 35 demo dry run and recording. **Freeze.**

**Tuesday**
- Final dry run, fix typos, export report PDF, submit early.

---

## 8. Technical rules (to avoid wasting time)

- Fit scaler, encoder, PCA and RP on **train only**. Transform test with the fitted objects.
- Always use `random_state` and log the seed.
- KDDTest+ is a fixed test set. Do not tune on it. Tune on a validation split taken from train, or use CV on train.
- Report accuracy, FPR, **and** per class recall and macro F1. Accuracy alone is what the paper did and it hides missed attacks.
- FPR = FP / (FP + TN) with "attack" as the positive class, same as the paper.
- Expect simple models to land around 75 to 82% on KDDTest+. If something shows above 90% on KDDTest+, check for leakage before celebrating.
- RP and PCA operate on numeric vectors, so apply them after one hot encoding and scaling.
- Git: one branch per person, merge to `main` at lunch and at night. Never edit someone else's file without telling them.

---

## 9. Likely viva questions

1. Why NSL-KDD and not KDD99 or CIC-IDS2017? What are NSL-KDD's limitations (age, synthetic origin)?
2. Why is accuracy on KDDTest+ so much lower than cross validation accuracy on train?
3. What is FPR and why does it matter more in an IDS than in most classifiers?
4. How does Random Projection work, and what does the Johnson-Lindenstrauss lemma say?
5. Why can RP be faster than PCA? What did we actually measure?
6. Why is the RP result not reproducible without a fixed seed, and how did we handle that?
7. Why did the paper's PCA work best with 1 component, and do we agree?
8. Why did we add 5 class evaluation? Which attacks were hardest and why (R2L, U2R)?
9. How did we prevent data leakage?
10. What is the difference between J48 and sklearn's decision tree?
11. What happens when a new attack type appears that was not in training?
12. How does the live demo work, and what does "live" mean here (replay, not real traffic)?
13. What would be needed to deploy this on real traffic (flow extraction, drift, retraining)?
14. Why Gaussian RP over Sparse RP according to the paper, and did our results agree?
15. What is the weakness in the paper's conclusions (single run, no timing, binary only)?

---

## 10. Risks and cut list

| Risk | Mitigation |
|---|---|
| sklearn results differ from the paper | Expected. Report both, explain Weka vs sklearn differences, do not tune to match |
| PART and BayesNet unavailable | Task 11 via Weka GUI, otherwise state the substitution clearly |
| Someone blocked on shared code | Use stand ins (dummy model, example JSONs). C is the shared-code unblock owner; coordinate promptly. |
| Integration surprises on Monday | Real model plugged in Sunday evening, not Monday |
| Not enough time | Cut in this order: 17 (modern models), 18 (kbest), 21 (SHAP), 11 (Weka), dashboard polish |

**Never cut:** 3, 6, 8, 9, 12, 16, 20, 23, 25, 27, 31, and a working replay demo.

---

## 11. Definition of done

- [ ] Paper's three settings reproduced with a side by side table (paper vs ours)
- [ ] Multi seed RP result with mean and std
- [ ] Timing table with measured numbers
- [ ] 5 class results with per attack recall and error analysis
- [ ] `models/final.joblib` loads in the API and dashboard shows live predictions
- [ ] Fresh clone runs from README instructions
- [ ] Report PDF and backup demo recording
- [ ] Slide deck (task 34; deferred until later)
- [ ] Every member can explain the full pipeline, not only their own tasks
