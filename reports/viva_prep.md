# Viva Preparation

## Likely questions and short answers

1. **Why NSL-KDD instead of KDD99 or CIC-IDS2017?** NSL-KDD removes much of KDD99's duplicate-record bias and is small enough for repeatable experiments. It is old and synthetic, so it does not represent today's full traffic.
2. **Why is train cross-validation accuracy much higher than KDDTest+?** KDDTest+ includes attack types absent from training and is intentionally difficult. Cross-validation reuses the training distribution.
3. **What is false positive rate?** FP/(FP+TN): the share of normal flows incorrectly flagged as attacks. High FPR creates alert fatigue.
4. **How does random projection work?** It multiplies feature vectors by a random matrix to reduce dimension while approximately preserving pairwise distances, as described by the Johnson–Lindenstrauss lemma.
5. **Why might RP be faster than PCA, and what did we measure?** RP avoids computing an eigendecomposition. Our timing measures projection and classifier fit under the same data and hardware conditions.
6. **Why do random-projection results vary?** The projection matrix changes with its random seed. We log fixed seeds and repeat the best dimensions to report mean and standard deviation.
7. **Why might one PCA component look unusually strong?** Unscaled high-magnitude variables can dominate variance. We compare scaled and unscaled PCA; variance explained does not guarantee useful class separation.
8. **Why add five-class evaluation?** Binary detection hides which families are missed. Macro F1 and per-class recall give rare R2L and U2R attacks equal visibility.
9. **How did we avoid leakage?** Encoders, scalers, and projections are fitted on training data only, then applied to held-out test data.
10. **How close is sklearn's entropy tree to J48?** Both are tree learners using information gain, but pruning, missing-value handling, and implementation details differ; it is an approximation, not an exact Weka reproduction.
11. **What about a new attack type?** The classifier can only assign a known family. Unknown-attack detection would need an explicit novelty detector or retraining process.
12. **What does “live” mean in this demo?** The replay sends stored KDDTest+ flows in batches. There is no live packet capture.
13. **What would production deployment require?** A flow extractor, secure transport, durable alert storage, drift monitoring, access controls, and a retraining and validation process.
14. **Why compare Gaussian and sparse RP?** Their random matrix structures have different computation and sparsity tradeoffs. We compare measured metrics and timing rather than assuming either wins.
15. **What are the paper's key evidence gaps?** It uses a single random-projection run, reports no timing, evaluates binary labels only, and makes claims about overfitting/noise reduction without direct experiments.
16. **Why use false positive rate alongside accuracy?** Accuracy can hide operationally expensive false alarms; FPR isolates that burden among normal traffic.
17. **What is the role of one-hot encoding?** It converts protocol, service, and flag categories into numeric indicators before scaling and projection.
18. **Why keep preprocessing inside the final pipeline?** It ensures training and inference apply the same transformations and prevents downstream callers from preprocessing inconsistently.
19. **What does the confidence field mean?** It is the largest class probability returned by the model, not a guarantee of correctness or calibration.
20. **What is the API's alert rate?** The fraction of processed flows predicted as non-normal; it is a model output statistic, not the true attack prevalence.

## Glossary

- **Attack family:** A broad class such as DoS, Probe, R2L, or U2R.
- **False positive (FP):** A normal flow incorrectly labeled as an attack.
- **False negative (FN):** An attack flow incorrectly labeled normal.
- **FPR:** FP divided by all true normal flows, FP/(FP+TN).
- **Macro F1:** Mean F1 across classes, weighting each class equally.
- **One-hot encoding:** Numeric indicator columns representing categorical values.
- **PCA:** Linear projection onto directions of greatest training-data variance.
- **Random projection (RP):** Dimension reduction using a randomly generated matrix.
- **Data leakage:** Use of held-out information while fitting preprocessing or models.
- **KDDTest+:** Fixed NSL-KDD test split with attack types not all represented in training.
- **Replay:** Sending saved flows through the service as a simulation of streaming input.
- **Alert rate:** Portion of processed records predicted to be attacks.
