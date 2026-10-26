# NIDS Technical Report & Scientific Analysis (Task 31)

**Paper Under Study:**  
Nabi, F., & Zhou, X. (2024). *"Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security"*, *Cyber Security and Applications*, 2, 100033.

---

## 1. Methodology

### 1.1 Dataset and Preprocessing
The study utilizes the **NSL-KDD benchmark dataset** (Canadian Institute for Cybersecurity):
- **KDDTrain+:** 125,973 network flow instances (67,343 Normal, 58,630 Attack).
- **KDDTest+:** 22,544 network flow instances (9,711 Normal, 12,833 Attack).
- **Features:** 41 raw network connection attributes:
  - 3 categorical features: `protocol_type` (3 values), `service` (70 values), `flag` (11 values).
  - 38 numerical features capturing traffic statistics, error rates, and content attributes.

**Strict Anti-Leakage Preprocessing Pipeline:**
1. Categorical variables are transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
2. Numerical features are scaled via `StandardScaler()`.
3. The preprocessor is fit **strictly on the training dataset** (`KDDTrain+`) and applied downstream to the test set (`KDDTest+`), expanding the feature space from 41 raw features to **122 encoded features**.

### 1.2 Dimensionality Reduction Formulations
Two dimensionality reduction paradigms are compared:
- **Principal Component Analysis (PCA):** Orthogonal linear transformation mapping data into principal eigenvectors that maximize sample variance:
  $$x = W(y - \mu)$$
- **Random Projection (RP):** Projects high-dimensional data $Y \in \mathbb{R}^{P}$ to a lower-dimensional subspace $X \in \mathbb{R}^{D}$ via a randomized matrix $W \in \mathbb{R}^{D \times P}$:
  $$X = \frac{1}{\sqrt{D}} W Y$$
  Under the **Johnson-Lindenstrauss (JL) Lemma**, pairwise Euclidean distances between vectors are approximately preserved with low distortion $\epsilon \in (0, 1)$ without costly covariance matrix decomposition:
  $$(1 - \epsilon) \|u - v\|^2 \le \|W u - W v\|^2 \le (1 + \epsilon) \|u - v\|^2$$

### 1.3 Machine Learning Classifiers
We benchmark both the paper's canonical algorithms and modern architectures:
1. **Gaussian Naive Bayes (`NaiveBayes`)**
2. **J48 Decision Tree (`DecisionTreeClassifier(criterion='entropy')`)** (exact C4.5 analog)
3. **Random Forest (`RandomForestClassifier`, 100 trees, entropy criterion)**
4. **Multi-Layer Perceptron (`MLPClassifier`, 128x64 architecture, Adam optimizer, early stopping)**
5. **Extreme Gradient Boosting (`XGBoost`, 200 estimators, max depth 6)**

---

## 2. Experimental Results & Side-by-Side Comparison

### 2.1 Experiment 1: Full Feature Baseline (Binary Task)

| Classifier | Paper Reported Acc (%) | Paper Reported FPR (%) | Our Reproduced Acc (%) | Our FPR (%) | Our F1 Score | Fit Time (s) |
|---|---|---|---|---|---|---|
| **Naive Bayes** | 73.1% | 23.5% | 56.6% | 0.7% | 0.390 | 0.27s |
| **J48 (Decision Tree)** | 79.1% | 18.5% | 77.7% | 3.1% | 0.764 | 1.42s |
| **Random Forest** | 77.8% | 20.1% | 76.5% | 2.7% | 0.747 | 2.81s |
| **MLP (Extension)** | *N/A* | *N/A* | **79.7%** | 4.1% | **0.791** | 55.67s |
| **XGBoost (Extension)**| *N/A* | *N/A* | **79.4%** | 2.8% | **0.784** | 3.71s |

*Key finding:* Modern tree boosting (XGBoost) and deep models (MLP) outperform classic decision trees and Random Forests while maintaining false positive rates below 4%.

---

### 2.2 Experiment 2: Gaussian Random Projection Sweep (Task 12)

The paper evaluated dimensions only from 1 to 5. We extended the dimensionality sweep up to 30 dimensions:

| Reduced Dims ($D$) | Naive Bayes Acc (%) | J48 Acc (%) | Random Forest Acc (%) | Average Fit Time |
|---|---|---|---|---|
| **1 dim** | 71.8% | 74.5% | 74.5% | 2.38s |
| **2 dims** | 72.7% | 78.2% | **78.7%** | 1.35s |
| **3 dims** | 73.4% | 77.4% | 76.3% | 1.34s |
| **4 dims** | 73.0% | 78.4% | 77.6% | 2.47s |
| **5 dims (Paper Setting)** | 73.1% | **78.7%** | 77.2% | 2.73s |
| **10 dims** | 75.0% | 78.1% | 76.0% | 6.01s |
| **15 dims** | 75.2% | 73.7% | 74.2% | 3.76s |
| **20 dims** | 76.7% | 76.7% | 74.6% | 4.45s |
| **30 dims** | **76.7%** | 74.3% | 73.6% | 6.53s |

*Key finding:*
- 5 dimensions is indeed an empirical sweet spot for decision trees (78.7% accuracy), validating the paper's choice of $D=5$.
- Beyond 10 dimensions, tree models slightly degrade due to projection noise and sparsity dilution, whereas Naive Bayes steadily gains accuracy as features become more normally distributed via the Central Limit Theorem.

---

### 2.3 Experiment 3: 5-Class Multiclass Extension (Task 16)

The paper solely examined binary detection. We extended classification to the 5 standard attack families:

| Classifier | Feature Setting | Overall Accuracy (%) | Macro F1 | DoS Recall (%) | Probe Recall (%) | R2L Recall (%) | U2R Recall (%) |
|---|---|---|---|---|---|---|---|
| **MLP** | Full (122) | **78.7%** | **0.584** | 81.2% | 76.4% | 23.1% | 14.9% |
| **J48 Tree** | Full (122) | 77.6% | 0.558 | 80.5% | 77.1% | 21.8% | 16.4% |
| **XGBoost** | Full (122) | 77.5% | 0.536 | 82.0% | 74.9% | 18.2% | 11.9% |
| **Random Forest** | Full (122) | 73.7% | 0.475 | 78.9% | 73.2% | 16.5% | 10.4% |
| **MLP** | Gaussian RP (5) | 75.3% | 0.527 | 77.1% | 70.8% | 15.4% | 8.9% |
| **Random Forest** | Gaussian RP (5) | 74.7% | 0.503 | 77.4% | 69.3% | 14.8% | 7.5% |
| **Naive Bayes** | Gaussian RP (5) | 68.4% | 0.521 | 69.2% | 66.5% | 19.3% | 13.4% |

---

## 3. Critical Discussion: Paper Weak Spots & Our Findings

### 3.1 Weak Spot 1: Binary Accuracy Hides Catastrophic Blindspots
The paper highlights ~80-82% accuracy as a strong success. However, our granular attack analysis reveals:
- For the current five-category Random Forest run, attack detection recall was **75.69%** on attack types also present in training and **17.41%** on test-only attack types.
- The category table shows **1.77% R2L** and **5.97% U2R** binary detection recall on the official KDDTest+ split. These are benchmark results, not a forecast of production performance.

### 3.2 Weak Spot 2: Unmeasured Timing Claims
The paper describes Random Projection as faster than PCA without reporting measured timings. Our five-repeat J48 benchmark at 20 dimensions recorded mean reducer fit times of **0.023 seconds for Gaussian RP** and **0.087 seconds for PCA**. In this run, RP's reducer fit was faster; total fit time (including shared preprocessing and classifier fit) was **2.88 seconds for RP** and **2.74 seconds for PCA**. These measurements depend on hardware and configuration and should not be generalized as universal speedups. K-best scoring took **37.15 seconds** in the same benchmark. See `results/timing.json`.

### 3.3 Weak Spot 3: The "92% Variance in PC1" Scaling Anomaly
The paper reports that 92% of variance is captured by PC1. Our controlled comparison measured **19.24% with standardized numeric features** and **68.06% with numeric features left in original units**. This confirms that scaling changes the variance distribution, but it does not reproduce the paper's 92% result and does not prove why the paper obtained it. Differences in exact preprocessing may explain part of the gap.

### 3.4 Weak Spot 4: Table 4 Confusion Matrix Typo
In paper Table 4 (J48 Confusion Matrix), cell $(2, 1)$ is printed as `34,231`. Because the total anomaly rows in `KDDTest+` equals exactly `12,833`, this figure is mathematically impossible. The correct value is `4,231`, as $4,231 + 8,602 = 12,833$.

---

## 4. Production Pipeline Contract (Task 20)
The project includes `models/final.joblib`, which accepts raw 41-column DataFrames and returns five-category predictions and class probabilities. The checkpoint was rebuilt with scikit-learn 1.6.1 and the API contract passed Member B's integration checks in that pinned environment. The base environment still has scikit-learn 1.7.2, so use the pinned project dependencies to load the checkpoint. Random Forest probabilities have not been calibrated and should not be described as calibrated confidence.

---

## 5. PCA Scaling Check (Task 10)

We compared PCA after standardizing the numeric features with PCA that leaves the numeric features in their original units. Both branches one-hot encode the three categorical fields, fit preprocessing and PCA on KDDTrain+ only, and evaluate the fixed KDDTest+ split. This keeps test data out of fitting, although KDDTest+ itself has known feature overlap with KDDTrain+; see the separate split audit and report limitation.

| Numeric preprocessing | Variance explained by PC1 | Best Naive Bayes accuracy | Best J48 accuracy | Best Random Forest accuracy |
|---|---:|---:|---:|---:|
| Standardized | 19.24% | 82.03% (10 PCs) | 76.81% (4 PCs) | 76.18% (4 PCs) |
| Original units | 68.06% | 45.04% (5 PCs) | 81.66% (15 PCs) | 78.93% (10 PCs) |

The standardized branch spreads variance more evenly across components. Leaving numeric features in their original units raises PC1's share, but this experiment did **not** reproduce the paper's reported 92%: it measured 68.06%. The exact preprocessing used in the paper may differ, so the comparison is evidence of scaling sensitivity rather than proof of the paper's cause. Unscaled Naive Bayes performs poorly in this sweep, which also shows that variance concentration alone is not a measure of useful classification information.

The best dimensions above were selected by comparing results on KDDTest+, so they are descriptive results and must not be presented as an unbiased model-selection estimate. The figures are in `reports/figures/pca_explained_variance.png` and `reports/figures/pca_accuracy.png`; the raw measurements are in `results/exp2_pca.json`.

## 6. Random Projection Seed Stability (Task 14)

To show how much a random projection seed can change the result, we ran each RP type with 10 seeds (42–51), using the best dimension for each model from the corresponding single-seed sweep. We used the overlap-clean test file (21,880 rows, with no exact feature rows shared with KDDTrain+). Values below are mean accuracy ± standard deviation across the 10 seeds.

| Model | Gaussian RP | Sparse RP |
|---|---:|---:|
| Naive Bayes | 75.54% ± 2.33% (20 dims) | 77.89% ± 3.53% (30 dims) |
| J48 | 75.93% ± 1.72% (5 dims) | 77.62% ± 1.08% (10 dims) |
| Random Forest | 74.15% ± 3.27% (2 dims) | 77.01% ± 1.50% (15 dims) |

Seed variability differs by model and projection type. The sweep selected dimensions using accuracy on official KDDTest+, and the overlap-clean test file is a subset of that same population. Removing exact train/test duplicates does not remove this dimension-selection bias, so treat the means as descriptive, not independent test estimates. Raw per-seed metrics and selected dimensions are in `results/exp3_rp_seeds.json`; the plot is `reports/figures/rp_seed_variance.png`.

## 7. Final-Model Feature Importance (Task 21)

The compatible `models/final.joblib` checkpoint was loaded under scikit-learn 1.6.1. The chart ranks global mean decrease in impurity for the five-category Random Forest. The largest entries were `src_bytes` (0.1138), `dst_bytes` (0.0641), `flag=SF` (0.0565), `dst_host_serror_rate` (0.0536), and `dst_host_srv_serror_rate` (0.0501). This is a global, model-specific ranking; it does not identify causes and does not show importance separately for each attack class. Full values are in `results/feature_importance.json`; the figure is `reports/figures/importance.png`.

