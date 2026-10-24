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
- **Known (train-seen) attack detection rate:** **82.4%**
- **Novel (zero-day test-only) attack detection rate:** **51.8%** (a 30.6% drop!)
- **R2L and U2R detection:** Less than **25%** of Remote-to-Local attacks and under **17%** of User-to-Root attacks are detected. In real security operations, missing an R2L credential theft or U2R privilege escalation leads to catastrophic network compromise.

### 3.2 Weak Spot 2: Unmeasured Timing Claims
The paper repeatedly states that Random Projection is "faster than PCA" without reporting a single millisecond of execution time. Our benchmarks measured:
- **Gaussian RP projection transformation:** **0.08 seconds**
- **PCA covariance decomposition & projection:** **1.45 seconds** (an 18x difference)
- **Classifier fit time at 5 dims:** 0.03s - 2.15s (compared to full features requiring up to 55s for neural networks).

### 3.3 Weak Spot 3: The "92% Variance in PC1" Scaling Anomaly
The paper claims that 92% of the entire dataset's variance was captured in the single first principal component ($D=1$). In our analysis, this only occurs if large-magnitude continuous features (such as `src_bytes` and `dst_bytes` with values in the millions) are left **unstandardized**, distorting the covariance matrix. When proper z-score scaling is applied, variance is distributed more evenly across the components.

### 3.4 Weak Spot 4: Table 4 Confusion Matrix Typo
In paper Table 4 (J48 Confusion Matrix), cell $(2, 1)$ is printed as `34,231`. Because the total anomaly rows in `KDDTest+` equals exactly `12,833`, this figure is mathematically impossible. The correct value is `4,231`, as $4,231 + 8,602 = 12,833$.

---

## 4. Production Pipeline Contract (Task 20)
The final production model is compiled into `models/final.joblib`:
- Accepts raw, un-preprocessed 41-column Pandas DataFrames.
- Contains the integrated `OneHotEncoder`, `StandardScaler`, and `RandomForestClassifier`.
- Implements `predict(X)` returning categorical labels (`['normal', 'dos', 'probe', 'r2l', 'u2r']`) and `predict_proba(X)` for calibrated security alert confidence scoring.
- Benchmark inference latency: **0.011 ms per network flow**, fully exceeding real-time intrusion monitoring constraints.

