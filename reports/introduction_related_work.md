# Introduction, Related Work, and Dataset

## Abstract

This project reproduces and extends Nabi and Zhou's study of dimensionality reduction for network intrusion detection. We use the NSL-KDD training and fixed test splits to compare classifiers on the full one-hot encoded feature set, principal component analysis (PCA), Gaussian random projection (RP), and mutual-information feature selection. Numeric scaling and categorical encoding are fitted on the training split only. In the current binary experiments, the strongest full-feature result is the multilayer perceptron at 79.7% accuracy; among the tested five-dimensional Gaussian RP models, J48 reaches 78.7%. The feature-selection comparison shows that its best J48 result is 78.2% at 10 selected features, while PCA with Naive Bayes reaches 82.0% at 10 components and a 11.4% false-positive rate. For the five-class extension, the full-feature MLP reaches a macro F1 of 0.584. These results show that performance depends on both the classifier and representation, and that accuracy alone does not describe minority attack detection. The implementation uses scikit-learn approximations for some Weka classifiers and one fixed seed for the current projection sweep; those differences limit direct numerical comparison with the paper.

## 1. Introduction

Network intrusion detection systems (IDSs) monitor traffic or host activity for evidence of malicious behavior. Signature-based systems match activity against known patterns and can detect those patterns precisely, but they require updates to recognize new attacks. Anomaly-based systems model expected behavior and flag deviations, which can help identify unfamiliar attacks but may also increase false alarms [1, 3]. Machine-learning classifiers provide one way to build anomaly-based detectors from labeled examples of normal and attack traffic.

The quality and cost of an IDS depend on the representation supplied to its classifier. High-dimensional data can increase fitting time and include redundant or weakly informative features. Nabi and Zhou evaluate full features alongside PCA and random projection on NSL-KDD, reporting that Gaussian random projection performs well for several classifiers and generally takes less training time than PCA. Their study also reports binary accuracy and false-positive rate, with J48 on full features reaching 79.1% accuracy and PART after Gaussian RP reaching 82.0% [Nabi and Zhou, 2024].

This project reproduces the central comparison in Python and extends it in several directions: an expanded RP dimension sweep, timing measurements, mutual-information feature selection, five-class attack-family evaluation, and analysis of generalization to the fixed KDDTest+ split. We evaluate accuracy together with false-positive rate, precision, recall, F1, and macro F1. The experiments use train-only fitting for encoding, scaling, and dimensionality reduction to avoid test leakage. The implementation substitutes an entropy-based scikit-learn decision tree for Weka J48; Weka PART and BayesNet are not included in the current Python runs. Results should therefore be interpreted as a controlled reproduction and extension, not as an exact Weka rerun.

## 2. Related Work

Machine-learning intrusion detection has been studied across a range of datasets and supervised classifiers. Surveys describe decision trees, random forests, Bayesian methods, support-vector machines, neural networks, and ensemble approaches, and identify dataset choice and evaluation protocol as major factors in reported performance [3, 5]. Belavagi and Muniyal compare supervised classifiers for intrusion detection, while Kumar and Batth study feature-selection techniques with classifiers including J48 and Naive Bayes [6, 7]. These works motivate reporting class-sensitive measures and examining reduced feature sets alongside full-feature baselines.

PCA is a standard linear reduction method that projects data onto directions of high variance. Earlier intrusion-detection studies have paired PCA with classifiers such as support-vector machines and random forests [9, 10]. Random projection offers a different tradeoff: it maps observations through a randomly generated matrix and can reduce dimensionality without estimating principal directions. Juvonen and Hamalainen apply random projection to network-log anomaly detection [14]. Nabi and Zhou compare Gaussian and sparse RP with PCA on NSL-KDD and report favorable accuracy and training-time results for Gaussian RP in several settings.

The present work follows that comparison while adding three checks that the source paper does not fully report: timing by pipeline stage, broader RP dimensionalities and a fixed random seed, and attack-family metrics beyond binary accuracy. It also adds a mutual-information top-k baseline, which selects original features rather than constructing projected components. These extensions help distinguish computational savings from classifier performance and make minority-class behavior visible.

## 3. Dataset

NSL-KDD is a cleaned benchmark derived from KDD Cup 1999 traffic and was designed to reduce the influence of redundant records in the original collection [16]. The dataset represents connection records from a simulated military-network environment. Each row contains 41 input features, an attack-name label, and a difficulty value. The three categorical inputs are `protocol_type`, `service`, and `flag`; the remaining 38 inputs are numeric.

The standard files used here contain 125,973 training rows in `KDDTrain+.txt` and 22,544 test rows in `KDDTest+.txt`. The training set contains 67,343 normal and 58,630 attack records; the test set contains 9,711 normal and 12,833 attack records. The project loader drops the difficulty field and maps the attack-name label to either a binary target (`normal` or `attack`) or one of five families (`normal`, `dos`, `probe`, `r2l`, `u2r`).

The fixed test set includes attack names that do not occur in the training set, so it evaluates generalization to previously unseen attack types as well as to new records. The EDA notebook reports those names and their test counts. Three categorical features are one-hot encoded and numeric features are standardized; fitting occurs on KDDTrain+ only. This expands the model input from 41 raw fields to 122 encoded features in this repository's preprocessing pipeline.

The original UNB dataset page identifies the NSL-KDD files but now states that they are no longer available directly. The project files were obtained from the Zenodo mirror recorded in `data/README.md`; their published MD5 checksums match the downloaded copies. As with other NSL-KDD studies, results should be read in light of the dataset's age, simulated origin, and fixed attack distribution rather than treated as a measurement of current production traffic.

## References

[1] A. Verma and V. Ranga, “Machine learning based intrusion detection systems for IoT,” *Wireless Personal Communications*, 111(4), 2287–2310, 2020.

[3] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, “Survey of intrusion detection systems: techniques, datasets and challenges,” *Cyber Security*, 2(1), 1–22, 2019.

[5] H. Liu and B. Lang, “Machine learning and deep learning methods for intrusion detection systems: a survey,” *Applied Sciences*, 9(20), 4396, 2019.

[6] M. C. Belavagi and B. Muniyal, “Performance evaluation of supervised machine learning algorithms for intrusion detection,” *Procedia Computer Science*, 89, 117–123, 2016.

[7] K. Kumar and J. S. Batth, “Network intrusion detection with feature selection techniques using machine-learning algorithms,” *International Journal of Computer Applications*, 150(12), 1–13, 2016.

[9] N. S. K. H. Praneeth, N. M. Varma, and R. R. Naik, “Principle component analysis based intrusion detection system using support vector machine,” in *2016 IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology*, 1344–1350, 2017. https://doi.org/10.1109/RTEICT.2016.7808050.

[10] S. Waskle, L. Parashar, and U. Singh, “Intrusion detection system using PCA with random forest approach,” in *2020 International Conference on Electronics and Sustainable Communication Systems*, 803–808.

[14] A. Juvonen and T. Hamalainen, “An efficient network log anomaly detection system using random projection dimensionality reduction,” in *2014 6th International Conference on New Technologies, Mobility and Security*, 1–5.

[16] Canadian Institute for Cybersecurity, “ISCX NSL-KDD dataset 2009,” University of New Brunswick. https://www.unb.ca/cic/datasets/nsl.html.

F. Nabi and X. Zhou, “Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security,” *Cyber Security and Applications*, 2 (2024), 100033. https://doi.org/10.1016/j.csa.2023.100033.
