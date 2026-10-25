# Paper vs project comparison

The paper results are from the transcribed tables. This project uses Weka for PART/BayesNet and scikit-learn for the other models; see the report for implementation differences.

| Setting | Model | Paper accuracy | Project accuracy | Paper FPR | Project FPR |
|---|---|---:|---:|---:|---:|
| Full features | BayesNet | 0.7140 | 0.7479 | 0.2550 | 0.0268 |
| Full features | NaiveBayes | 0.7310 | 0.5664 | 0.2350 | 0.0072 |
| Full features | J48 | 0.7910 | 0.7773 | 0.1850 | 0.0307 |
| Full features | PART | 0.7390 | 0.7831 | 0.2400 | 0.0763 |
| Full features | RandomForest | 0.7780 | 0.7650 | 0.2010 | 0.0272 |
| Gaussian RP (5) | BayesNet | 0.7840 | 0.7455 | 0.1600 | 0.0670 |
| Gaussian RP (5) | NaiveBayes | 0.7730 | 0.7312 | 0.2050 | 0.0216 |
| Gaussian RP (5) | J48 | 0.7960 | 0.7875 | 0.1700 | 0.0262 |
| Gaussian RP (5) | PART | 0.8200 | 0.7391 | 0.1620 | 0.0784 |
| Gaussian RP (5) | RandomForest | 0.7750 | 0.7716 | 0.2000 | 0.0196 |
