"""
models.py — Model zoo

get_models() returns a dict of classifiers:
  - GaussianNB           (≈ Naive Bayes)
  - DecisionTreeClassifier(criterion="entropy")  (≈ J48 / C4.5)
  - RandomForestClassifier
  - XGBClassifier         (extension)
  - MLPClassifier          (extension)

All with fixed random_state for reproducibility.
"""

