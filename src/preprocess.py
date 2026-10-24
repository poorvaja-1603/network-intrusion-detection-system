"""
preprocess.py — Preprocessing pipeline

ColumnTransformer with:
  - OneHotEncoder for categorical columns (protocol_type, service, flag)
  - StandardScaler for numeric columns
  
Fit on train only. Transform test with the fitted objects.
"""

