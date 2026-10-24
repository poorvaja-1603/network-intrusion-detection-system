# Network Intrusion Detection System — ML Mini Project

Reproducing and extending: **"Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security"** — Nabi & Zhou (2024)

## Project Structure

```
ml-mini-project/
├── data/                  # NSL-KDD dataset (KDDTrain+, KDDTest+)
├── src/                   # Core source code
│   ├── data.py            # Data loading & preprocessing contract
│   ├── preprocess.py      # ColumnTransformer (OneHot + Scaler)
│   ├── models.py          # Model zoo (NB, DT/J48, RF, XGBoost, MLP)
│   ├── evaluate.py        # Metrics, timing, results writer
│   └── attack_map.py      # Attack name → category mapping
├── models/                # Saved models (final.joblib, dummy.joblib)
├── results/               # Experiment output JSONs
├── api/                   # FastAPI inference service
│   ├── main.py            # API endpoints
│   └── schemas.py         # Pydantic request/response schemas
├── app/                   # Streamlit dashboard & replay
│   ├── dashboard.py       # Streamlit UI
│   └── replay.py          # KDDTest+ replay client
├── notebooks/             # EDA and experiment notebooks
├── reports/               # Report sections & generated figures
│   └── figures/           # Plots, confusion matrices, charts
├── tests/                 # Unit and integration tests
├── docs/                  # Research paper, plan, references
├── configs/               # Experiment configs (hyperparams, sweeps)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NSL-KDD dataset
# Place KDDTrain+.txt and KDDTest+.txt in data/
# Source: https://www.unb.ca/cic/datasets/nsl.html

# 4. Run experiments
python -m src.run_experiments

# 5. Start the API
uvicorn api.main:app --reload

# 6. Launch the dashboard
streamlit run app/dashboard.py
```

## Dataset

**NSL-KDD** — 125,973 training rows, 22,544 test rows, 41 features.
- Binary task: Normal vs Anomaly
- 5-class task: Normal, DoS, Probe, R2L, U2R

## Reference

Nabi, F., & Zhou, X. (2024). Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security. *Cyber Security and Applications*, 2, 100033.

