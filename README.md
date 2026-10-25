# Network Intrusion Detection System — ML Mini Project

Reproducing and extending Nabi & Zhou (2024), *Enhancing intrusion detection systems through dimensionality reduction: A comparative study of machine learning techniques for cyber security*.

## Run the system demo (Windows)

Double-click `run.bat` (or run it from a terminal). On first launch it creates `.venv`, installs the demo dependencies, creates `models/dummy.joblib` when the trained model is not present, then starts the FastAPI service and Streamlit dashboard. The dummy model emits repeatable random labels for interface demonstration only; it is not a trained detector and its metrics must not be reported.

In a second terminal, replay benchmark flows:

```powershell
.venv\Scripts\python.exe -m app.replay --limit 300 --rate 4
```

You can filter to one raw attack name with `--attack-type neptune`, change the batch size with `--batch-size 32`, and set `--rate 0` to disable pacing. The API docs are at `http://127.0.0.1:8000/docs`.

## API

- `GET /health` — service status and loaded model version
- `POST /predict` — batch of raw flows containing all 41 features
- `GET /alerts?limit=50` — recent attack predictions
- `GET /stats` — class counts, total flows, and predicted alert rate

## Project structure

```text
├── data/                  # NSL-KDD files (download separately)
├── src/                   # data, preprocessing, models, experiments
├── models/                # final.joblib and generated dummy.joblib
├── results/               # experiment JSON outputs
├── api/                   # FastAPI service and request schemas
├── app/                   # Streamlit dashboard and replay client
├── reports/               # report sections, figures, demo and viva material
├── docs/                  # plan and references
├── requirements.txt       # full experiment environment
└── requirements-demo.txt  # minimal API/replay/dashboard environment
```

## Full experiment setup

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

NSL-KDD sources and instructions are in `data/README.md`. Paper details and project scope are in `docs/plan.md`.
