@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Creating the project environment...
  py -m venv .venv || exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt || exit /b 1
if not exist models\final.joblib (
  if not exist models\dummy.joblib .venv\Scripts\python.exe -m api.create_dummy || exit /b 1
)
start "NIDS API" /min .venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.venv\Scripts\python.exe -m streamlit run app\dashboard.py
