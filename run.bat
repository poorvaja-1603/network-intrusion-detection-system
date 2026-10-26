@echo off
setlocal
if exist venv\Scripts\python.exe (
  set "PY_EXE=venv\Scripts\python.exe"
) else if exist .venv\Scripts\python.exe (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  echo Creating the project environment...
  py -m venv .venv || exit /b 1
  set "PY_EXE=.venv\Scripts\python.exe"
)
%PY_EXE% -m pip install -r requirements-demo.txt || exit /b 1
if not exist models\final.joblib (
  if not exist models\dummy.joblib %PY_EXE% -m api.create_dummy || exit /b 1
)
start "NIDS API" /min %PY_EXE% -m uvicorn api.main:app --host 127.0.0.1 --port 8000
%PY_EXE% -m streamlit run app\dashboard.py
