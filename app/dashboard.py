"""Streamlit dashboard for live API alerts and saved experiment metrics."""
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NIDS Monitor", page_icon="🛡️", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
API_URL = st.sidebar.text_input("API address", "http://127.0.0.1:8000").rstrip("/")
st.title("Network Intrusion Detection System")
st.caption("NSL-KDD replay monitor — predictions are based on replayed benchmark flows, not live packet capture.")

try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    stats = requests.get(f"{API_URL}/stats", timeout=2).json()
    live = True
except requests.RequestException:
    health, stats, live = {}, {}, False

status_col, model_col, total_col, rate_col = st.columns(4)
status_col.metric("Service", "Online" if live else "Offline")
model_col.metric("Model", health.get("model_version", "—"))
total_col.metric("Flows processed", stats.get("total_flows", 0))
rate_col.metric("Alert rate", f"{stats.get('alert_rate', 0):.1%}")

live_tab, metrics_tab = st.tabs(["Live alerts", "Model metrics"])
with live_tab:
    st.subheader("Predicted flows by class")
    counts = stats.get("counts_per_label", {})
    if counts:
        st.bar_chart(pd.Series(counts, name="Flows"))
    else:
        st.info("No predictions have been received yet.")

    st.subheader("Recent attack alerts")
    if live:
        try:
            records = requests.get(f"{API_URL}/alerts", params={"limit": 100}, timeout=5).json().get("alerts", [])
            if records:
                alert_frame = pd.DataFrame(records)
                alert_frame["timestamp"] = pd.to_datetime(alert_frame["timestamp"], errors="coerce", utc=True)
                alert_frame["confidence"] = pd.to_numeric(alert_frame["confidence"], errors="coerce")
                by_time = alert_frame.dropna(subset=["timestamp"]).set_index("timestamp").resample("1min").size()
                st.caption("Alerts per minute")
                st.line_chart(by_time)
                st.dataframe(alert_frame[["timestamp", "label", "confidence"]], use_container_width=True, hide_index=True)
            else:
                st.info("No attack predictions yet. Start replay with: python -m app.replay --limit 300")
        except requests.RequestException as exc:
            st.error(f"Could not read alerts: {exc}")
    else:
        st.warning("Start the API with run.bat to view live alerts.")

with metrics_tab:
    st.subheader("Saved experiment results")
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            for result in value if isinstance(value, list) else [value]:
                if isinstance(result, dict) and "accuracy" in result:
                    rows.append({"file": path.name, **{key: result.get(key) for key in (
                        "model", "setting", "n_dims", "seed", "accuracy", "fpr", "precision",
                        "recall", "f1", "macro_f1", "fit_seconds",
                    )}})
        except (OSError, json.JSONDecodeError):
            continue
    if rows:
        metrics = pd.DataFrame(rows)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        chosen = st.selectbox("Compare metric", ["accuracy", "fpr", "macro_f1", "fit_seconds"])
        chart_data = metrics.dropna(subset=[chosen]).copy()
        chart_data["run"] = chart_data["file"] + " · " + chart_data["model"].fillna("model")
        st.bar_chart(chart_data.set_index("run")[chosen])
    else:
        st.info("No experiment result JSON files found under results/.")

if st.sidebar.button("Refresh data"):
    st.rerun()
st.sidebar.caption("Alert history is in memory and resets when the API restarts.")

