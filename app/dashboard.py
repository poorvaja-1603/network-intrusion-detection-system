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

live_tab, metrics_tab = st.tabs(["Live alerts", "Model metrics"])


@st.fragment(run_every="10s")
def render_live_panel():
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        stats = requests.get(f"{API_URL}/stats", timeout=2).json()
        timeline = requests.get(f"{API_URL}/timeline", params={"minutes": 10}, timeout=2).json()["minutes"]
        recent = requests.get(f"{API_URL}/alerts", params={"limit": 10}, timeout=5).json().get("alerts", [])
    except requests.RequestException:
        st.warning("API is offline. Start it with run.bat to view live alerts.")
        return

    status_col, model_col, total_col, rate_col = st.columns(4)
    status_col.metric("Service", "Online")
    model_col.metric("Model", health.get("model_version", "—"))
    total_col.metric("Flows processed", stats.get("total_flows", 0))
    rate_col.metric("Alert rate", f"{stats.get('alert_rate', 0):.1%}")

    st.subheader("Predicted flows by class")
    counts = stats.get("counts_per_label", {})
    if any(counts.values()):
        st.bar_chart(pd.Series(counts, name="Flows"), height=260)
    else:
        st.info("No predictions have been received yet.")

    st.subheader("Recent attack alerts")
    st.caption("Alerts per minute · rolling 10-minute window · refreshes every 10 seconds")
    timeline_frame = pd.DataFrame(timeline)
    if not timeline_frame.empty:
        timeline_frame["minute"] = pd.to_datetime(timeline_frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        timeline_frame = timeline_frame.set_index("minute").rename(columns={"count": "Attack alerts"})
        timeline_frame.index.name = "Minute (local time)"
        st.bar_chart(timeline_frame[["Attack alerts"]], height=300, x_label="Minute", y_label="Attack alerts")
    else:
        st.info("No alert timeline is available yet.")

    if recent:
        alert_frame = pd.DataFrame(recent)
        alert_frame["timestamp"] = pd.to_datetime(alert_frame["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
        alert_frame["confidence"] = pd.to_numeric(alert_frame["confidence"], errors="coerce")
        st.dataframe(alert_frame[["timestamp", "label", "confidence"]], use_container_width=True, hide_index=True)
    else:
        st.info("No attack predictions yet. Start replay with: python -m app.replay --limit 300")


with live_tab:
    render_live_panel()

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
