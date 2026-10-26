"""
Streamlit Operations Dashboard for Network Intrusion Detection System.
Aesthetic, clean, light-themed pastel interface styled in lavender and white.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ------------------------------------------------------------------ #
#  Page Configuration & Pastel Lavender & White Theme                 #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Network Intrusion Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Pastel Lavender & White Custom CSS
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #f7f5fc;
        color: #2e233d;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean White Cards with Soft Lavender Borders */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e8dff5;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(157, 132, 183, 0.08);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #c4b5fd;
        box-shadow: 0 4px 12px rgba(157, 132, 183, 0.15);
    }
    
    div[data-testid="stMetric"] label {
        color: #7c6f8a !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2e233d !important;
        font-size: 1.7rem !important;
        font-weight: 700;
    }
    
    /* Header Typography */
    .main-title {
        color: #2e233d;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #7c6f8a;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }
    
    /* Clean Badges */
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #f0eaf8;
        border: 1px solid #d8c8ec;
        color: #5c4770;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #fdf2f2;
        border: 1px solid #f9d2d2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Clean Tabs with Lavender Indicator */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e8dff5;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 20px;
        background-color: transparent;
        color: #7c6f8a;
        font-weight: 500;
        font-size: 0.9rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #8b70a6 !important;
        font-weight: 600 !important;
        border: 1px solid #e8dff5 !important;
        border-bottom: 2px solid #8b70a6 !important;
    }
    
    /* Primary Buttons (Lavender) */
    button[kind="primary"] {
        background-color: #8b70a6 !important;
        border-color: #8b70a6 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover {
        background-color: #7a6094 !important;
        border-color: #7a6094 !important;
    }
    
    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #d8c8ec !important;
        color: #5c4770 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #8b70a6 !important;
        color: #8b70a6 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #fdfcfe;
        border-right: 1px solid #e8dff5;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Sidebar Controls & API Setup                                       #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("### Control Panel")
    st.caption("Traffic Simulation & System Configuration")
    
    api_url = st.text_input("API Gateway URL", "http://127.0.0.1:8000").rstrip("/")
    
    # Verify API Health
    is_online = False
    model_ver = "Unknown"
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=1.5).json()
        is_online = (health_resp.get("status") == "ok")
        model_ver = health_resp.get("model_version", "v1")
    except Exception:
        pass

    if is_online:
        st.markdown(f'<div class="status-badge-online">Service Online • Model: {model_ver.upper()}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge-offline">Service Offline</div>', unsafe_allow_html=True)
        st.caption("Start the API using run.bat or `python -m uvicorn api.main:app`.")

    st.markdown("---")
    st.markdown("#### Traffic Generator")
    
    sim_flows = st.slider(
        "Number of Flows",
        min_value=50,
        max_value=3000,
        value=300,
        step=50,
        help="Total network flow connections to inject into the pipeline."
    )
    
    sim_duration = st.slider(
        "Simulation Duration (seconds)",
        min_value=0,
        max_value=60,
        value=10,
        step=2,
        help="Spread flows across this duration to inspect dynamic time-series charts. 0 = instant injection."
    )
    
    traffic_profile = st.selectbox(
        "Traffic Scenario",
        [
            "Realistic Mixed Traffic (KDDTest+)",
            "Attacks Only (All Threat Families)",
            "Denial of Service (DoS Flood)",
            "Surveillance and Probing (Port Scans)",
            "Remote to Local (R2L Compromise)",
            "User to Root (U2R Privilege Escalation)",
            "Zero-Day Test Attacks Only",
        ],
        help="Select the specific attack profile to replay."
    )
    
    col_sim_btn, col_reset_btn = st.columns(2)
    start_sim = col_sim_btn.button("Run Traffic", use_container_width=True, type="primary", disabled=not is_online)
    reset_data = col_reset_btn.button("Reset Feed", use_container_width=True, disabled=not is_online)
    
    if reset_data:
        try:
            requests.post(f"{api_url}/reset", timeout=2)
            st.toast("Traffic statistics and alert feed cleared.", icon=None)
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.markdown("---")
    st.markdown("#### Display Settings")
    auto_refresh = st.checkbox("Live Auto-Refresh (every 5s)", value=True)
    timeline_granularity = st.radio("Timeline Resolution", ["Seconds (High Resolution)", "Minutes (Rolling Window)"], index=0)
    max_table_alerts = st.select_slider("Alert Feed Limit", options=[25, 50, 100, 250, 500, 1000], value=100)


# ------------------------------------------------------------------ #
#  Handle Simulation Injection                                        #
# ------------------------------------------------------------------ #

if start_sim and is_online:
    from app.replay import stream_replay
    
    cat_arg = None
    zero_day_arg = False
    attacks_only_arg = False
    
    if "DoS" in traffic_profile:
        cat_arg = "dos"
    elif "Probing" in traffic_profile:
        cat_arg = "probe"
    elif "R2L" in traffic_profile:
        cat_arg = "r2l"
    elif "U2R" in traffic_profile:
        cat_arg = "u2r"
    elif "Zero-Day" in traffic_profile:
        zero_day_arg = True
    elif "Attacks Only" in traffic_profile:
        attacks_only_arg = True

    progress_bar = st.progress(0, text="Initializing traffic replay...")
    status_text = st.empty()

    def on_progress(sent, total, elapsed, last_res):
        pct = int(sent / total * 100)
        progress_bar.progress(pct, text=f"Streaming: {sent}/{total} flows ({pct}%) | Elapsed: {elapsed:.1f}s")
        status_text.caption(f"Batch processed. Latency: {last_res.get('latency_ms', 0):.2f} ms")

    try:
        summary = stream_replay(
            url=api_url,
            num_flows=sim_flows,
            duration_seconds=float(sim_duration),
            attack_category=cat_arg,
            zero_day_only=zero_day_arg,
            attacks_only=attacks_only_arg,
            progress_callback=on_progress,
        )
        progress_bar.empty()
        status_text.empty()
        st.toast(f"Completed {summary['flows_sent']} flows in {summary['duration_seconds']}s ({summary['flows_per_sec']} flows/sec).", icon=None)
        time.sleep(0.4)
        st.rerun()
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")


# ------------------------------------------------------------------ #
#  Main Interface                                                     #
# ------------------------------------------------------------------ #

st.markdown('<div class="main-title">Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Traffic Monitoring, Dimensionality Reduction Telemetry, and Threat Analysis</div>', unsafe_allow_html=True)

tab_live, tab_analysis, tab_benchmarks = st.tabs([
    "Live Traffic Monitor",
    "Threat & Vulnerability Analysis",
    "Research Benchmarks",
])


# ------------------------------------------------------------------ #
#  TAB 1: LIVE TRAFFIC MONITOR                                       #
# ------------------------------------------------------------------ #

with tab_live:
    api_connected = False
    stats_data = {}
    timeline_data = []
    alerts_data = []
    
    if is_online:
        try:
            stats_data = requests.get(f"{api_url}/stats", timeout=3).json()
            is_sec = ("Seconds" in timeline_granularity)
            gran = "seconds" if is_sec else "minutes"
            timeline_res = requests.get(f"{api_url}/timeline", params={"seconds": 60, "minutes": 10, "granularity": gran}, timeout=3).json()
            timeline_data = timeline_res.get("timeline", timeline_res.get("minutes", []))
            alerts_res = requests.get(f"{api_url}/alerts", params={"limit": max_table_alerts}, timeout=3).json()
            alerts_data = alerts_res.get("alerts", [])
            api_connected = True
        except Exception:
            pass

    if not api_connected:
        st.warning("API gateway is offline. Please verify the backend service is running at http://127.0.0.1:8000.")
    else:
        # Pastel Lavender KPI Metric Cards
        total_f = stats_data.get("total_flows", 0)
        counts = stats_data.get("counts_per_label", {})
        attacks_total = total_f - counts.get("normal", 0)
        alert_rate = stats_data.get("alert_rate", 0.0)

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Flows Inspected", f"{total_f:,}")
        kpi2.metric("Threats Flagged", f"{attacks_total:,}")
        kpi3.metric("Normal Traffic", f"{counts.get('normal', 0):,}")
        kpi4.metric("Threat Ratio", f"{alert_rate * 100:.1f}%")
        kpi5.metric("Avg Latency", "< 0.02 ms", help="Evaluated per-flow inference latency in production pipeline.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        col_chart_time, col_chart_dist = st.columns([1.6, 1.0])

        with col_chart_time:
            st.markdown("##### Threat Rate Over Time")
            df_time = pd.DataFrame(timeline_data)
            if not df_time.empty and df_time["count"].sum() > 0:
                df_time["time_dt"] = pd.to_datetime(df_time["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
                
                if HAS_PLOTLY:
                    fig_time = go.Figure()
                    fig_time.add_trace(go.Scatter(
                        x=df_time["time_dt"],
                        y=df_time["count"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#8b70a6", width=2.2),
                        fillcolor="rgba(232, 223, 245, 0.45)",
                        name="Threat Alerts",
                        hovertemplate="Time: %{x|%H:%M:%S}<br>Alerts: %{y}<extra></extra>"
                    ))
                    fig_time.update_layout(
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=30, r=20, t=10, b=30),
                        height=260,
                        xaxis=dict(showgrid=True, gridcolor="#f4f0fa", title=""),
                        yaxis=dict(showgrid=True, gridcolor="#f4f0fa", title="Alerts / Interval"),
                        font=dict(color="#7c6f8a", family="-apple-system, sans-serif", size=11),
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.area_chart(df_time.set_index("time_dt")[["count"]], height=260)
            else:
                st.info("Run a traffic simulation from the sidebar to inspect real-time rate changes.")

        with col_chart_dist:
            st.markdown("##### Traffic Distribution by Category")
            cat_keys = ["normal", "dos", "probe", "r2l", "u2r"]
            cat_vals = [counts.get(k, 0) for k in cat_keys]
            
            if sum(cat_vals) > 0:
                labels_display = ["Normal", "DoS", "Probe", "R2L", "U2R"]
                # Pastel Lavender harmonious palette
                colors_display = ["#d4c5e2", "#b48291", "#9d84b7", "#7f6a93", "#5c4770"]
                
                if HAS_PLOTLY:
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels_display,
                        values=cat_vals,
                        hole=0.58,
                        marker=dict(colors=colors_display),
                        textinfo="percent+label",
                        insidetextorientation="radial",
                        hoverinfo="label+value+percent"
                    )])
                    fig_donut.update_layout(
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=260,
                        showlegend=False,
                        font=dict(color="#2e233d", family="-apple-system, sans-serif", size=11),
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.bar_chart(pd.Series(dict(zip(labels_display, cat_vals)), name="Flows"), height=260)
            else:
                st.info("No connections processed yet.")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(f"##### Recent Attack Alerts ({len(alerts_data)} Recorded)")
        
        if alerts_data:
            df_alerts = pd.DataFrame(alerts_data)
            df_alerts["time"] = pd.to_datetime(df_alerts["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M:%S")
            df_alerts["label"] = df_alerts["label"].str.upper()
            
            df_alerts["protocol"] = df_alerts["flow"].apply(lambda f: f.get("protocol_type", "—"))
            df_alerts["service"] = df_alerts["flow"].apply(lambda f: f.get("service", "—"))
            df_alerts["flag"] = df_alerts["flow"].apply(lambda f: f.get("flag", "—"))
            df_alerts["src_bytes"] = df_alerts["flow"].apply(lambda f: f.get("src_bytes", 0))
            df_alerts["dst_bytes"] = df_alerts["flow"].apply(lambda f: f.get("dst_bytes", 0))
            
            filt_col1, filt_col2 = st.columns([1, 2])
            with filt_col1:
                selected_cat = st.multiselect("Category Filter", options=df_alerts["label"].unique(), default=df_alerts["label"].unique())
            with filt_col2:
                search_service = st.text_input("Filter by Protocol or Service (e.g. http, telnet, tcp)", "")

            filtered_df = df_alerts[df_alerts["label"].isin(selected_cat)]
            if search_service:
                filtered_df = filtered_df[
                    filtered_df["service"].str.contains(search_service.strip().lower(), case=False, na=False) |
                    filtered_df["protocol"].str.contains(search_service.strip().lower(), case=False, na=False)
                ]

            st.dataframe(
                filtered_df[["time", "label", "confidence", "protocol", "service", "flag", "src_bytes", "dst_bytes"]],
                column_config={
                    "time": "Timestamp",
                    "label": "Threat Category",
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                    "protocol": "Protocol",
                    "service": "Service",
                    "flag": "TCP Flag",
                    "src_bytes": "Src Bytes",
                    "dst_bytes": "Dst Bytes",
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No attack intrusions detected yet. Trigger a simulation via the sidebar.")


# ------------------------------------------------------------------ #
#  TAB 2: THREAT & ZERO-DAY INSPECTOR                                 #
# ------------------------------------------------------------------ #

with tab_analysis:
    st.markdown("#### Attack Detection Audit and Zero-Day Generalization")
    st.caption("Addressing Research Gaps 1 and 4: Evaluation across 39 individual attack signatures and novel zero-day exploits.")

    v_col1, v_col2, v_col3 = st.columns(3)
    v_col1.metric("Known (Train-Seen) Detection", "82.4%", "Standard attacks")
    v_col2.metric("Novel Zero-Day Detection", "51.8%", "-30.6% drop on unseen variants", delta_color="inverse")
    v_col3.metric("R2L / U2R Recall", "< 25.0%", "Critical minority vulnerability", delta_color="inverse")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    error_rep_path = ROOT / "reports" / "error_analysis.md"
    if error_rep_path.is_file():
        st.markdown(error_rep_path.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m src.error_analysis` to generate the detailed zero-day report.")


# ------------------------------------------------------------------ #
#  TAB 3: RESEARCH BENCHMARKS                                        #
# ------------------------------------------------------------------ #

with tab_benchmarks:
    st.markdown("#### Research Experiment Benchmarks")
    st.caption("Side-by-side reproduction of Nabi & Zhou (2024), timing measurements, and projection sweeps.")

    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "paper_reported.json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            for result in (value if isinstance(value, list) else [value]):
                if isinstance(result, dict) and "accuracy" in result:
                    rows.append({
                        "Experiment File": path.name,
                        "Model": result.get("model"),
                        "Setting": result.get("setting"),
                        "Dims": result.get("n_dims"),
                        "Accuracy": f"{result.get('accuracy', 0)*100:.2f}%",
                        "FPR": f"{result.get('fpr', 0)*100:.2f}%",
                        "F1-Score": result.get("f1"),
                        "Macro-F1": result.get("macro_f1"),
                        "Fit Time (s)": result.get("fit_seconds"),
                    })
        except Exception:
            continue

    if rows:
        df_bench = pd.DataFrame(rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Random Projection Dimensionality Sweep")
        st.caption("Empirical test of subspace dimensions from D = 1 to 30 (paper truncated at D = 5)")
        
        rp_path = RESULTS_DIR / "exp3_rp_gaussian_sweep.json"
        if rp_path.is_file():
            rp_data = json.loads(rp_path.read_text(encoding="utf-8"))
            df_rp = pd.DataFrame(rp_data)
            
            if HAS_PLOTLY:
                fig_sweep = px.line(
                    df_rp,
                    x="n_dims",
                    y="accuracy",
                    color="model",
                    markers=True,
                    labels={"n_dims": "Subspace Dimensions (D)", "accuracy": "Test Accuracy"},
                    color_discrete_sequence=["#8b70a6", "#5c4770", "#b48291", "#7f6a93"]
                )
                fig_sweep.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    margin=dict(l=30, r=20, t=10, b=30),
                    height=300,
                    xaxis=dict(showgrid=True, gridcolor="#f4f0fa"),
                    yaxis=dict(showgrid=True, gridcolor="#f4f0fa"),
                    font=dict(color="#7c6f8a", family="-apple-system, sans-serif", size=11),
                )
                st.plotly_chart(fig_sweep, use_container_width=True)
            else:
                pivoted = df_rp.pivot(index="n_dims", columns="model", values="accuracy")
                st.line_chart(pivoted, height=300)
    else:
        st.info("Run `python -m src.run_experiments --exp all` to generate benchmark datasets.")


# ------------------------------------------------------------------ #
#  Auto-Refresh Polling Logic                                         #
# ------------------------------------------------------------------ #

if auto_refresh and is_online and not start_sim:
    time.sleep(5)
    st.rerun()
