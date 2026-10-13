import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.set_page_config(
    page_title="SentinelX AI | SOC Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- THEME / LAYOUT --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 82% 0%, rgba(45,212,191,.08), transparent 35%),
      linear-gradient(135deg, #171C24 0%, #191F28 55%, #141920 100%);
    color: #E2E8F0;
}
[data-testid="stHeader"] { background: rgba(23,28,36,.94); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #202833 0%, #1B222C 100%);
    border-right: 1px solid #35414E;
    min-width: 250px;
}
/* Compact, balanced sidebar navigation. */
[data-testid="stSidebar"] > div { padding-top: 1rem; padding-bottom: 1rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .6rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { margin-bottom: .2rem; font-size: 1.45rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .7rem !important; padding-top: .55rem; padding-bottom: .75rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    background: rgba(38,47,59,.82); border: 1px solid rgba(65,78,92,.72);
    border-radius: 11px; padding: .72rem .8rem; min-height: 3.15rem;
    font-size: .98rem; font-weight: 600;
    transition: background .15s ease, border-color .15s ease, transform .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: rgba(45,212,191,.12); border-color: #2DD4BF; transform: translateX(2px); }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { margin-right: .65rem; }
.block-container { padding-top: 1.15rem; padding-bottom: 1.5rem; max-width: 1800px; }
h1, h2, h3, h4 { color: #F1F5F9 !important; letter-spacing: -.025em; }
p, label, span { color: inherit; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #262F3B, #222A35);
    border: 1px solid #3A4654; border-radius: 13px;
    padding: 15px 17px; min-height: 132px;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] { color: #C7D2DE !important; font-size: .92rem; }
[data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: .88rem; }
div[data-testid="stDataFrame"] {
    border: 1px solid #3A4654; border-radius: 10px; overflow: hidden;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #3A4654 !important; border-radius: 13px !important;
    background: rgba(38,47,59,.72);
}
.stButton > button {
    border-radius: 8px; border: 1px solid #258F86;
    background: linear-gradient(90deg, #16877E, #2DD4BF);
    color: white; font-weight: 600;
}
.stButton > button:hover { border-color: #7CE8D7; color: #FFFFFF; }
div[data-testid="stAlert"] { border-radius: 9px; }
.small-muted { color: #9AA8B8; font-size: .82rem; }
.brand-title { font-size: 1.4rem; font-weight: 800; color: #F8FAFC; }
.brand-subtitle { color: #A6B4C3; font-size: .75rem; margin-top: -4px; }
.section-label { color: #D7E0E8; font-weight: 650; font-size: 1.02rem; }
.status-pill {
 display:inline-block; border:1px solid #8a6827; color:#ffd783;
 background:rgba(138,104,39,.18); border-radius:22px; padding:6px 12px;
 font-size:.78rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- DEMO DATA --------------------------
# Clearly labelled demo values; replace with actual team outputs during integration.
alerts = pd.DataFrame([
    {"ID":"ALT-001","Type":"Suspicious Login","Risk":"High","Status":"New","Time":"10:32 AM","Source":"192.168.1.23"},
    {"ID":"ALT-002","Type":"Port Scanning","Risk":"Medium","Status":"Investigating","Time":"09:47 AM","Source":"10.0.2.15"},
    {"ID":"ALT-003","Type":"Data Transfer","Risk":"High","Status":"New","Time":"08:21 AM","Source":"10.0.4.21"},
    {"ID":"ALT-004","Type":"Failed Logins","Risk":"Low","Status":"Resolved","Time":"07:12 AM","Source":"10.0.5.44"},
    {"ID":"ALT-005","Type":"Malware Activity","Risk":"Medium","Status":"Investigating","Time":"06:45 AM","Source":"172.16.0.8"},
])
severity = pd.DataFrame({"Severity":["High","Medium","Low"],"Count":[18,41,69]})
trend = pd.DataFrame([
    {"Time":"12AM","High":16,"Medium":12,"Low":7},
    {"Time":"2AM","High":14,"Medium":11,"Low":8},
    {"Time":"4AM","High":17,"Medium":14,"Low":9},
    {"Time":"6AM","High":15,"Medium":13,"Low":8},
    {"Time":"8AM","High":19,"Medium":16,"Low":11},
    {"Time":"10AM","High":23,"Medium":18,"Low":12},
    {"Time":"12PM","High":28,"Medium":22,"Low":15},
    {"Time":"2PM","High":22,"Medium":17,"Low":13},
    {"Time":"4PM","High":21,"Medium":16,"Low":12},
    {"Time":"6PM","High":19,"Medium":15,"Low":11},
    {"Time":"8PM","High":25,"Medium":20,"Low":16},
    {"Time":"10PM","High":27,"Medium":22,"Low":18},
    {"Time":"12AM","High":32,"Medium":26,"Low":21},
])
threats = pd.DataFrame({
    "Attack Type":["Suspicious Login","Port Scanning","Data Transfer","Failed Logins","Malware Activity","Other"],
    "Share":[28,22,18,12,9,11]
})
recent_alerts = pd.DataFrame([
    {"Severity":"High","Alert":"Suspicious login attempt from 192.168.1.23","Time":"10:32 AM"},
    {"Severity":"Medium","Alert":"Unusual data transfer (3.4 GB)","Time":"09:47 AM"},
    {"Severity":"High","Alert":"Multiple failed logins (5 attempts)","Time":"08:21 AM"},
    {"Severity":"Low","Alert":"Port scan detected from 10.0.2.15","Time":"07:12 AM"},
    {"Severity":"Medium","Alert":"Malware signature detected","Time":"06:45 AM"},
    {"Severity":"Low","Alert":"Policy violation (blocked)","Time":"05:32 AM"},
])

# -------------------------- SIDEBAR --------------------------
with st.sidebar:
    st.markdown("## 🛡️ SentinelX AI")
    st.markdown('<div class="brand-subtitle">SOC Operations Center</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;letter-spacing:.12em;font-weight:700;color:#7399c9;padding:0 3px">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATION",
        ["📊  Overview", "🔎  Incident Investigation", "🌐  Threat Intelligence", "🛡️  Response Center", "📄  Reports", "⚙️  Settings"],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[1] if "  " in page else page
    st.markdown("---")
    st.markdown('<div class="status-pill">● Demo Environment</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="padding:5px 3px 0">Demo Mode · No live actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="padding:7px 3px 0;font-size:.72rem">SENTINELX AI · SOC DASHBOARD DEMO ENVIRONMENT</div>', unsafe_allow_html=True)

# -------------------------- TOP BAR --------------------------
top_left, top_right = st.columns([4, 1])
with top_left:
    search = st.text_input(" ", placeholder="⌕  Search alerts, IPs, or keywords...", label_visibility="collapsed")
with top_right:
    st.markdown("<div style='text-align:right;padding-top:5px;color:#C7D2DE;font-weight:600'>🔔  Harsha ▾</div>", unsafe_allow_html=True)

head_left, head_right = st.columns([4, 1])
with head_left:
    st.title("Security Operations Center" if page == "Overview" else page)
    st.caption("AI-powered threat detection, investigation and response")
with head_right:
    st.markdown("<div style='text-align:right;color:#A6B4C3;font-size:.8rem'>PROTOTYPE ENVIRONMENT</div><div style='text-align:right;padding-top:8px'><span class='status-pill'>● Prototype Mode</span></div>", unsafe_allow_html=True)

if search.strip():
    filtered = alerts[
        alerts.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
    ]
else:
    filtered = alerts

# -------------------------- OVERVIEW PAGE --------------------------
if page == "Overview":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🛡️ Total Alerts", "128", "↑ 12% vs. previous 24h", delta_color="inverse")
    m2.metric("🚨 High Risk", "18", "↑ 5% vs. previous 24h", delta_color="inverse")
    m3.metric("🟠 Under Investigation", "32", "→ 0% vs. previous 24h", delta_color="off")
    m4.metric("✅ Resolved", "78", "↑ 8% vs. previous 24h", delta_color="normal")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # First analytics row: donut, incidents, trend.
    chart_col, incident_col, trend_col = st.columns([1.02, 1.45, 1.25], gap="small")

    with chart_col:
        with st.container(border=True):
            st.markdown("**♟ Alert Severity Distribution**")
            pie = alt.Chart(severity).mark_arc(innerRadius=54, outerRadius=83, stroke="#262F3B", strokeWidth=2).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Severity:N", scale=alt.Scale(domain=["High","Medium","Low"], range=["#ff3e63","#ffad35","#20d4a0"]), legend=None),
                tooltip=["Severity:N","Count:Q"],
            )
            st.altair_chart(pie, use_container_width=True)
            st.caption("🔴 High 14% (18)   🟠 Medium 32% (41)   🟢 Low 54% (69)")

    with incident_col:
        with st.container(border=True):
            st.markdown("**◈ Recent Incidents**")
            st.dataframe(
                filtered[["ID","Type","Risk","Status","Time"]],
                use_container_width=True, hide_index=True, height=250
            )
            st.caption("Demo incident records • select Incident Investigation for details")

    with trend_col:
        with st.container(border=True):
            st.markdown("**Alerts Trend (Last 24 Hours)**")
            trend_long = trend.melt("Time", var_name="Severity", value_name="Count")
            line = alt.Chart(trend_long).mark_line(interpolate="monotone", strokeWidth=2.3).encode(
                x=alt.X("Time:N", sort=None, axis=alt.Axis(labelColor="#A6B4C3", title=None, labelAngle=0, grid=False)),
                y=alt.Y("Count:Q", axis=alt.Axis(labelColor="#A6B4C3", title=None, gridColor="#3A4654")),
                color=alt.Color("Severity:N", scale=alt.Scale(domain=["High","Medium","Low"], range=["#ff3e63","#ffad35","#20d4a0"]), legend=alt.Legend(orient="top", labelColor="#D2DCE6", title=None)),
                tooltip=["Time:N","Severity:N","Count:Q"],
            ).properties(height=220)
            st.altair_chart(line, use_container_width=True)

    # Second analytics row: threat types, alert feed, status/assistant.
    threat_col, feed_col, system_col = st.columns([1.2, 1.3, 1], gap="small")
    with threat_col:
        with st.container(border=True):
            st.markdown("**Top Threats / Attack Types**")
            threat_chart = alt.Chart(threats).mark_bar(cornerRadiusEnd=5).encode(
                y=alt.Y("Attack Type:N", sort="-x", axis=alt.Axis(labelColor="#D2DCE6", title=None, labelLimit=140)),
                x=alt.X("Share:Q", axis=alt.Axis(labelColor="#A6B4C3", title=None, gridColor="#3A4654", format=".0f")),
                color=alt.Color("Share:Q", scale=alt.Scale(range=["#2DD4BF","#63C8C0","#F4C66A","#F59E67","#FB7185"]), legend=None),
                tooltip=["Attack Type:N","Share:Q"],
            ).properties(height=240)
            st.altair_chart(threat_chart, use_container_width=True)
    with feed_col:
        with st.container(border=True):
            st.markdown("**Recent Alerts**")
            st.dataframe(recent_alerts, use_container_width=True, hide_index=True, height=275)
    with system_col:
        with st.container(border=True):
            st.markdown("**⚙️ System Status**")
            for service in ["Data Pipeline","Model Service","Database","Alert Engine","Notification Service"]:
                st.markdown(f"🟢 **{service}** — Online")
            st.markdown("---")
            st.markdown("**✨ AI Assistant**")
            st.caption("Ask about alerts, threats and AI-powered insights.")
            if st.button("Open Assistant →", use_container_width=True):
                st.session_state["assistant_open"] = True
            if st.session_state.get("assistant_open"):
                st.info("AI assistant integration is pending. No generated investigation or model output is shown in this demo.")

    st.caption("⚠️ Prototype UI reference: all metrics, trends, rankings, and status indicators are illustrative demo data until connected to verified backend services.")

elif page == "Incident Investigation":
    st.subheader("Investigate an alert")
    chosen = st.selectbox("Alert ID", alerts["ID"])
    row = alerts[alerts["ID"] == chosen].iloc[0]
    left, right = st.columns(2)
    with left:
        st.markdown("### Incident Details")
        st.write("**Type:**", row["Type"])
        st.write("**Severity:**", row["Risk"])
        st.write("**Status:**", row["Status"])
        st.write("**Source:**", row["Source"])
        st.write("**Time:**", row["Time"])
    with right:
        st.markdown("### AI Investigation")
        st.info("Backend pending: connect the team's model prediction, SHAP explanation, correlated events, and LLM summary here.")
        st.markdown("**Verified evidence:** Not loaded")
        st.markdown("**Model prediction:** Not available yet")

elif page == "Response Center":
    st.subheader("Analyst-controlled response")
    chosen = st.selectbox("Select an alert", alerts["ID"], key="response_select")
    st.warning("Simulation only. No real security action will execute.")
    note = st.text_input("Analyst note (optional)")
    a, b = st.columns(2)
    with a:
        if st.button("Approve simulated response", use_container_width=True):
            st.success(f"Demo approval recorded for {chosen}.")
            st.caption(note if note else "No note provided.")
    with b:
        if st.button("Reject simulated response", use_container_width=True):
            st.error(f"Demo rejection recorded for {chosen}.")
            st.caption(note if note else "No note provided.")

elif page == "Threat Intelligence":
    st.subheader("Threat Intelligence")
    st.info("Threat feeds and enrichment results will appear here after backend integration.")
    st.dataframe(threats, use_container_width=True, hide_index=True)

elif page == "Reports":
    st.subheader("Reports")
    st.info("Exportable incident reports will be connected after backend integration.")
    st.dataframe(alerts, use_container_width=True, hide_index=True)

else:
    st.subheader("Settings")
    st.info("Prototype settings page. Live service configuration is not connected.")
    st.toggle("Demo mode", value=True, disabled=True)
    st.toggle("Allow automated response", value=False, disabled=True)

