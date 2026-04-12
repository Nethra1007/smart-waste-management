"""
Smart Waste Collection Data Analytics Dashboard
Trichy, Tamil Nadu — Real-Time Simulation
"""

import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Analytics — Trichy",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ZONES = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E", "Zone F"]
ZONE_LABELS = {
    "Zone A": "Srirangam",
    "Zone B": "Ariyamangalam",
    "Zone C": "Aviyur",
    "Zone D": "Golden Rock",
    "Zone E": "Palakkarai",
    "Zone F": "Thiruverumbur",
}
WASTE_TYPES = ["organic", "recyclable", "general", "hazardous"]
WASTE_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

# Approximate lat/lon for Trichy zones
ZONE_COORDS = {
    "Zone A": (10.8650, 78.6930),   # Srirangam
    "Zone B": (10.7950, 78.7350),   # Ariyamangalam
    "Zone C": (10.8200, 78.6600),   # Aviyur
    "Zone D": (10.8400, 78.7200),   # Golden Rock
    "Zone E": (10.8050, 78.6850),   # Palakkarai
    "Zone F": (10.7750, 78.7100),   # Thiruverumbur
}

ALERT_PHONE = "+91-9876543210"

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
if "pending_cycles" not in st.session_state:
    st.session_state.pending_cycles = {z: 0 for z in ZONES}
if "zone_history" not in st.session_state:
    st.session_state.zone_history = {z: [] for z in ZONES}
if "last_whatsapp" not in st.session_state:
    st.session_state.last_whatsapp = None
if "overflow_log" not in st.session_state:
    st.session_state.overflow_log = []

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
def generate_live_data(festival_mode: bool = False) -> pd.DataFrame:
    """Generate one row per zone with realistic simulated sensor readings."""
    now = datetime.now()
    hour = now.hour

    # Time-of-day pattern: more waste in morning (7-10) and evening (17-20)
    if 7 <= hour <= 10:
        base_multiplier = 1.3
    elif 17 <= hour <= 20:
        base_multiplier = 1.2
    elif 0 <= hour <= 5:
        base_multiplier = 0.6
    else:
        base_multiplier = 1.0

    if festival_mode:
        base_multiplier *= 1.5

    rows = []
    for zone in ZONES:
        waste_kg = round(random.uniform(300, 1200) * base_multiplier, 1)
        waste_kg = min(waste_kg, 1400)  # cap

        if waste_kg > 1000:
            status = "overflow"
        elif random.random() < 0.15:
            status = "pending"
        else:
            status = "collected"

        rows.append({
            "zone": zone,
            "area": ZONE_LABELS[zone],
            "timestamp": now,
            "waste_kg": waste_kg,
            "waste_type": random.choices(WASTE_TYPES, weights=WASTE_WEIGHTS)[0],
            "trucks_deployed": random.randint(2, 5),
            "collection_status": status,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PROBLEM SCORE
# ─────────────────────────────────────────────
def calculate_problem_scores(history: pd.DataFrame) -> dict:
    """Calculate problem score per zone as % of problematic records."""
    scores = {}
    if history.empty:
        return {z: 0 for z in ZONES}
    for zone in ZONES:
        zdf = history[history["zone"] == zone]
        if len(zdf) == 0:
            scores[zone] = 0
            continue
        total = len(zdf)
        overflow_count = (zdf["collection_status"] == "overflow").sum()
        pending_count  = (zdf["collection_status"] == "pending").sum()
        score = (overflow_count * 2 + pending_count) / total * 100
        scores[zone] = round(min(score, 100), 1)
    return scores


# ─────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────
def build_map(problem_scores: dict) -> folium.Map:
    """Build a Folium map of Trichy with colour-coded zone markers."""
    m = folium.Map(location=[10.8200, 78.6900], zoom_start=12, tiles="CartoDB positron")

    for zone, (lat, lon) in ZONE_COORDS.items():
        score = problem_scores.get(zone, 0)

        if score >= 50:
            color = "red"
        elif score >= 20:
            color = "orange"
        else:
            color = "green"

        radius = 10 + score * 0.3  # bigger circle = bigger problem

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{zone} — {ZONE_LABELS[zone]}</b><br>Problem Score: {score:.1f}%",
                max_width=200
            ),
            tooltip=f"{zone}: {score:.1f}%"
        ).add_to(m)

    return m


# ─────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────
def predict_next_waste(zone_history: list) -> float | None:
    """Predict next waste_kg for a zone using linear regression on last 10 values."""
    if len(zone_history) < 5:
        return None
    values = zone_history[-10:]
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)
    next_x = np.array([[len(values)]])
    pred = model.predict(next_x)[0]
    return round(max(pred, 0), 1)


# ─────────────────────────────────────────────
# WHATSAPP SIMULATION
# ─────────────────────────────────────────────
def simulate_whatsapp_alert(zone: str):
    """Simulate sending a WhatsApp alert for overflow."""
    msg = f"[SIMULATED WHATSAPP] Alert sent to {ALERT_PHONE}: 🚨 Overflow in {zone} ({ZONE_LABELS[zone]})! Extra truck needed immediately."
    print(msg)
    st.session_state.last_whatsapp = {
        "message": f"🚨 Overflow in {zone} ({ZONE_LABELS[zone]})! Extra truck needed.",
        "time": datetime.now().strftime("%H:%M:%S"),
        "phone": ALERT_PHONE
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/waste-sorting.png", width=60)
    st.title("🗑️ Smart Waste Analytics")
    st.caption("Trichy, Tamil Nadu — Live Dashboard")
    st.divider()

    refresh_interval = st.slider("⏱️ Refresh every (seconds)", 2, 10, 5)

    st.divider()
    selected_zones = st.multiselect("📍 Filter Zones", ZONES, default=ZONES)
    selected_types = st.multiselect("♻️ Filter Waste Type", WASTE_TYPES, default=WASTE_TYPES)

    st.divider()
    festival_mode = st.checkbox("🎉 Festival Mode (Diwali / Pongal)", value=False)

    st.divider()
    st.subheader("📱 Last WhatsApp Alert")
    if st.session_state.last_whatsapp:
        wa = st.session_state.last_whatsapp
        st.error(f"**{wa['message']}**")
        st.caption(f"Sent to {wa['phone']} at {wa['time']}")
    else:
        st.info("No alerts yet.")

    st.divider()
    st.caption("Data refreshes automatically. All data is simulated.")


# ─────────────────────────────────────────────
# FESTIVAL BANNER
# ─────────────────────────────────────────────
if festival_mode:
    st.warning("🎉 **Festival Mode ON** — Waste volume increased by 50% (Diwali / Pongal simulation)")

# ─────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────
st.title("🗑️ Smart Waste Collection — Trichy Live Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}  |  Auto-refreshing every {refresh_interval}s")

# ─────────────────────────────────────────────
# GENERATE NEW DATA
# ─────────────────────────────────────────────
new_data = generate_live_data(festival_mode=festival_mode)

# Append to history (keep last 30 minutes of data)
st.session_state.history = pd.concat(
    [st.session_state.history, new_data], ignore_index=True
)
cutoff = datetime.now() - timedelta(minutes=30)
st.session_state.history = st.session_state.history[
    st.session_state.history["timestamp"] >= cutoff
]

# Update zone histories for ML
for zone in ZONES:
    zone_val = new_data[new_data["zone"] == zone]["waste_kg"].values
    if len(zone_val) > 0:
        st.session_state.zone_history[zone].append(float(zone_val[0]))
        st.session_state.zone_history[zone] = st.session_state.zone_history[zone][-20:]

# Update pending cycle tracker
for zone in ZONES:
    status = new_data[new_data["zone"] == zone]["collection_status"].values
    if len(status) > 0 and status[0] == "pending":
        st.session_state.pending_cycles[zone] += 1
    else:
        st.session_state.pending_cycles[zone] = 0

# Check for overflows → WhatsApp
for _, row in new_data.iterrows():
    if row["collection_status"] == "overflow":
        if row["zone"] not in [x.get("zone") for x in st.session_state.overflow_log[-3:]]:
            simulate_whatsapp_alert(row["zone"])
            st.session_state.overflow_log.append({"zone": row["zone"], "time": datetime.now()})

# Apply filters
filtered = new_data[
    (new_data["zone"].isin(selected_zones)) &
    (new_data["waste_type"].isin(selected_types))
]

history_filtered = st.session_state.history[
    (st.session_state.history["zone"].isin(selected_zones)) &
    (st.session_state.history["waste_type"].isin(selected_types))
]

# ─────────────────────────────────────────────
# KPI CARDS — ROW 1
# ─────────────────────────────────────────────
total_waste    = int(history_filtered["waste_kg"].sum())
total_trucks   = int(new_data["trucks_deployed"].sum())
pending_zones  = int((new_data["collection_status"] == "pending").sum())
overflow_zones = int((new_data["collection_status"] == "overflow").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏋️ Total Waste Today", f"{total_waste:,} kg")
k2.metric("🚛 Trucks Deployed", total_trucks)
k3.metric("⏳ Pending Zones", pending_zones, delta=f"{pending_zones} need attention", delta_color="inverse")
k4.metric("🚨 Overflow Zones", overflow_zones, delta=f"{overflow_zones} critical", delta_color="inverse")

# ─────────────────────────────────────────────
# KPI CARDS — ROW 2 (Cost & Carbon)
# ─────────────────────────────────────────────
problem_count    = overflow_zones + pending_zones
extra_fuel_cost  = problem_count * 500
co2_savings      = problem_count * 25

c1, c2 = st.columns(2)
c1.metric("💰 Extra Fuel Cost Today (₹)", f"₹{extra_fuel_cost:,}", help="Calculated as (overflow + pending zones) × ₹500")
c2.metric("🌱 CO₂ Savings if Optimised", f"{co2_savings} kg", help="Calculated as (overflow + pending zones) × 25 kg CO₂")

st.divider()

# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Waste Collected per Zone")
    bar_fig = px.bar(
        filtered.sort_values("waste_kg", ascending=True),
        x="waste_kg", y="zone", orientation="h",
        color="collection_status",
        color_discrete_map={"collected": "#52B788", "pending": "#F9C74F", "overflow": "#E63946"},
        text="waste_kg",
        labels={"waste_kg": "Waste (kg)", "zone": "Zone"},
        hover_data=["area", "trucks_deployed"]
    )
    bar_fig.update_traces(textposition="outside")
    bar_fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend_title="Status", height=350,
        margin=dict(l=10, r=20, t=10, b=10)
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    st.subheader("♻️ Waste Type Composition")
    pie_data = history_filtered.groupby("waste_type")["waste_kg"].sum().reset_index()
    pie_fig = px.pie(
        pie_data, values="waste_kg", names="waste_type",
        color_discrete_sequence=["#52B788", "#1A7A9A", "#F9C74F", "#E63946"],
        hole=0.4
    )
    pie_fig.update_traces(textinfo="percent+label")
    pie_fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)

# ─────────────────────────────────────────────
# LINE CHART — TREND + ML PREDICTION
# ─────────────────────────────────────────────
st.subheader("📈 Waste Trend — Last 30 Minutes")

if not history_filtered.empty:
    trend_data = (
        history_filtered.groupby(["timestamp", "zone"])["waste_kg"]
        .sum().reset_index()
    )
    line_fig = px.line(
        trend_data, x="timestamp", y="waste_kg", color="zone",
        labels={"waste_kg": "Waste (kg)", "timestamp": "Time"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Add ML predictions as annotations
    for zone in selected_zones:
        pred = predict_next_waste(st.session_state.zone_history.get(zone, []))
        if pred is not None:
            line_fig.add_annotation(
                x=datetime.now(), y=pred,
                text=f"🔮 {zone}: {pred}kg",
                showarrow=True, arrowhead=2,
                font=dict(size=10, color="#1A7A9A"),
                bgcolor="white", bordercolor="#1A7A9A", borderwidth=1
            )

    line_fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend_title="Zone"
    )
    st.plotly_chart(line_fig, use_container_width=True)
    st.caption("🔮 Annotations show ML-predicted next value per zone (Linear Regression on last 10 readings)")
else:
    st.info("Building trend data... please wait a few seconds.")

st.divider()

# ─────────────────────────────────────────────
# ALERTS PANEL + DATA TABLE
# ─────────────────────────────────────────────
col_alert, col_table = st.columns([1, 2])

with col_alert:
    st.subheader("🚨 Live Alerts")
    any_alert = False
    for _, row in new_data.iterrows():
        if row["collection_status"] == "overflow":
            st.error(f"🔴 **{row['zone']} ({ZONE_LABELS[row['zone']]})** — Overflow! Extra truck needed.")
            any_alert = True
        elif st.session_state.pending_cycles.get(row["zone"], 0) >= 2:
            st.warning(f"🟡 **{row['zone']} ({ZONE_LABELS[row['zone']]})** — Pending for {st.session_state.pending_cycles[row['zone']]} cycles. Check crew!")
            any_alert = True
    if not any_alert:
        st.success("✅ All zones operating normally.")

with col_table:
    st.subheader("📋 Current Zone Status")
    display_df = filtered[["zone", "area", "waste_kg", "waste_type", "trucks_deployed", "collection_status"]].copy()
    display_df.columns = ["Zone", "Area", "Waste (kg)", "Type", "Trucks", "Status"]

    def color_status(val):
        if val == "overflow":   return "background-color: #FADADD; color: #C0392B; font-weight: bold"
        elif val == "pending":  return "background-color: #FFF3CD; color: #856404; font-weight: bold"
        else:                   return "background-color: #D4EDDA; color: #155724"

    st.dataframe(
        display_df.style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True
    )

st.divider()

# ─────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────
st.subheader("🗺️ Trichy Zone Problem Heatmap")
st.caption("🔴 High risk (≥50%)  •  🟠 Medium risk (20–50%)  •  🟢 Low risk (<20%)  •  Circle size = problem severity")

problem_scores = calculate_problem_scores(st.session_state.history)
trichy_map = build_map(problem_scores)
st_folium(trichy_map, width=None, height=420, returned_objects=[])

# Score table
score_df = pd.DataFrame([
    {"Zone": z, "Area": ZONE_LABELS[z], "Problem Score (%)": problem_scores[z],
     "Risk Level": "🔴 High" if problem_scores[z] >= 50 else ("🟠 Medium" if problem_scores[z] >= 20 else "🟢 Low")}
    for z in ZONES
]).sort_values("Problem Score (%)", ascending=False)
st.dataframe(score_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────
# ML PREDICTIONS TABLE
# ─────────────────────────────────────────────
st.subheader("🔮 ML Predictions — Next Waste (kg) per Zone")
pred_rows = []
for zone in ZONES:
    history_vals = st.session_state.zone_history.get(zone, [])
    current = history_vals[-1] if history_vals else None
    predicted = predict_next_waste(history_vals)
    if current and predicted:
        diff = predicted - current
        trend = "📈 Up" if diff > 20 else ("📉 Down" if diff < -20 else "➡️ Stable")
    else:
        trend = "⏳ Collecting data..."
    pred_rows.append({
        "Zone": zone,
        "Area": ZONE_LABELS[zone],
        "Current (kg)": round(current, 1) if current else "—",
        "Predicted Next (kg)": predicted if predicted else "Need 5+ readings",
        "Trend": trend
    })
st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────
time.sleep(refresh_interval)
st.rerun()
