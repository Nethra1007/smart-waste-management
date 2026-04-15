"""
Smart Waste Collection Data Analytics Dashboard
Trichy, Tamil Nadu — Real-Time Simulation
Features: Login, Theme Customisation, SQLite Historical Data, Live Dashboard
"""

import time, random, base64, sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be very first Streamlit call)
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
CREDENTIALS   = {"admin": "admin123"}
ZONES         = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E", "Zone F"]
ZONE_LABELS   = {
    "Zone A": "Srirangam",       "Zone B": "Ariyamangalam",
    "Zone C": "Aviyur",          "Zone D": "Golden Rock",
    "Zone E": "Palakkarai",      "Zone F": "Thiruverumbur",
}
ZONE_COORDS   = {
    "Zone A": (10.8650, 78.6930), "Zone B": (10.7950, 78.7350),
    "Zone C": (10.8200, 78.6600), "Zone D": (10.8400, 78.7200),
    "Zone E": (10.8050, 78.6850), "Zone F": (10.7750, 78.7100),
}
WASTE_TYPES   = ["organic", "recyclable", "general", "hazardous"]
WASTE_WEIGHTS = [0.50, 0.30, 0.15, 0.05]
ALERT_PHONE   = "+91-9876543210"
DB_PATH       = "waste_data.db"

# ─────────────────────────────────────────────
# THEME PALETTES
# ─────────────────────────────────────────────
THEMES = {
    "Light": {
        "bg":          "#F0F7F4",
        "card_bg":     "rgba(255,255,255,0.88)",
        "text":        "#1B2428",
        "subtext":     "#4A5568",
        "header_bg":   "#0B4F6C",
        "header_text": "#FFFFFF",
        "accent":      "#0B4F6C",
        "border":      "#C8E6C9",
    },
    "Dark": {
        "bg":          "#0D1B2A",
        "card_bg":     "rgba(20,40,60,0.90)",
        "text":        "#E8F5E9",
        "subtext":     "#A8D8EA",
        "header_bg":   "#052030",
        "header_text": "#A8D8EA",
        "accent":      "#52B788",
        "border":      "#1A5276",
    },
}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "logged_in":      False,
    "history":        pd.DataFrame(),
    "pending_cycles": {z: 0 for z in ZONES},
    "zone_history":   {z: [] for z in ZONES},
    "last_whatsapp":  None,
    "overflow_log":   [],
    "current_date":   datetime.now().date(),
    "theme":          "Light",
    "bg_css":         "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
def init_db():
    """Create SQLite DB and daily_stats table if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date               TEXT PRIMARY KEY,
            total_waste_kg     REAL,
            avg_waste_per_zone REAL,
            overflow_count     INTEGER,
            pending_count      INTEGER
        )
    """)
    con.commit()
    con.close()


def save_daily_stats(date_str: str, df: pd.DataFrame):
    """Aggregate session history and upsert into daily_stats."""
    if df.empty:
        return
    total   = round(float(df["waste_kg"].sum()), 2)
    avg     = round(float(df.groupby("zone")["waste_kg"].mean().mean()), 2)
    oflow   = int((df["collection_status"] == "overflow").sum())
    pending = int((df["collection_status"] == "pending").sum())
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO daily_stats
            (date, total_waste_kg, avg_waste_per_zone, overflow_count, pending_count)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_waste_kg     = excluded.total_waste_kg,
            avg_waste_per_zone = excluded.avg_waste_per_zone,
            overflow_count     = excluded.overflow_count,
            pending_count      = excluded.pending_count
    """, (date_str, total, avg, oflow, pending))
    con.commit()
    con.close()


def load_historical(days: int = 30) -> pd.DataFrame:
    """Load last N days of daily stats from SQLite."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?",
        con, params=(days,)
    )
    con.close()
    return df.sort_values("date").reset_index(drop=True)


init_db()

# ─────────────────────────────────────────────
# CUSTOM CSS / THEME
# ─────────────────────────────────────────────
def apply_theme(theme_name: str, bg_css: str = ""):
    """Inject CSS for the selected theme and optional background."""
    t = THEMES[theme_name]
    bg_rule = bg_css if bg_css else f"background-color: {t['bg']};"
    st.markdown(f"""
    <style>
    .stApp {{
        {bg_rule}
        color: {t['text']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {t['header_bg']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {t['header_text']} !important;
    }}
    [data-testid="stMetric"] {{
        background: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(6px);
    }}
    [data-testid="stMetricLabel"] {{ color: {t['subtext']} !important; font-size: 13px; }}
    [data-testid="stMetricValue"] {{ color: {t['text']}    !important; font-weight: 700; }}
    h1, h2, h3 {{ color: {t['accent']} !important; }}
    [data-testid="stDataFrame"] {{ background: {t['card_bg']}; border-radius: 10px; }}
    hr {{ border-color: {t['border']}; }}
    .stAlert {{ background: {t['card_bg']} !important; backdrop-filter: blur(4px); }}
    .stButton > button {{
        background-color: {t['accent']};
        color: #FFFFFF; border: none;
        border-radius: 8px; padding: 8px 20px; font-weight: 600;
    }}
    .stButton > button:hover {{ opacity: 0.85; }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def show_login():
    """Render the login screen."""
    apply_theme("Light")
    st.markdown("""
        <div style='text-align:center; padding:60px 0 10px 0;'>
            <span style='font-size:64px;'>🗑️</span>
            <h1 style='color:#0B4F6C; margin-bottom:4px;'>Smart Waste Analytics</h1>
            <p style='color:#4A5568; font-size:16px;'>Trichy, Tamil Nadu — Urban Management System</p>
        </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
            <div style='background:white;border-radius:16px;padding:32px 28px;
                        box-shadow:0 4px 24px rgba(0,0,0,0.10);margin-top:10px;'>
                <h3 style='color:#0B4F6C;text-align:center;margin-bottom:20px;'>🔐 Login</h3>
            </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Login →", use_container_width=True):
            if CREDENTIALS.get(username) == password:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        st.markdown(
            "<p style='text-align:center;color:#888;font-size:12px;margin-top:12px;'>"
            "Demo credentials: <b>admin / admin123</b></p>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
def generate_live_data(festival_mode: bool = False) -> pd.DataFrame:
    """Produce one sensor reading per zone with time-of-day patterns."""
    now  = datetime.now()
    hour = now.hour
    if   7  <= hour <= 10: base = 1.3
    elif 17 <= hour <= 20: base = 1.2
    elif 0  <= hour <= 5:  base = 0.6
    else:                   base = 1.0
    if festival_mode:
        base *= 1.5

    rows = []
    for zone in ZONES:
        waste_kg = round(min(random.uniform(300, 1200) * base, 1400), 1)
        if   waste_kg > 1000:          status = "overflow"
        elif random.random() < 0.15:   status = "pending"
        else:                           status = "collected"
        rows.append({
            "zone":               zone,
            "area":               ZONE_LABELS[zone],
            "timestamp":          now,
            "waste_kg":           waste_kg,
            "waste_type":         random.choices(WASTE_TYPES, weights=WASTE_WEIGHTS)[0],
            "trucks_deployed":    random.randint(2, 5),
            "collection_status":  status,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# PROBLEM SCORE
# ─────────────────────────────────────────────
def calculate_problem_scores(history: pd.DataFrame) -> dict:
    """Return problem score (%) per zone based on overflow & pending history."""
    if history.empty:
        return {z: 0 for z in ZONES}
    scores = {}
    for zone in ZONES:
        zdf = history[history["zone"] == zone]
        if len(zdf) == 0:
            scores[zone] = 0
            continue
        ov  = (zdf["collection_status"] == "overflow").sum()
        pen = (zdf["collection_status"] == "pending").sum()
        scores[zone] = round(min((ov * 2 + pen) / len(zdf) * 100, 100), 1)
    return scores

# ─────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────
def build_map(problem_scores: dict) -> folium.Map:
    """Build colour-coded Trichy zone map."""
    m = folium.Map(location=[10.8200, 78.6900], zoom_start=12, tiles="CartoDB positron")
    for zone, (lat, lon) in ZONE_COORDS.items():
        score = problem_scores.get(zone, 0)
        color = "red" if score >= 50 else ("orange" if score >= 20 else "green")
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 + score * 0.3,
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{zone} — {ZONE_LABELS[zone]}</b><br>Problem Score: {score:.1f}%",
                max_width=200),
            tooltip=f"{zone}: {score:.1f}%"
        ).add_to(m)
    return m

# ─────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────
def predict_next_waste(zone_hist: list):
    """Predict next waste_kg using linear regression on last 10 values."""
    if len(zone_hist) < 5:
        return None
    values = zone_hist[-10:]
    X = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression().fit(X, np.array(values))
    return round(max(float(model.predict([[len(values)]])[0]), 0), 1)

# ─────────────────────────────────────────────
# WHATSAPP SIMULATION
# ─────────────────────────────────────────────
def simulate_whatsapp_alert(zone: str):
    """Simulate a WhatsApp overflow alert (console + session state)."""
    msg = (f"[SIMULATED WHATSAPP] Alert → {ALERT_PHONE}: "
           f"Overflow in {zone} ({ZONE_LABELS[zone]})! Extra truck needed.")
    print(msg)
    st.session_state.last_whatsapp = {
        "message": f"🚨 Overflow in {zone} ({ZONE_LABELS[zone]})! Extra truck needed.",
        "time":    datetime.now().strftime("%H:%M:%S"),
        "phone":   ALERT_PHONE,
    }

# ─────────────────────────────────────────────
# HISTORICAL DATA TAB
# ─────────────────────────────────────────────
def render_historical_tab(theme: str):
    """Render the Historical Data tab content."""
    t = THEMES[theme]
    st.subheader("📅 Historical Daily Statistics — Last 30 Days")
    hist_df = load_historical(30)

    if hist_df.empty:
        st.info("No historical data yet. Data is saved every refresh cycle. Keep the app running and records will appear here.")
        return

    # Line chart — total waste trend
    fig_line = px.line(
        hist_df, x="date", y="total_waste_kg",
        markers=True, title="Total Waste Collected per Day (kg)",
        labels={"total_waste_kg": "Total Waste (kg)", "date": "Date"},
        color_discrete_sequence=[t["accent"]],
    )
    fig_line.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=300, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart — overflow vs pending
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=hist_df["date"], y=hist_df["overflow_count"],
                              name="Overflow", marker_color="#E63946"))
    fig_bar.add_trace(go.Bar(x=hist_df["date"], y=hist_df["pending_count"],
                              name="Pending",  marker_color="#F9C74F"))
    fig_bar.update_layout(
        barmode="group", title="Overflow vs Pending Count per Day",
        plot_bgcolor="white", paper_bgcolor="white",
        height=300, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Table
    st.subheader("📋 All Stored Records")
    st.dataframe(
        hist_df.rename(columns={
            "date":               "Date",
            "total_waste_kg":     "Total Waste (kg)",
            "avg_waste_per_zone": "Avg per Zone (kg)",
            "overflow_count":     "Overflow Count",
            "pending_count":      "Pending Count",
        }),
        use_container_width=True, hide_index=True
    )

    # CSV download
    csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Historical Data as CSV",
        data=csv_bytes,
        file_name="waste_historical_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
def render_dashboard():
    """Render the full dashboard after login."""

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("### 🗑️ Smart Waste Analytics")
        st.caption("Trichy, Tamil Nadu")
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

        st.divider()

        # Theme
        st.subheader("🎨 Appearance")
        theme = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            horizontal=True
        )
        st.session_state.theme = theme

        # Background image
        st.caption("Background Image (optional)")
        bg_url  = st.text_input("Image URL", placeholder="https://…")
        bg_file = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])

        bg_css = ""
        if bg_file:
            b64 = base64.b64encode(bg_file.read()).decode()
            ext = bg_file.name.split(".")[-1]
            bg_css = (f"background-image: url('data:image/{ext};base64,{b64}');"
                      "background-size:cover;background-attachment:fixed;background-position:center;")
        elif bg_url.strip():
            bg_css = (f"background-image: url('{bg_url.strip()}');"
                      "background-size:cover;background-attachment:fixed;background-position:center;")
        st.session_state.bg_css = bg_css

        st.divider()

        # Controls
        refresh_interval = st.slider("⏱️ Refresh interval (s)", 2, 10, 5)
        selected_zones   = st.multiselect("📍 Zones",      ZONES,       default=ZONES)
        selected_types   = st.multiselect("♻️ Waste Type", WASTE_TYPES, default=WASTE_TYPES)

        st.divider()
        festival_mode = st.checkbox("🎉 Festival Mode (Diwali / Pongal)")

        st.divider()
        st.subheader("📱 Last WhatsApp Alert")
        if st.session_state.last_whatsapp:
            wa = st.session_state.last_whatsapp
            st.error(f"**{wa['message']}**")
            st.caption(f"Sent to {wa['phone']} at {wa['time']}")
        else:
            st.info("No alerts yet.")

    # Apply theme
    apply_theme(st.session_state.theme, st.session_state.bg_css)

    # Festival banner
    if festival_mode:
        st.warning("🎉 **Festival Mode ON** — Waste volume increased by 50% (Diwali / Pongal simulation)")

    # ── HEADER ──
    st.title("🗑️ Smart Waste Collection — Trichy Live Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}  |  "
               f"Auto-refreshing every {refresh_interval}s  |  "
               f"👤 Logged in as admin")

    # ── TABS ──
    tab_live, tab_hist = st.tabs(["📡 Live Dashboard", "📅 Historical Data"])

    # ─────────────────────────────────────────
    # GENERATE LIVE DATA & BOOKKEEPING
    # ─────────────────────────────────────────
    new_data = generate_live_data(festival_mode=festival_mode)

    # Day-change detection → save yesterday's stats, reset history
    today = datetime.now().date()
    if today != st.session_state.current_date:
        save_daily_stats(str(st.session_state.current_date), st.session_state.history)
        st.session_state.current_date = today
        st.session_state.history      = pd.DataFrame()

    # Rolling 30-min history
    st.session_state.history = pd.concat(
        [st.session_state.history, new_data], ignore_index=True
    )
    cutoff = datetime.now() - timedelta(minutes=30)
    st.session_state.history = st.session_state.history[
        st.session_state.history["timestamp"] >= cutoff
    ]

    # Save today's running aggregates to DB on every cycle
    save_daily_stats(str(today), st.session_state.history)

    # Update ML zone histories
    for zone in ZONES:
        val = new_data[new_data["zone"] == zone]["waste_kg"].values
        if len(val) > 0:
            st.session_state.zone_history[zone].append(float(val[0]))
            st.session_state.zone_history[zone] = st.session_state.zone_history[zone][-20:]

    # Pending cycle counter
    for zone in ZONES:
        s = new_data[new_data["zone"] == zone]["collection_status"].values
        if len(s) > 0 and s[0] == "pending":
            st.session_state.pending_cycles[zone] += 1
        else:
            st.session_state.pending_cycles[zone] = 0

    # WhatsApp overflow alerts (deduplicated against last 3)
    recent = [x["zone"] for x in st.session_state.overflow_log[-3:]]
    for _, row in new_data.iterrows():
        if row["collection_status"] == "overflow" and row["zone"] not in recent:
            simulate_whatsapp_alert(row["zone"])
            st.session_state.overflow_log.append({"zone": row["zone"], "time": datetime.now()})

    # Apply filters
    filtered      = new_data[
        new_data["zone"].isin(selected_zones) &
        new_data["waste_type"].isin(selected_types)
    ]
    hist_filtered = st.session_state.history[
        st.session_state.history["zone"].isin(selected_zones) &
        st.session_state.history["waste_type"].isin(selected_types)
    ]

    # ─────────────────────────────────────────
    # TAB 1: LIVE DASHBOARD
    # ─────────────────────────────────────────
    with tab_live:

        # KPI Row 1
        total_waste    = int(hist_filtered["waste_kg"].sum())
        total_trucks   = int(new_data["trucks_deployed"].sum())
        pending_zones  = int((new_data["collection_status"] == "pending").sum())
        overflow_zones = int((new_data["collection_status"] == "overflow").sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏋️ Total Waste Today",  f"{total_waste:,} kg")
        k2.metric("🚛 Trucks Deployed",     total_trucks)
        k3.metric("⏳ Pending Zones",       pending_zones,
                  delta=f"{pending_zones} need attention", delta_color="inverse")
        k4.metric("🚨 Overflow Zones",      overflow_zones,
                  delta=f"{overflow_zones} critical",      delta_color="inverse")

        # KPI Row 2 — Cost & Carbon
        prob = overflow_zones + pending_zones
        c1, c2 = st.columns(2)
        c1.metric("💰 Extra Fuel Cost Today (₹)", f"₹{prob * 500:,}",
                  help="(overflow + pending zones) × ₹500")
        c2.metric("🌱 CO₂ Savings if Optimised",  f"{prob * 25} kg",
                  help="(overflow + pending zones) × 25 kg CO₂")

        st.divider()

        # Bar + Pie charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Waste Collected per Zone")
            fig_bar = px.bar(
                filtered.sort_values("waste_kg"),
                x="waste_kg", y="zone", orientation="h",
                color="collection_status",
                color_discrete_map={
                    "collected": "#52B788",
                    "pending":   "#F9C74F",
                    "overflow":  "#E63946",
                },
                text="waste_kg",
                hover_data=["area", "trucks_deployed"],
                labels={"waste_kg": "Waste (kg)", "zone": "Zone"},
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=350, margin=dict(l=10, r=20, t=10, b=10),
                legend_title="Status",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("♻️ Waste Type Composition")
            pie_data = hist_filtered.groupby("waste_type")["waste_kg"].sum().reset_index()
            fig_pie  = px.pie(
                pie_data, values="waste_kg", names="waste_type", hole=0.4,
                color_discrete_sequence=["#52B788", "#1A7A9A", "#F9C74F", "#E63946"],
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Line chart + ML annotations
        st.subheader("📈 Waste Trend — Last 30 Minutes")
        if not hist_filtered.empty:
            trend = (hist_filtered
                     .groupby(["timestamp", "zone"])["waste_kg"]
                     .sum().reset_index())
            fig_line = px.line(
                trend, x="timestamp", y="waste_kg", color="zone",
                labels={"waste_kg": "Waste (kg)", "timestamp": "Time"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            for zone in selected_zones:
                pred = predict_next_waste(st.session_state.zone_history.get(zone, []))
                if pred is not None:
                    fig_line.add_annotation(
                        x=datetime.now(), y=pred,
                        text=f"🔮 {zone}: {pred}kg",
                        showarrow=True, arrowhead=2,
                        font=dict(size=10, color="#0B4F6C"),
                        bgcolor="white", bordercolor="#0B4F6C", borderwidth=1,
                    )
            fig_line.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=320, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("🔮 Annotations = ML-predicted next value (Linear Regression on last 10 readings)")
        else:
            st.info("Building trend data… please wait a few seconds.")

        st.divider()

        # Alerts + Status table
        col_a, col_t = st.columns([1, 2])
        with col_a:
            st.subheader("🚨 Live Alerts")
            any_alert = False
            for _, row in new_data.iterrows():
                if row["collection_status"] == "overflow":
                    st.error(
                        f"🔴 **{row['zone']} ({ZONE_LABELS[row['zone']]})** — "
                        "Overflow! Extra truck needed."
                    )
                    any_alert = True
                elif st.session_state.pending_cycles.get(row["zone"], 0) >= 2:
                    cyc = st.session_state.pending_cycles[row["zone"]]
                    st.warning(
                        f"🟡 **{row['zone']} ({ZONE_LABELS[row['zone']]})** — "
                        f"Pending {cyc} cycles. Check crew!"
                    )
                    any_alert = True
            if not any_alert:
                st.success("✅ All zones operating normally.")

        with col_t:
            st.subheader("📋 Current Zone Status")
            disp = filtered[
                ["zone","area","waste_kg","waste_type","trucks_deployed","collection_status"]
            ].copy()
            disp.columns = ["Zone","Area","Waste (kg)","Type","Trucks","Status"]

            def color_status(val):
                if val == "overflow":
                    return "background-color:#FADADD;color:#C0392B;font-weight:bold"
                elif val == "pending":
                    return "background-color:#FFF3CD;color:#856404;font-weight:bold"
                return "background-color:#D4EDDA;color:#155724"

            st.dataframe(
                disp.style.map(color_status, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        # Folium map
        st.subheader("🗺️ Trichy Zone Problem Heatmap")
        st.caption("🔴 High (≥50%)  •  🟠 Medium (20–50%)  •  🟢 Low (<20%)  •  Circle size = severity")
        scores = calculate_problem_scores(st.session_state.history)
        st_folium(build_map(scores), width=None, height=420, returned_objects=[])

        score_rows = [{
            "Zone": z, "Area": ZONE_LABELS[z],
            "Problem Score (%)": scores[z],
            "Risk": ("🔴 High" if scores[z] >= 50
                     else ("🟠 Medium" if scores[z] >= 20 else "🟢 Low")),
        } for z in ZONES]
        st.dataframe(
            pd.DataFrame(score_rows).sort_values("Problem Score (%)", ascending=False),
            use_container_width=True, hide_index=True,
        )

        st.divider()

        # ML predictions table
        st.subheader("🔮 ML Predictions — Next Waste (kg) per Zone")
        pred_rows = []
        for zone in ZONES:
            zh   = st.session_state.zone_history.get(zone, [])
            curr = zh[-1] if zh else None
            pred = predict_next_waste(zh)
            if curr and pred:
                diff  = pred - curr
                trend = "📈 Up" if diff > 20 else ("📉 Down" if diff < -20 else "➡️ Stable")
            else:
                trend = "⏳ Collecting data…"
            pred_rows.append({
                "Zone":          zone,
                "Area":          ZONE_LABELS[zone],
                "Current (kg)":  round(curr, 1) if curr else "—",
                "Predicted (kg)": pred if pred else "Need 5+ readings",
                "Trend":          trend,
            })
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────
    # TAB 2: HISTORICAL DATA
    # ─────────────────────────────────────────
    with tab_hist:
        render_historical_tab(st.session_state.theme)

    # ── AUTO REFRESH ──
    time.sleep(refresh_interval)
    st.rerun()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if not st.session_state["logged_in"]:
    show_login()
else:
    render_dashboard()
