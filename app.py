"""
Smart Waste Collection Data Analytics Dashboard
Trichy, Tamil Nadu — Real-Time Simulation
Features: Public Login, Multiple Contrast Themes, SQLite Historical Data, Live Dashboard
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
ADMIN_CREDENTIALS = {"admin": "admin123"}
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
# MULTIPLE CONTRAST THEMES (More attractive!)
# ─────────────────────────────────────────────
THEMES = {
    # Professional Dark (High Contrast)
    "Professional Dark": {
        "bg":          "#0A0E17",
        "card_bg":     "rgba(20, 28, 40, 0.92)",
        "text":        "#F0F4F8",
        "subtext":     "#94A3B8",
        "header_bg":   "#020617",
        "header_text": "#38BDF8",
        "accent":      "#38BDF8",
        "border":      "#1E293B",
        "success":     "#22C55E",
        "warning":     "#F59E0B",
        "danger":      "#EF4444",
    },
    # Nature Green (Earthy & Calming)
    "Nature Green": {
        "bg":          "#E8F5E9",
        "card_bg":     "rgba(255, 255, 255, 0.90)",
        "text":        "#1B5E20",
        "subtext":     "#2E7D32",
        "header_bg":   "#1B5E20",
        "header_text": "#FFFFFF",
        "accent":      "#4CAF50",
        "border":      "#A5D6A7",
        "success":     "#2E7D32",
        "warning":     "#F57C00",
        "danger":      "#D32F2F",
    },
    # Ocean Blue (Cool & Professional)
    "Ocean Blue": {
        "bg":          "#E0F7FA",
        "card_bg":     "rgba(255, 255, 255, 0.88)",
        "text":        "#006064",
        "subtext":     "#00838F",
        "header_bg":   "#006064",
        "header_text": "#E0F7FA",
        "accent":      "#00ACC1",
        "border":      "#80DEEA",
        "success":     "#00897B",
        "warning":     "#FFB74D",
        "danger":      "#E53935",
    },
    # Sunset Warm (Vibrant & Energetic)
    "Sunset Warm": {
        "bg":          "#FFF3E0",
        "card_bg":     "rgba(255, 245, 235, 0.92)",
        "text":        "#BF360C",
        "subtext":     "#D84315",
        "header_bg":   "#BF360C",
        "header_text": "#FFF3E0",
        "accent":      "#FF7043",
        "border":      "#FFCCBC",
        "success":     "#689F38",
        "warning":     "#FFA726",
        "danger":      "#E53935",
    },
    # Monochrome Elegant (Minimalist)
    "Monochrome": {
        "bg":          "#F5F5F5",
        "card_bg":     "rgba(255, 255, 255, 0.95)",
        "text":        "#212121",
        "subtext":     "#616161",
        "header_bg":   "#212121",
        "header_text": "#E0E0E0",
        "accent":      "#757575",
        "border":      "#E0E0E0",
        "success":     "#43A047",
        "warning":     "#FB8C00",
        "danger":      "#E53935",
    },
    # Purple Royal (Premium Look)
    "Purple Royal": {
        "bg":          "#F3E5F5",
        "card_bg":     "rgba(255, 255, 255, 0.90)",
        "text":        "#4A148C",
        "subtext":     "#6A1B9A",
        "header_bg":   "#4A148C",
        "header_text": "#F3E5F5",
        "accent":      "#AB47BC",
        "border":      "#CE93D8",
        "success":     "#2E7D32",
        "warning":     "#F57C00",
        "danger":      "#D32F2F",
    },
}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "logged_in":      False,
    "username":       None,
    "is_admin":       False,
    "history":        pd.DataFrame(),
    "pending_cycles": {z: 0 for z in ZONES},
    "zone_history":   {z: [] for z in ZONES},
    "last_whatsapp":  None,
    "overflow_log":   [],
    "current_date":   datetime.now().date(),
    "theme":          "Ocean Blue",
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
# CUSTOM CSS / THEME (Enhanced with gradients)
# ─────────────────────────────────────────────
def apply_theme(theme_name: str, bg_css: str = ""):
    """Inject CSS for the selected theme with enhanced styling."""
    t = THEMES[theme_name]
    
    # Default gradient backgrounds per theme
    gradient_bgs = {
        "Professional Dark": "linear-gradient(135deg, #0A0E17 0%, #1A1F2E 100%)",
        "Nature Green":      "linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%)",
        "Ocean Blue":        "linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 100%)",
        "Sunset Warm":       "linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%)",
        "Monochrome":        "linear-gradient(135deg, #F5F5F5 0%, #E0E0E0 100%)",
        "Purple Royal":      "linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%)",
    }
    
    bg_style = bg_css if bg_css else f"background: {gradient_bgs.get(theme_name, t['bg'])};"
    
    st.markdown(f"""
    <style>
    /* Main app container */
    .stApp {{
        {bg_style}
        background-attachment: fixed;
        color: {t['text']};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: {t['header_bg']} !important;
        border-right: 2px solid {t['border']};
    }}
    [data-testid="stSidebar"] * {{
        color: {t['header_text']} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown {{
        color: {t['header_text']} !important;
    }}
    
    /* Metric cards with glassmorphism */
    [data-testid="stMetric"] {{
        background: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 18px 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}
    [data-testid="stMetricLabel"] {{ 
        color: {t['subtext']} !important; 
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    [data-testid="stMetricValue"] {{ 
        color: {t['text']} !important; 
        font-weight: 800;
        font-size: 28px;
    }}
    
    /* Headers */
    h1, h2, h3, h4 {{
        color: {t['accent']} !important;
        font-weight: 700;
    }}
    h1 {{
        background: linear-gradient(135deg, {t['accent']}, {t['text']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* Data frames */
    [data-testid="stDataFrame"] {{ 
        background: {t['card_bg']}; 
        border-radius: 12px;
        border: 1px solid {t['border']};
    }}
    
    /* Dividers */
    hr {{ 
        border-color: {t['border']};
        margin: 1rem 0;
    }}
    
    /* Alerts with theme colors */
    .stAlert {{ 
        background: {t['card_bg']} !important; 
        backdrop-filter: blur(8px);
        border-left: 4px solid;
        border-radius: 10px;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {t['accent']}, {t['text']});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Success/Warning/Error messages */
    .element-container div[data-testid="stMarkdown"] .st-emotion-cache-1y4p8pa {{
        color: {t['success']};
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {t['card_bg']};
        border-radius: 12px 12px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['accent']};
        color: white;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {t['card_bg']};
        border-radius: 10px;
        border: 1px solid {t['border']};
    }}
    
    /* Code blocks */
    .stCodeBlock {{
        background: {t['card_bg']};
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN PAGE (Updated: Anyone can login!)
# ─────────────────────────────────────────────
def show_login():
    """Render the login screen with public access option."""
    apply_theme("Ocean Blue")
    
    st.markdown("""
        <div style='text-align:center; padding:50px 0 20px 0;'>
            <span style='font-size:72px;'>🗑️🌿</span>
            <h1 style='background: linear-gradient(135deg, #006064, #00ACC1); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent;
                       margin-bottom:8px; font-size:42px;'>
                Smart Waste Analytics
            </h1>
            <p style='color:#4A5568; font-size:18px;'>
                Trichy, Tamil Nadu — Urban Management System
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
            <div style='background:rgba(255,255,255,0.92); border-radius:24px; 
                        padding:36px 32px; box-shadow:0 8px 32px rgba(0,0,0,0.12);
                        backdrop-filter:blur(4px);'>
                <h3 style='color:#006064; text-align:center; margin-bottom:24px;'>🔐 Access Dashboard</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Two login options
        login_option = st.radio(
            "Login as:",
            ["👤 Guest (View Only)", "👑 Admin (Full Access)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if "Admin" in login_option:
            # Admin login
            username = st.text_input("Username", placeholder="admin", key="admin_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="admin_pass")
            
            if st.button("🔓 Admin Login", use_container_width=True):
                if ADMIN_CREDENTIALS.get(username) == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials. Use admin / admin123")
        else:
            # Guest login - anyone can enter!
            guest_name = st.text_input("Your Name", placeholder="Enter your name (e.g., Ramesh, Priya)", key="guest_name")
            st.caption("💡 No password needed — just enter any name to continue as guest.")
            
            if st.button("🚪 Enter as Guest", use_container_width=True):
                if guest_name and guest_name.strip():
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = guest_name.strip()
                    st.session_state["is_admin"] = False
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter your name to continue.")
        
        st.markdown("""
            <p style='text-align:center; color:#888; font-size:12px; margin-top:24px;'>
            🌟 <b>Guest Mode</b>: View real-time waste analytics<br>
            👑 <b>Admin Mode</b>: Full access including historical data export
            </p>
        """, unsafe_allow_html=True)

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
def build_map(problem_scores: dict, theme_name: str) -> folium.Map:
    """Build colour-coded Trichy zone map with theme-aware tiles."""
    t = THEMES[theme_name]
    # Choose tile based on theme
    if theme_name in ["Professional Dark", "Monochrome"]:
        tiles = "CartoDB dark_matter"
    else:
        tiles = "CartoDB positron"
    
    m = folium.Map(location=[10.8200, 78.6900], zoom_start=12, tiles=tiles)
    
    for zone, (lat, lon) in ZONE_COORDS.items():
        score = problem_scores.get(zone, 0)
        if score >= 50:
            color = "#EF4444"  # Red
            fill_color = "#EF4444"
        elif score >= 20:
            color = "#F59E0B"  # Orange
            fill_color = "#F59E0B"
        else:
            color = "#22C55E"  # Green
            fill_color = "#22C55E"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 + score * 0.3,
            color=color, fill=True, fill_color=fill_color, fill_opacity=0.7,
            weight=2,
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
# HISTORICAL DATA TAB (Admin only)
# ─────────────────────────────────────────────
def render_historical_tab(theme: str):
    """Render the Historical Data tab content (Admin only)."""
    t = THEMES[theme]
    
    if not st.session_state.is_admin:
        st.warning("🔒 **Admin Access Required**")
        st.info("Historical data and CSV export are only available for admin users. Please login as admin (admin/admin123) to access this feature.")
        return
    
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
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=t["text"])
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart — overflow vs pending
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=hist_df["date"], y=hist_df["overflow_count"],
                              name="Overflow", marker_color=t["danger"]))
    fig_bar.add_trace(go.Bar(x=hist_df["date"], y=hist_df["pending_count"],
                              name="Pending",  marker_color=t["warning"]))
    fig_bar.update_layout(
        barmode="group", title="Overflow vs Pending Count per Day",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color=t["text"])
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
        # User info with role badge
        if st.session_state.is_admin:
            st.markdown(f"""
            <div style='text-align:center; padding:10px; background:rgba(255,255,255,0.1); border-radius:12px;'>
                <span style='font-size:32px;'>👑</span>
                <p style='margin:0; font-weight:bold;'>{st.session_state.username}</p>
                <span style='background:#22C55E; padding:2px 10px; border-radius:20px; font-size:11px;'>ADMIN</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:10px; background:rgba(255,255,255,0.1); border-radius:12px;'>
                <span style='font-size:32px;'>👤</span>
                <p style='margin:0; font-weight:bold;'>{st.session_state.username}</p>
                <span style='background:#F59E0B; padding:2px 10px; border-radius:20px; font-size:11px;'>GUEST</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 🗑️ Smart Waste Analytics")
        st.caption("Trichy, Tamil Nadu")
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["is_admin"] = False
            st.rerun()

        st.divider()

        # Theme selector (6 attractive themes!)
        st.subheader("🎨 Theme")
        theme_options = list(THEMES.keys())
        theme = st.selectbox(
            "Select Theme",
            theme_options,
            index=theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else 1,
            label_visibility="collapsed"
        )
        st.session_state.theme = theme
        
        # Theme preview hint
        theme_previews = {
            "Professional Dark": "🌙 High contrast • Dark mode",
            "Nature Green": "🌿 Earthy & Calming",
            "Ocean Blue": "🌊 Cool & Professional",
            "Sunset Warm": "🌅 Vibrant & Energetic",
            "Monochrome": "⚪ Minimalist Elegant",
            "Purple Royal": "👑 Premium Royal Look",
        }
        st.caption(theme_previews.get(theme, ""))

        # Background image
        st.caption("Background Image (optional)")
        bg_url  = st.text_input("Image URL", placeholder="https://…", key="bg_url")
        bg_file = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"], key="bg_file")

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
    
    # Role-based welcome message
    if st.session_state.is_admin:
        st.caption(f"👑 Welcome back, Admin! Full access granted.")
    else:
        st.caption(f"👋 Welcome, {st.session_state.username}! You have view-only access.")
    
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}  |  "
               f"Auto-refreshing every {refresh_interval}s")

    # ── TABS (Historical tab hidden for guests) ──
    if st.session_state.is_admin:
        tab_live, tab_hist = st.tabs(["📡 Live Dashboard", "📅 Historical Data"])
    else:
        tab_live = st.tabs(["📡 Live Dashboard"])[0]
        tab_hist = None

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
       # Apply filters
    filtered = new_data[
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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(l=10, r=20, t=10, b=10),
                legend_title="Status",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("♻️ Waste Type Composition")
            pie_data = hist_filtered.groupby("waste_type")["waste_kg"].sum().reset_index()
            fig_pie = px.pie(
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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
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
        st_folium(build_map(scores, st.session_state.theme), width=None, height=420, returned_objects=[])

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
    # TAB 2: HISTORICAL DATA (Admin only)
    # ─────────────────────────────────────────
    if st.session_state.is_admin:
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

