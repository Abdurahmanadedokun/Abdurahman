import streamlit as st
import plotly.express as px
import pandas as pd

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Farmer Engagement Dashboard",
    layout="wide"
)

# -----------------------
# CUSTOM COLOURFUL THEME
# -----------------------
st.markdown("""
<style>
    .big-header {
        text-align:center;
        padding:20px;
        border-radius:15px;
        background: linear-gradient(90deg, #34d399, #10b981);
        color:white;
        margin-bottom:15px;
    }

    .kpi-box {
        background:white;
        padding:15px;
        border-radius:15px;
        border:1px solid #e5e7eb;
        box-shadow:0 2px 6px rgba(0,0,0,0.05);
        text-align:center;
    }

    .section-box {
        background:white;
        padding:18px;
        border-radius:15px;
        border:1px solid #e5e7eb;
        box-shadow:0 2px 6px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# SAFE USER INFO (NO HTML)
# -----------------------
name = "John Adebayo"
user_id = "FRM-2024-0847"

st.markdown(
    f"<div class='big-header'><h2>Farmer Engagement Dashboard</h2>"
    f"<p>Welcome, <b>{name}</b> — ID: <b>{user_id}</b></p></div>",
    unsafe_allow_html=True
)

# -----------------------
# KPI CARDS (Colourful)
# -----------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric("Total Farmers", "12,458", "+12%")
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric("Diagnoses Today", "847", "+28%")
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric("Active Sessions", "234", "-5%")
    st.markdown("</div>", unsafe_allow_html=True)

with k4:
    st.markdown("<div class='kpi-box'>", unsafe_allow_html=True)
    st.metric("Page Views", "45.2K", "+18%")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# WEEKLY ACTIVITY CHART
# -----------------------
weekly_data = pd.DataFrame({
    "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "Diagnoses": [120,180,230,290,310,260,150],
    "Visitors":  [350,420,500,630,700,560,400]
})

fig_area = px.area(
    weekly_data,
    x="Day",
    y=["Diagnoses", "Visitors"],
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Weekly Activity"
)
fig_area.update_layout(height=300)

# -----------------------
# DISEASE PIE CHART
# -----------------------
disease_data = pd.DataFrame({
    "Disease": ["Leaf Blight","Powdery Mildew","Root Rot","Rust","Others"],
    "Value": [35,25,20,12,8]
})

fig_pie = px.pie(
    disease_data,
    names="Disease",
    values="Value",
    hole=0.5,
    color_discrete_sequence=px.colors.qualitative.Prism,
    title="Disease Distribution"
)
fig_pie.update_layout(height=300)

# -----------------------
# CHART LAYOUT
# -----------------------
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.plotly_chart(fig_area, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# REGION BAR + ACTIVITY LIST
# -----------------------
# -----------------------
# REGION BAR (Clean + Bold + Spacious)
# -----------------------

region_data = pd.DataFrame({
    "Region": ["North", "South", "East", "West", "Central"],
    "Farmers": [900, 760, 850, 1200, 680]
})

fig_bar = px.bar(
    region_data,
    x="Farmers",
    y="Region",
    orientation="h",
    color="Farmers",
    color_continuous_scale="Greens",
)

fig_bar.update_layout(
    title="<b>Farmers by Region</b>",
    height=330,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis_title="<b>Number of Farmers</b>",
    yaxis_title="<b>Region</b>",
    xaxis=dict(tickfont=dict(size=13)),
    yaxis=dict(tickfont=dict(size=13)),
)

# ADD MORE SPACE BETWEEN BARS
fig_bar.update_traces(marker=dict(line=dict(width=1.5, color="white")), width=0.55)

# -----------------------
# Recent Activity (Bold + Clean)
# -----------------------
recent_activity = [
    ("Farmer #8234", "Wheat disease diagnosis", "2 min ago", "Punjab"),
    ("Farmer #5621", "Viewed market prices", "5 min ago", "Haryana"),
    ("Farmer #9012", "Rice leaf analysis", "8 min ago", "West Bengal"),
    ("Farmer #3456", "New registration", "12 min ago", "Maharashtra"),
    ("Farmer #7890", "Cotton pest detection", "15 min ago", "Gujarat")
]

b1, b2 = st.columns([1.4, 1])

with b1:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### <b>Recent Activity</b>", unsafe_allow_html=True)

    for farmer, activity, time, location in recent_activity:
        st.markdown(
            f"""
            <div style="
                background:#f0fdf4; 
                padding:12px; 
                border-radius:12px; 
                margin-bottom:10px;
                border:1px solid #d1fae5;
                font-size:14px;
            ">
                <b>{farmer}</b><br>
                {activity}<br>
                <span style="color:gray; font-size:12px;">⏱ {time} • 📍 {location}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

