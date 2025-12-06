import streamlit as st
import plotly.express as px
import pandas as pd

# -----------------------------
# PAGE CONFIG (COMPACT LAYOUT)
# -----------------------------
st.set_page_config(
    page_title="Farmer Engagement Dashboard",
    layout="wide"
)

# Remove default padding (compact look)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# USER INFO (Name + ID)
# -----------------------------
name = "John Adebayo"
user_id = "FRM-2024-0847"

st.markdown(f"""
    <div style="text-align: center; margin-bottom: -20px;">
        <span style="background:#d1fae5; color:#065f46; padding:6px 14px; 
        border-radius: 12px; font-size: 13px;">
            Real-time Analytics
        </span>
        <h2 style="margin-bottom:-5px;">Farmer Engagement Dashboard</h2>
        <p style="color:gray; margin-top:0;">Track how farmers interact with the platform</p>

        <h4 style="margin-top:5px;">👤 {name} — <span style="color:#059669;">{user_id}</span></h4>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI SECTION (Compact 4 Cards)
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Farmers", "12,458", "+12%")
col2.metric("Diagnoses Today", "847", "+28%")
col3.metric("Active Sessions", "234", "-5%")
col4.metric("Page Views", "45.2K", "+18%")

# -----------------------------
# WEEKLY ACTIVITY (Compact area chart)
# -----------------------------
weekly_data = pd.DataFrame({
    "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "Diagnoses": [120,180,230,290,310,260,150],
    "Visitors":  [350,420,500,630,700,560,400]
})

fig_area = px.area(
    weekly_data,
    x="Day",
    y=["Diagnoses", "Visitors"],
    title="Weekly Activity",
)
fig_area.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))

# -----------------------------
# DISEASE DISTRIBUTION PIE
# -----------------------------
disease_data = pd.DataFrame({
    "Disease": ["Leaf Blight","Powdery Mildew","Root Rot","Rust","Others"],
    "Value": [35,25,20,12,8]
})

fig_pie = px.pie(
    disease_data,
    names="Disease",
    values="Value",
    hole=0.5,
    title="Disease Distribution"
)
fig_pie.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))

# -----------------------------
# LAYOUT: CHARTS
# -----------------------------
left, right = st.columns([2,1])
left.plotly_chart(fig_area, use_container_width=True)
right.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# FARMERS BY REGION (Small Bar Chart)
# -----------------------------
region_data = pd.DataFrame({
    "Region": ["North", "South", "East", "West", "Central"],
    "Farmers": [900, 760, 850, 1200, 680]
})

fig_bar = px.bar(
    region_data,
    x="Farmers",
    y="Region",
    orientation="h",
    title="Farmers by Region"
)
fig_bar.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))

# -----------------------------
# RECENT ACTIVITY (Compact List)
# -----------------------------
recent_activity = [
    ("Farmer #8234", "Wheat disease diagnosis", "2 min ago", "Punjab"),
    ("Farmer #5621", "Viewed market prices", "5 min ago", "Haryana"),
    ("Farmer #9012", "Rice leaf analysis", "8 min ago", "West Bengal"),
    ("Farmer #3456", "New registration", "12 min ago", "Maharashtra"),
    ("Farmer #7890", "Cotton pest detection", "15 min ago", "Gujarat")
]

# -----------------------------
# LAYOUT: BOTTOM SECTION
# -----------------------------
b1, b2 = st.columns([1.2, 1])

b1.plotly_chart(fig_bar, use_container_width=True)

with b2:
    st.markdown("### Recent Activity")
    for farmer, activity, time, location in recent_activity:
        st.markdown(f"""
            <div style="
                background:#f9fafb; 
                padding:10px 12px; 
                border-radius:10px; 
                margin-bottom:8px;
                border:1px solid #e5e7eb;
            ">
                <b>{farmer}</b> — {activity}  
                <div style="color:gray; font-size:12px;">
                    {time} • {location}
                </div>
            </div>
        """, unsafe_allow_html=True)
