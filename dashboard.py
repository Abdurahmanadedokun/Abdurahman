import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium

# -------------------------
# User Information
# -------------------------
USERNAME = "Abdurahman Adedokun"
USERID = "SF-001"

# -------------------------
# Sample Data
# -------------------------
data = {
    'Farmer ID': ['F001','F002','F003','F004','F005'],
    'Farmer Name': ['Ali','Bola','Chidi','Dami','Emeka'],
    'Location': ['Ife','Ibadan','Lagos','Kano','Abuja'],
    'Crop': ['Maize','Cassava','Rice','Maize','Cassava'],
    'Soil Moisture (%)': [45, 50, 38, 60, 55],
    'Temperature (°C)': [30, 32, 29, 33, 31],
    'Rainfall (mm)': [100, 120, 80, 150, 110],
    'Disease Detected': ['None','Fungal','None','Bacterial','None'],
    'Disease Severity': [0,2,0,3,0],
    'Recommended Action': ['N/A','Spray fungicide','N/A','Remove infected','N/A'],
    'Date': pd.to_datetime(['2025-12-01','2025-12-02','2025-12-03','2025-12-04','2025-12-05'])
}
df = pd.DataFrame(data)

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="SmartFarm Dashboard", layout="wide")

# -------------------------
# User Identity Box (Compact)
# -------------------------
st.markdown("""
<style>
.user-box {
    background-color: #E9F7EF;
    padding: 10px 16px;
    border-radius: 10px;
    border-left: 4px solid #0A7F2E;
    margin-bottom: 5px;
}
.user-text {
    font-size: 14px;
    color: #0A7F2E;
}
.user-bold {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='user-box'>
    <span class='user-text'>👤 <span class='user-bold'>User:</span> {USERNAME}  
    &nbsp;&nbsp; | &nbsp;&nbsp; 🔢 <span class='user-bold'>User ID:</span> {USERID}</span>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown("""
    <h3 style='text-align:center; color:#0A7F2E; margin-top:-10px;'>
        🌿 SMARTFARM COMPACT DASHBOARD 🌿
    </h3>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Filters + user info
# -------------------------
with st.sidebar:
    st.markdown("### 👤 User Info")
    st.info(f"**Name:** {USERNAME}\n\n**User ID:** {USERID}")

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    crop_filter = st.multiselect("Crop", df['Crop'].unique(), df['Crop'].unique())
    location_filter = st.multiselect("Location", df['Location'].unique(), df['Location'].unique())
    disease_filter = st.multiselect("Disease", df['Disease Detected'].unique(), df['Disease Detected'].unique())

df_filtered = df[
    (df['Crop'].isin(crop_filter)) &
    (df['Location'].isin(location_filter)) &
    (df['Disease Detected'].isin(disease_filter))
]

# -------------------------
# KPI Cards (Box Style)
# -------------------------
st.markdown("""
<style>
.kpi-box {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-value {
    font-size: 22px;
    font-weight: bold;
    color: #0A7F2E;
}
.kpi-label {
    font-size: 13px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='kpi-box'><div class='kpi-value'>{df_filtered['Farmer ID'].nunique()}</div><div class='kpi-label'>Total Farms</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='kpi-box'><div class='kpi-value'>{df_filtered[df_filtered['Disease Detected']!='None']['Farmer ID'].nunique()}</div><div class='kpi-label'>Farms w/ Disease</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='kpi-box'><div class='kpi-value'>{round(df_filtered['Soil Moisture (%)'].mean(),1)}</div><div class='kpi-label'>Avg Soil Moisture</div></div>", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["📊 Charts", "🗺️ Map", "📁 Table"])

# Charts Tab
with tabs[0]:
    st.subheader("📈 Crop Trend")
    metric = st.selectbox("Select Metric", ["Soil Moisture (%)", "Temperature (°C)", "Rainfall (mm)"])

    fig = px.line(df_filtered, x="Date", y=metric, color="Crop", markers=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Map Tab
with tabs[1]:
    st.subheader("🗺️ Farm Map (Compact)")
    m = folium.Map(location=[9.0,7.0], zoom_start=5, tiles="cartodbpositron")

    for i, row in df_filtered.iterrows():
        lat = 7 + i*0.4
        lon = 9 + i*0.4
        color = "green" if row["Disease Severity"] == 0 else "orange" if row["Disease Severity"] <= 2 else "red"

        folium.CircleMarker(
            [lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['Farmer Name']} – {row['Crop']}",
        ).add_to(m)

    st_folium(m, width=330, height=350)

# Table Tab
with tabs[2]:
    st.subheader("📁 Farm Data Table")
    st.dataframe(df_filtered, height=300)
