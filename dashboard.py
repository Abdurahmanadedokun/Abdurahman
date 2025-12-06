import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium

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
# Streamlit Dashboard
# -------------------------
st.set_page_config(page_title="SmartFarm Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🌱 SmartFarm: Crop & Disease Monitoring 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Interactive dashboard to monitor crops, soil, and disease.</p>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filters")
crop_filter = st.sidebar.multiselect("Select Crop", options=df['Crop'].unique(), default=df['Crop'].unique())
location_filter = st.sidebar.multiselect("Select Location", options=df['Location'].unique(), default=df['Location'].unique())
disease_filter = st.sidebar.multiselect("Select Disease", options=df['Disease Detected'].unique(), default=df['Disease Detected'].unique())

# Filter dataframe
df_filtered = df[(df['Crop'].isin(crop_filter)) &
                 (df['Location'].isin(location_filter)) &
                 (df['Disease Detected'].isin(disease_filter))]

# -------------------------
# KPI Cards
# -------------------------
st.markdown("### Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Farms", df_filtered['Farmer ID'].nunique())
col2.metric("Total Crops", df_filtered['Crop'].nunique())
col3.metric("Farms with Disease", df_filtered[df_filtered['Disease Detected'] != 'None']['Farmer ID'].nunique())
col4.metric("Avg Soil Moisture (%)", round(df_filtered['Soil Moisture (%)'].mean(),2))

col5, col6 = st.columns(2)
col5.metric("Avg Temperature (°C)", round(df_filtered['Temperature (°C)'].mean(),2))
col6.metric("Total Rainfall (mm)", round(df_filtered['Rainfall (mm)'].sum(),2))

# -------------------------
# Charts
# -------------------------
fig_moisture = px.line(df_filtered, x='Date', y='Soil Moisture (%)', color='Crop', markers=True, template='plotly_white', title="Soil Moisture Over Time")
st.plotly_chart(fig_moisture, use_container_width=True)

fig_temp = px.line(df_filtered, x='Date', y='Temperature (°C)', color='Crop', markers=True, template='plotly_white', title="Temperature Over Time")
st.plotly_chart(fig_temp, use_container_width=True)

fig_rain = px.line(df_filtered, x='Date', y='Rainfall (mm)', color='Crop', markers=True, template='plotly_white', title="Rainfall Over Time")
st.plotly_chart(fig_rain, use_container_width=True)

df_disease = df_filtered[df_filtered['Disease Detected'] != 'None']
fig_disease = px.bar(df_disease, x='Crop', y='Disease Severity', color='Disease Detected', text='Disease Severity', template='plotly_white', title="Disease Severity per Crop")
fig_disease.update_traces(textposition='outside')
st.plotly_chart(fig_disease, use_container_width=True)

# -------------------------
# Map
# -------------------------
m = folium.Map(location=[9.0,7.0], zoom_start=6, tiles='CartoDB positron')
for i, row in df_filtered.iterrows():
    lat = 7.0 + i*0.5  # dummy latitude offset
    lon = 9.0 + i*0.5  # dummy longitude offset
    color = 'green' if row['Disease Severity']==0 else 'yellow' if row['Disease Severity']<=2 else 'red'
    folium.CircleMarker(
        location=[lat, lon], radius=8, color=color, fill=True, fill_opacity=0.7,
        popup=f"<b>Farmer:</b> {row['Farmer Name']}<br><b>Crop:</b> {row['Crop']}<br><b>Disease:</b> {row['Disease Detected']}<br><b>Severity:</b> {row['Disease Severity']}<br><b>Action:</b> {row['Recommended Action']}",
        tooltip=row['Farmer Name']
    ).add_to(m)
st_folium(m, width=700, height=450)

# -------------------------
# Data Table
# -------------------------
st.dataframe(df_filtered[['Farmer Name','Location','Crop','Soil Moisture (%)','Temperature (°C)','Rainfall (mm)','Disease Detected','Disease Severity','Recommended Action']])
