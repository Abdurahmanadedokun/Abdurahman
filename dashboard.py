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
st.markdown("<h2 style='text-align:center; color:green;'>🌱 SmartFarm Dashboard 🌱</h2>", unsafe_allow_html=True)

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    crop_filter = st.multiselect("Select Crop", df['Crop'].unique(), df['Crop'].unique())
    location_filter = st.multiselect("Select Location", df['Location'].unique(), df['Location'].unique())
    disease_filter = st.multiselect("Select Disease", df['Disease Detected'].unique(), df['Disease Detected'].unique())

# Filter Data
df_filtered = df[(df['Crop'].isin(crop_filter)) &
                 (df['Location'].isin(location_filter)) &
                 (df['Disease Detected'].isin(disease_filter))]

# -------------------------
# KPI Cards in compact layout
# -------------------------
kpi_cols = st.columns(3)
kpi_cols[0].metric("Total Farms", df_filtered['Farmer ID'].nunique())
kpi_cols[1].metric("Farms with Disease", df_filtered[df_filtered['Disease Detected'] != 'None']['Farmer ID'].nunique())
kpi_cols[2].metric("Avg Soil Moisture (%)", round(df_filtered['Soil Moisture (%)'].mean(),2))

kpi_cols2 = st.columns(3)
kpi_cols2[0].metric("Avg Temp (°C)", round(df_filtered['Temperature (°C)'].mean(),2))
kpi_cols2[1].metric("Total Rainfall (mm)", round(df_filtered['Rainfall (mm)'].sum(),2))
kpi_cols2[2].metric("Total Crops", df_filtered['Crop'].nunique())

# -------------------------
# Tabs for Charts, Map, and Table
# -------------------------
tabs = st.tabs(["📊 Charts", "🗺️ Map", "📄 Data Table"])

# Charts Tab
with tabs[0]:
    st.subheader("Crop Monitoring")
    fig1 = px.line(df_filtered, x='Date', y='Soil Moisture (%)', color='Crop', markers=True, template='plotly_white', title="Soil Moisture")
    fig2 = px.line(df_filtered, x='Date', y='Temperature (°C)', color='Crop', markers=True, template='plotly_white', title="Temperature")
    fig3 = px.line(df_filtered, x='Date', y='Rainfall (mm)', color='Crop', markers=True, template='plotly_white', title="Rainfall")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Disease severity chart
    df_disease = df_filtered[df_filtered['Disease Detected'] != 'None']
    if not df_disease.empty:
        fig4 = px.bar(df_disease, x='Crop', y='Disease Severity', color='Disease Detected',
                      text='Disease Severity', template='plotly_white', title="Disease Severity")
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

# Map Tab
with tabs[1]:
    st.subheader("Farm Locations")
    m = folium.Map(location=[9.0,7.0], zoom_start=6, tiles='CartoDB positron')
    for i, row in df_filtered.iterrows():
        lat = 7.0 + i*0.5
        lon = 9.0 + i*0.5
        color = 'green' if row['Disease Severity']==0 else 'yellow' if row['Disease Severity']<=2 else 'red'
        folium.CircleMarker(
            location=[lat, lon], radius=6, color=color, fill=True, fill_opacity=0.7,
            popup=f"<b>Farmer:</b> {row['Farmer Name']}<br><b>Crop:</b> {row['Crop']}<br><b>Disease:</b> {row['Disease Detected']}<br><b>Severity:</b> {row['Disease Severity']}<br><b>Action:</b> {row['Recommended Action']}",
            tooltip=row['Farmer Name']
        ).add_to(m)
    st_folium(m, width=350, height=400)  # smaller width for mobile

# Data Table Tab
with tabs[2]:
    st.subheader("Farm Data")
    st.dataframe(df_filtered[['Farmer Name','Location','Crop','Soil Moisture (%)','Temperature (°C)','Rainfall (mm)','Disease Detected','Disease Severity','Recommended Action']])
