# ------------------------------------------------
# IMPORT
# ------------------------------------------------
import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from openai import OpenAI

# OPENAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Active Label Dashboard")

# ------------------------------------------------
# LOGO & TITLE
# ------------------------------------------------
logo = Image.open("assets/ActiveLabel_MARCHIO.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, use_container_width=True)
with col2:
    st.title("Active Label GDO_2")

# ------------------------------------------------
# DATA
# ------------------------------------------------
@st.cache_data
def load_data(path: str):
    columns = [
        'operator', 'device', 'reading_timestamp', 'exposure', 'actual_temperature',
        'threshold_min_temperature', 'threshold_max_temperature', 'shipment_id',
        'shipment_datetime', 'latitude', 'longitude'
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
        parse_dates=["reading_timestamp", "shipment_datetime"],
        dayfirst=True
    )
    return df

data = load_data("Dati_Lettura.txt")

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Filtro Device
if st.sidebar.checkbox("Select all Devices", value=True):
    selected_devices = list(data["device"].unique())
else:
    selected_devices = st.sidebar.multiselect(
        "Device",
        options=data["device"].unique(),
        default=list(data["device"].unique())
    )

# Filtro Operatori
if st.sidebar.checkbox("Select all Operators", value=True):
    selected_operators = list(data["operator"].unique())
else:
    selected_operators = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique())
    )

# Filtro Città
if st.sidebar.checkbox("Select all Cities", value=True):
    selected_cities = list(data["latitude"].astype(str) + ", " + data["longitude"].astype(str)).unique()
else:
    selected_cities = st.sidebar.multiselect(
        "City",
        options=data["latitude"].astype(str) + ", " + data["longitude"].astype(str),
        default=list(data["latitude"].astype(str) + ", " + data["longitude"].astype(str))
    )

# Filtraggio dati
data["city"] = data["latitude"].astype(str) + ", " + data["longitude"].astype(str)
filtered = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (data["device"].isin(selected_devices)) &
    (data["operator"].isin(selected_operators)) &
    (data["city"].isin(selected_cities))
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

compliance_pct = 100 - (filtered["actual_temperature"] > filtered["threshold_max_temperature"]).mean() * 100
incident_pct = 100 - compliance_pct
total_shipments = len(filtered)
cost_out_range = 0  # non presente nel file
co2_saved = 0       # non presente nel file

col1.metric("% Compliant Shipments", f"{compliance_pct:.1f}%")
col2.metric("% Shipments with Incidents", f"{incident_pct:.1f}%")
col3.metric("📦 Total Shipments", f"{total_shipments}")
col4.metric("Total Waste Cost (€)", f"{cost_out_range:.2f}")
col5.metric("🌱 CO₂ Saved (kg)", f"{co2_saved:.1f}")

# ------------------------------------------------
# 📌 OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert from the table below to view further details._")

# Definiamo alert come temperature fuori soglia
alert_df = filtered[
    (filtered["actual_temperature"] > filtered["threshold_max_temperature"]) |
    (filtered["actual_temperature"] < filtered["threshold_min_temperature"])
].sort_values('reading_timestamp', ascending=False)

if alert_df.empty:
    st.success("✅ No alerts to show.")
else:
    selection_alert_df = alert_df[[
        "shipment_id", "reading_timestamp", "operator", "device", "city", "latitude", "longitude"
    ]].copy()
    selection_alert_df.insert(0, "Select", False)

    edited_alert_df = st.data_editor(
        selection_alert_df.drop(columns=["latitude", "longitude"]),
        hide_index=True,
        use_container_width=True,
        height=300,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        key="alert_selector"
    )

# ------------------------------------------------
# 📋 ALL SHIPMENTS (aggiunto)
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered results, including both in-range and out-of-range shipments._")

full_view_df = filtered[[
    "shipment_id", "reading_timestamp", "operator", "device",
    "actual_temperature", "threshold_min_temperature", "threshold_max_temperature", "exposure", "city"
]].copy()

full_view_df.columns = [
    "Shipment ID", "Timestamp", "Operator", "Device",
    "Temperature (°C)", "Min Temp", "Max Temp
