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
    df["city"] = df["latitude"].astype(str) + ", " + df["longitude"].astype(str)
    return df

data = load_data("Dati_Lettura.txt")

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Product filter (uses `device` from the dataset)
if st.sidebar.checkbox("Select all Products", value=True):
    selected_products = list(data["device"].unique())
else:
    selected_products = st.sidebar.multiselect(
        "Product",
        options=data["device"].unique(),
        default=list(data["device"].unique())
    )

# Operator filter
if st.sidebar.checkbox("Select all Operators", value=True):
    selected_operators = list(data["operator"].unique())
else:
    selected_operators = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique())
    )

# City filter
if st.sidebar.checkbox("Select all Cities", value=True):
    selected_cities = list(data["city"].unique())
else:
    selected_cities = st.sidebar.multiselect(
        "City",
        options=data["city"].unique(),
        default=list(data["city"].unique())
    )

# Apply filters to data
filtered = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (data["device"].isin(selected_products)) &
    (data["operator"].isin(selected_operators)) &
    (data["city"].isin(selected_cities))
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
c1, c2, c3, c4, c5 = st.columns(5)

if not filtered.empty:
    total_shipments = len(filtered)
    out_of_range = (
        (filtered["actual_temperature"] > filtered["threshold_max_temperature"]) |
        (filtered["actual_temperature"] < filtered["threshold_min_temperature"])
    )
    compliance_pct = 100 - out_of_range.mean() * 100
    incident_pct = 100 - compliance_pct
else:
    total_shipments = 0
    compliance_pct = 0
    incident_pct = 0

c1.metric("% Compliant Shipments", f"{compliance_pct:.1f}%")
c2.metric("% Shipments with Incidents", f"{incident_pct:.1f}%")
c3.metric("📦 Total Shipments", f"{total_shipments}")
c4.metric("Total Waste Cost (€)", "N/A")
c5.metric("🌱 CO₂ Saved (kg)", "N/A")

# ------------------------------------------------
# 📌 OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert from the table below to view further details._")

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
# 📋 ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered results, including both in-range and out-of-range shipments._")

full_view_df = filtered[[
    "shipment_id", "reading_timestamp", "operator", "device",
    "actual_temperature", "threshold_min_temperature", "threshold_max_temperature", "exposure", "city"
]].copy()

full_view_df.columns = [
    "Shipment ID", "Timestamp", "Operator", "Product",
    "Temperature (°C)", "Min Temp", "Max Temp", "Exposure", "City"
]

st.dataframe(full_view_df.sort_values("Timestamp", ascending=False), use_container_width=True)
