import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from openai import OpenAI

# OPENAI
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

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
# DATA LOADING & PREPROCESSING
# ------------------------------------------------
@st.cache_data
"
"def load_data(path: str) -> pd.DataFrame:
"
"    df = pd.read_csv(path, parse_dates=["reading_timestamp"], dayfirst=True)
"
"    # Add ±5% noise to actual_temperature
"
"    np.random.seed(42)
"
"    noise = np.random.uniform(-0.05, 0.05, size=len(df))
"
"    df["actual_temperature"] *= (1 + noise)
"
"    # Compute dynamic exposure: deviation beyond thresholds
"
"    df["exposure"] = np.where(
"
"        df["actual_temperature"] > df["threshold_max_temperature"],
"
"        df["actual_temperature"] - df["threshold_max_temperature"],
"
"        np.where(
"
"            df["actual_temperature"] < df["threshold_min_temperature"],
"
"            df["threshold_min_temperature"] - df["actual_temperature"],
"
"            0.0
"
"        )
"
"    )
"
"    # Compute Time Lost (h)
"
"    df["Time Lost (h)"] = df["exposure"] * 24.0
"
"    return df

# Load dataset
data = load_data("Market_1_shipment_dataset.csv")

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
st.sidebar.header("Filters")
# Market Label
if st.sidebar.checkbox("Select all Market Labels", True):
    sel_label = data["Market Label"].unique().tolist()
else:
    sel_label = st.sidebar.multiselect(
        "Market Label", data["Market Label"].unique(), default=data["Market Label"].unique()
    )
# Date Range
date_range = st.sidebar.date_input(
    "Period",
    [data["reading_timestamp"].dt.date.min(), data["reading_timestamp"].dt.date.max()]
)
# Product
if st.sidebar.checkbox("Select all Products", True):
    sel_prod = data["product"].unique().tolist()
else:
    sel_prod = st.sidebar.multiselect(
        "Product", data["product"].unique(), default=data["product"].unique()
    )
# Operator
if st.sidebar.checkbox("Select all Operators", True):
    sel_op = data["operator"].unique().tolist()
else:
    sel_op = st.sidebar.multiselect(
        "Operator", data["operator"].unique(), default=data["operator"].unique()
    )
# Apply filters
filtered = data[
    data["Market Label"].isin(sel_label)
    & data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])
    & data["product"].isin(sel_prod)
    & data["operator"].isin(sel_op)
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)
# Total filtered records
total_records = len(filtered)
# Incidents: exposure != 0
num_incidents = filtered[filtered["exposure"] > 0].shape[0]
# Percentages
pct_incidents = num_incidents / total_records * 100 if total_records else 0
pct_compliant = 100 - pct_incidents
# Waste cost: only on incidents out_of_range
waste_cost = filtered.loc[
    (filtered["exposure"] > 0) & (filtered["out_of_range"]),
    "shipment_cost_eur"
].sum()
# CO2 saved: per incident average
incident_rows = filtered[filtered["exposure"] > 0]
co2_saved = (
    (0.05 - 0.01) * num_incidents * incident_rows["unit_co2_emitted"].mean()
) if num_incidents else 0
col1.metric("% Compliant", f"{pct_compliant:.1f}%")
col2.metric("% Incidents", f"{pct_incidents:.1f}%")
col3.metric("Total Shipments", f"{total_records}")
col4.metric("Waste Cost (€)", f"{waste_cost:.2f}")
col5.metric("CO₂ Saved (kg)", f"{co2_saved:.1f}")

# ------------------------------------------------
# OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert to view details._")
alerts = filtered[(filtered["exposure"] > 0) & (filtered["out_of_range"])]
alerts = alerts.sort_values("reading_timestamp", ascending=False)
if alerts.empty:
    st.success("No alerts.")
else:
    disp = alerts[[
        "Market Label", "shipment_id", "reading_timestamp", "operator", "product", "severity", "Time Lost (h)", "exposure"
    ]].copy()
    disp = disp.rename(columns={"exposure": "Exposure (°C)"})
    disp.insert(0, "Select", False)
    st.data_editor(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        key="alerts"
    )

# ------------------------------------------------
# ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments list._")
cols = [
    "Market Label", "shipment_id", "reading_timestamp", "operator", "product", "actual_temperature",
    "threshold_min_temperature", "threshold_max_temperature", "in_range", "out_of_range", 
    "shipment_cost_eur", "unit_co2_emitted"
]
all_ship = filtered[cols].copy()
all_ship.columns = [
    "Market Label", "Shipment ID", "Timestamp", "Operator", "Product", "Actual Temp (°C)",
    "Min Temp", "Max Temp", "In Range", "Out of Range", "Cost (€)", "CO₂ Emitted (kg)"
]
st.dataframe(all_ship.sort_values("Timestamp", ascending=False), use_container_width=True)
