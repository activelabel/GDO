import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
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
# DATA LOADING
# ------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["reading_timestamp"],
        dayfirst=True
    )
    return df

# Load dataset
data = load_data("Market_1_shipment_dataset.csv")

# ------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------
# Add ±5% random noise to actual_temperature
np.random.seed(42)
if "actual_temperature" in data.columns:
    noise = np.random.uniform(-0.05, 0.05, size=len(data))
    data["actual_temperature"] = data["actual_temperature"] * (1 + noise)

# Compute Time Lost (h): exposure * 24
if "exposure" in data.columns:
    data["Time Lost (h)"] = data["exposure"] * 24.0
else:
    data["Time Lost (h)"] = 0

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
st.sidebar.header("Filters")
# Market Label filter
if st.sidebar.checkbox("Select all Market Labels", value=True):
    sel_label = data["Market Label"].unique().tolist()
else:
    sel_label = st.sidebar.multiselect(
        "Market Label",
        options=data["Market Label"].unique(),
        default=data["Market Label"].unique()
    )
# Date filter
date_range = st.sidebar.date_input(
    "Period",
    [data["reading_timestamp"].dt.date.min(), data["reading_timestamp"].dt.date.max()]
)
# Product filter
if st.sidebar.checkbox("Select all Products", value=True):
    sel_prod = data["product"].unique().tolist()
else:
    sel_prod = st.sidebar.multiselect(
        "Product",
        options=data["product"].unique(),
        default=data["product"].unique()
    )
# Operator filter
if st.sidebar.checkbox("Select all Operators", value=True):
    sel_op = data["operator"].unique().tolist()
else:
    sel_op = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=data["operator"].unique()
    )
# Apply filters
filtered = data[
    data["Market Label"].isin(sel_label) &
    data["reading_timestamp"].dt.date.between(date_range[0], date_range[1]) &
    data["product"].isin(sel_prod) &
    data["operator"].isin(sel_op)
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
metrics = st.columns(5)
# Only consider records with exposure != 0
alert_data = filtered[filtered["exposure"] != 0]
if not alert_data.empty:
    comp = alert_data["in_range"].mean() * 100
    inc = alert_data["out_of_range"].mean() * 100
    tot = len(alert_data)
    waste = alert_data.loc[alert_data["out_of_range"], "shipment_cost_eur"].sum()
    saved = (0.05 - 0.01) * tot * alert_data["unit_co2_emitted"].mean()
else:
    comp = inc = tot = waste = saved = 0
metrics[0].metric("% Compliant", f"{comp:.1f}%")
metrics[1].metric("% Incidents", f"{inc:.1f}%")
metrics[2].metric("Total Shipments", f"{tot}")
metrics[3].metric("Waste Cost (€)", f"{waste:.2f}")
metrics[4].metric("CO₂ Saved (kg)", f"{saved:.1f}")

# ------------------------------------------------
# OPERATIONAL CONTROL
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments list._")
# Columns in desired order, with Market Label first
cols = [
    "Market Label",
    "shipment_id",
    "reading_timestamp",
    "operator",
    "product",
    "actual_temperature",
    "threshold_min_temperature",
    "threshold_max_temperature",
    "in_range",
    "out_of_range",
    "shipment_cost_eur",
    "unit_co2_emitted"
]
all_ship = filtered[cols].copy()
# Rename columns, ensure header for each
all_ship.columns = [
    "Market Label",
    "Shipment ID",
    "Timestamp",
    "Operator",
    "Product",
    "Actual Temp (°C)",
    "Min Temp",
    "Max Temp",
    "In Range",
    "Out of Range",
    "Cost (€)",
    "CO₂ Emitted (kg)"
]
st.dataframe(all_ship.sort_values("Timestamp", ascending=False), use_container_width=True)
