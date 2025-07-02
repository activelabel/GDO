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
# Total records selected by filters
total_records = len(filtered)
# Number of incidents defined by exposure != 0
num_incidents = filtered[filtered["exposure"] != 0].shape[0]
# Percentages
if total_records > 0:
    inc_pct = num_incidents / total_records * 100
    comp_pct = 100 - inc_pct
else:
    comp_pct = inc_pct = 0
# Waste cost: sum shipment_cost_eur for exposure !=0 and out_of_range
waste = filtered.loc[(filtered["exposure"] != 0) & (filtered["out_of_range"]), "shipment_cost_eur"].sum()
# CO2 saved: assume 0.05-0.01 per incident unit_co2_emitted average
df_alerts = filtered[filtered["exposure"] != 0]
co2_saved = ((0.05 - 0.01) * num_incidents * df_alerts["unit_co2_emitted"].mean()) if num_incidents > 0 else 0

metrics[0].metric("% Compliant", f"{comp_pct:.1f}%")
metrics[1].metric("% Incidents", f"{inc_pct:.1f}%")
metrics[2].metric("Total Shipments", f"{total_records}")
metrics[3].metric("Waste Cost (€)", f"{waste:.2f}")
metrics[4].metric("CO₂ Saved (kg)", f"{co2_saved:.1f}")("CO₂ Saved (kg)", f"{co2_saved:.1f}")
