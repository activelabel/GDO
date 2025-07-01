import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
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

# Load single dataset
data = load_data("Market_1_shipment_dataset.csv")

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
st.sidebar.header("Filters")

# Market Label filter
if st.sidebar.checkbox("Select all Market Labels", value=True):
    sel_label = list(data["Market Label"].unique())
else:
    sel_label = st.sidebar.multiselect(
        "Market Label",
        options=data["Market Label"].unique(),
        default=list(data["Market Label"].unique())
    )

# Date filter
date_range = st.sidebar.date_input(
    "Period",
    [data["reading_timestamp"].dt.date.min(), data["reading_timestamp"].dt.date.max()]
)

# Product filter
if st.sidebar.checkbox("Select all Products", value=True):
    sel_prod = list(data["product"].unique())
else:
    sel_prod = st.sidebar.multiselect(
        "Product",
        options=data["product"].unique(),
        default=list(data["product"].unique())
    )

# Operator filter
if st.sidebar.checkbox("Select all Operators", value=True):
    sel_op = list(data["operator"].unique())
else:
    sel_op = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique())
    )

# Location Info filter
if st.sidebar.checkbox("Select all Locations", value=True):
    sel_loc = list(data["Location Info"].unique())
else:
    sel_loc = st.sidebar.multiselect(
        "Location Info",
        options=data["Location Info"].unique(),
        default=list(data["Location Info"].unique())
    )

# Apply filters
filtered = data[
    (data["Market Label"].isin(sel_label)) &
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (data["product"].isin(sel_prod)) &
    (data["operator"].isin(sel_op)) &
    (data["Location Info"].isin(sel_loc))
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
metrics = st.columns(5)
if not filtered.empty:
    comp = filtered["in_range"].mean() * 100
    inc = filtered["out_of_range"].mean() * 100
    tot = len(filtered)
    waste = filtered.loc[filtered["out_of_range"], "shipment_cost_eur"].sum()
    saved = (0.05 - 0.01) * tot * filtered["unit_co2_emitted"].mean()
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
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert to view details._")
alerts = filtered[filtered["out_of_range"]].sort_values("reading_timestamp", ascending=False)
if alerts.empty:
    st.success("No alerts.")
else:
    disp = alerts[["shipment_id","reading_timestamp","operator","product","severity","Market Label"]].copy()
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
    "shipment_id","reading_timestamp","operator","product",
    "actual_temperature","threshold_min_temperature","threshold_max_temperature",
    "in_range","out_of_range","shipment_cost_eur","unit_co2_emitted","Market Label","Location Info"
]
all_ship = filtered[cols].copy()
all_ship.columns = [
    "Shipment ID","Timestamp","Operator","Product",
    "Actual Temp (°C)","Min Temp","Max Temp",
    "In Range","Out of Range","Cost (€)","CO₂ Emitted (kg)","Market Label","Location"
]
st.dataframe(all_ship.sort_values("Timestamp", ascending=False), use_container_width=True)
