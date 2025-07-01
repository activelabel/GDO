# ------------------------------------------------
# IMPORT
# ------------------------------------------------
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

# ------------------------------------------------
# MARKET SELECTION
# ------------------------------------------------
st.sidebar.header("Market Selection by City")
market_files = {
    "Florence": "Market_1_shipments_dataset.csv",
    "Rome": "Market_2_shipments_dataset.csv"
}
selected_cities = st.sidebar.multiselect(
    "Choose cities to include", options=list(market_files.keys()),
    default=list(market_files.keys())
)

# Load and annotate
frames = []
for city in selected_cities:
    file = market_files[city]
    if not os.path.exists(file):
        st.sidebar.error(f"File not found: {file}")
        continue
    df = load_data(file)
    df["Market City"] = city
    # build location info
    df["Location Info"] = df["latitude"].astype(str) + ", " + df["longitude"].astype(str) + f" - {city}"
    frames.append(df)

if not frames:
    st.error("No data loaded. Please check your Market files.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

if st.sidebar.checkbox("Select all Products", value=True):
    sel_prod = list(data["product"].unique())
else:
    sel_prod = st.sidebar.multiselect(
        "Product", options=data["product"].unique(), default=list(data["product"].unique())
    )

if st.sidebar.checkbox("Select all Operators", value=True):
    sel_op = list(data["operator"].unique())
else:
    sel_op = st.sidebar.multiselect(
        "Operator", options=data["operator"].unique(), default=list(data["operator"].unique())
    )

if st.sidebar.checkbox("Select all Locations", value=True):
    sel_loc = list(data["Location Info"].unique())
else:
    sel_loc = st.sidebar.multiselect(
        "Location", options=data["Location Info"].unique(), default=list(data["Location Info"].unique())
    )

# apply filters
filtered = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (data["product"].isin(sel_prod)) &
    (data["operator"].isin(sel_op)) &
    (data["Location Info"].isin(sel_loc))
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)
if not filtered.empty:
    compliance = filtered["in_range"].mean() * 100
    incidents = filtered["out_of_range"].mean() * 100
    total = len(filtered)
    waste = filtered.loc[filtered["out_of_range"], "shipment_cost_eur"].sum()
    saved = (0.05 - 0.01) * total * filtered["unit_co2_emitted"].mean()
else:
    compliance = incidents = total = waste = saved = 0
col1.metric("% Compliant", f"{compliance:.1f}%")
col2.metric("% Incidents", f"{incidents:.1f}%")
col3.metric("Total Shipments", f"{total}")
col4.metric("Waste Cost (€)", f"{waste:.2f}")
col5.metric("CO₂ Saved (kg)", f"{saved:.1f}")

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
    alert_display = alerts[["shipment_id","reading_timestamp","operator","product","severity","Market City"]].copy()
    alert_display.insert(0, "Select", False)
    st.data_editor(alert_display, hide_index=True, use_container_width=True,
                   column_config={"Select": st.column_config.CheckboxColumn(required=True)}, key="alerts")

# ------------------------------------------------
# ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments list._")
cols = [
    "shipment_id","reading_timestamp","operator","product",
    "actual_temperature","threshold_min_temperature","threshold_max_temperature",
    "in_range","out_of_range","shipment_cost_eur","unit_co2_emitted","Market City","Location Info"
]
all_ship = filtered[cols].copy()
all_ship.columns = [
    "Shipment ID","Timestamp","Operator","Product",
    "Actual Temp (°C)","Min Temp","Max Temp",
    "In Range","Out of Range","Cost (€)","CO2 (kg)","City","Location"
]
st.dataframe(all_ship.sort_values("Timestamp", ascending=False), use_container_width=True)
