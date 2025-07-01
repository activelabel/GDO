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
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["reading_timestamp"],
        dayfirst=True
    )
    df["city"] = df["city"].astype(str)
    # annotate source file
    df["market"] = Path(path).stem
    return df

# Sidebar: select market datasets
st.sidebar.header("Market Selection")
market_options = {
    "Market 1": "Market_1_shipments_dataset.csv",
    "Market 2": "Market_2_shipments_dataset.csv"
}
selected_markets = st.sidebar.multiselect(
    "Choose Market data to include", options=list(market_options.keys()),
    default=list(market_options.keys())
)

# Load and concatenate selected markets
dfs = []
for market in selected_markets:
    file = market_options[market]
    if os.path.exists(file):
        dfs.append(load_data(file))
    else:
        st.sidebar.error(f"File not found: {file}")

if not dfs:
    st.error("No market data selected or files missing.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Product filter
if st.sidebar.checkbox("Select all Products", value=True, key="prod_all"):
    selected_products = list(data["product"].unique())
else:
    selected_products = st.sidebar.multiselect(
        "Product",
        options=data["product"].unique(),
        default=list(data["product"].unique()),
        key="prod_sel"
    )

# Operator filter
if st.sidebar.checkbox("Select all Operators", value=True, key="op_all"):
    selected_operators = list(data["operator"].unique())
else:
    selected_operators = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique()),
        key="op_sel"
    )

# City filter
if st.sidebar.checkbox("Select all Cities", value=True, key="city_all"):
    selected_cities = list(data["city"].unique())
else:
    selected_cities = st.sidebar.multiselect(
        "City",
        options=data["city"].unique(),
        default=list(data["city"].unique()),
        key="city_sel"
    )

# Apply filters
data["date"] = data["reading_timestamp"].dt.date
filtered = data[
    data["date"].between(date_range[0], date_range[1]) &
    data["product"].isin(selected_products) &
    data["operator"].isin(selected_operators) &
    data["city"].isin(selected_cities)
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
c1, c2, c3, c4, c5 = st.columns(5)

if not filtered.empty:
    total_shipments = len(filtered)
    compliance_pct = filtered["in_range"].mean() * 100
    incident_pct = filtered["out_of_range"].mean() * 100
    cost_out_range = filtered.loc[filtered["out_of_range"], "shipment_cost_eur"].sum()
    co2_saved = ((0.05 - 0.01) * len(filtered) * filtered["unit_co2_emitted"].mean())
else:
    total_shipments = compliance_pct = incident_pct = cost_out_range = co2_saved = 0

c1.metric("% Compliant Shipments", f"{compliance_pct:.1f}%")
c2.metric("% Shipments with Incidents", f"{incident_pct:.1f}%")
c3.metric("📦 Total Shipments", f"{total_shipments}")
c4.metric("Total Waste Cost (€)", f"{cost_out_range:.2f}")
c5.metric("🌱 CO₂ Saved (kg)", f"{co2_saved:.1f}")

# ------------------------------------------------
# 📌 OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert from the table below to view further details._")

alert_df = filtered[filtered['out_of_range']].sort_values('reading_timestamp', ascending=False)

if alert_df.empty:
    st.success("✅ No alerts to show.")
else:
    sel = alert_df[[
        "shipment_id","reading_timestamp","operator","product",
        "severity","city","latitude","longitude"
    ]].copy()
    sel.insert(0, "Select", False)
    st.data_editor(
        sel.drop(columns=["latitude","longitude"]),
        hide_index=True, use_container_width=True, height=300,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        key="alert_selector"
    )

# ------------------------------------------------
# 📦 ALL PRODUCTS
# ------------------------------------------------
st.subheader("📦 All Products")
st.markdown("_All products matching current filters with full details._")
prod_df = filtered[[
    "shipment_id","reading_timestamp","operator","product",
        "severity","city","latitude","longitude"
]].drop_duplicates().copy()
prod_df.columns = [
    "Shipment ID","Timestamp","Operator","Product",
    "Severity","City","latitude","longitude"
]
st.dataframe(
    prod_df.drop(columns=["latitude","longitude"])\
        .sort_values("Timestamp",ascending=False),
    use_container_width=True
)

# ------------------------------------------------
# 📋 ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered results, including both in-range and out-of-range shipments._")
full = filtered[[
    "shipment_id","reading_timestamp","operator","product",
    "actual_temperature","threshold_min_temperature","threshold_max_temperature",
    "in_range","out_of_range","shipment_cost_eur","unit_co2_emitted","city"
]].copy()
full.columns = [
    "Shipment ID","Timestamp","Operator","Product",
    "Actual Temp (°C)","Min Temp","Max Temp",
    "In Range","Out of Range","Cost (€)","CO2 Emitted (kg)","City"
]
st.dataframe(full.sort_values("Timestamp",ascending=False),use_container_width=True)
