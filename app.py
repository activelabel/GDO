import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from openai import OpenAI
import folium
from streamlit_folium import folium_static

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
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["reading_timestamp"], dayfirst=True)
    # Add ±5% noise to actual_temperature
    np.random.seed(42)
    noise = np.random.uniform(-0.05, 0.05, size=len(df))
    df["actual_temperature"] *= (1 + noise)
    # Compute dynamic exposure: deviation beyond thresholds
    df["exposure"] = np.where(
        df["actual_temperature"] > df["threshold_max_temperature"],
        df["actual_temperature"] - df["threshold_max_temperature"],
        np.where(
            df["actual_temperature"] < df["threshold_min_temperature"],
            df["threshold_min_temperature"] - df["actual_temperature"],
            0.0
        )
    )
    # Compute Time Lost (h)
    df["Time Lost (h)"] = df["exposure"] * 24.0
    return df

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
# Incidents: exposure > 0
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

# ------------------------------------------------
# MARKET INCIDENT RATES
# ------------------------------------------------
st.subheader("📈 Market Incident Rates")
# Show a summary table for % incidents per market
summary = (
    filtered
    .groupby("Market Label")
    .apply(lambda df: (df["exposure"] > 0).mean() * 100)
    .reset_index(name="Pct Incidents")
)
st.dataframe(summary, use_container_width=True)

# ------------------------------------------------
# MAPPA MERCATI
# ------------------------------------------------
st.subheader("🗺️ Market Locations")
city_coords = {
    "Rome": (41.9028, 12.4964),
    "Florence": (43.7696, 11.2558),
    "Turin": (45.0703, 7.6869)
}
tile_map = folium.Map(location=[42.5, 12.5], zoom_start=5)
market_map = {}
for label in filtered["Market Label"].unique():
    city = label.split()[-1]
    market_map.setdefault(city, []).append(label)
for city, labels in market_map.items():
    if city not in city_coords:
        continue
    base_lat, base_lon = city_coords[city]
    n = len(labels)
    for i, label in enumerate(labels):
        angle = 2 * np.pi * i / max(n, 1)
        r = 0.05
        lat = base_lat + r * np.sin(angle)
        lon = base_lon + r * np.cos(angle)
        pct_label = float(summary.loc[summary["Market Label"] == label, "Pct Incidents"])
        if pct_label < 2:
            color = 'green'
        elif pct_label > 15:
            color = 'red'
        else:
            color = 'yellow'
        folium.Marker(
            location=[lat, lon],
            popup=f"{label}: {pct_label:.1f}% incidents",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(tile_map)
folium_static(tile_map)

# ------------------------------------------------
st.header("📝 Quick Report")
if st.button("Generate Report"):
    # Basic summary
    total_records = len(filtered)
    num_incidents = filtered[filtered["exposure"] > 0].shape[0]
    pct_incidents = num_incidents / total_records * 100 if total_records else 0.0
    pct_compliant = 100 - pct_incidents
    waste_cost = filtered.loc[(filtered["exposure"] > 0) & (filtered["out_of_range"]), "shipment_cost_eur"].sum()
    co2_saved = ((0.05 - 0.01) * num_incidents * filtered[filtered["exposure"] > 0]["unit_co2_emitted"].mean()) if num_incidents else 0.0
    # Top markets by incident rate
    market_rates = summary.sort_values("Pct Incidents", ascending=False)
    top_markets = market_rates.head(3)
    # Build report lines
    report_lines = []
    report_lines.append(f"Report for period: {date_range[0]} to {date_range[1]}")
    report_lines.append(f"Total shipments: {total_records}")
    report_lines.append(f"Incidents: {num_incidents} ({pct_incidents:.1f}%), Compliant: {pct_compliant:.1f}%")
    report_lines.append(f"Total waste cost: €{waste_cost:.2f}")
    report_lines.append(f"Estimated CO₂ saved: {co2_saved:.1f} kg")
    report_lines.append("Top 3 markets by incident rate:")
    for _, row in top_markets.iterrows():
        report_lines.append(f"- {row['Market Label']}: {row['Pct Incidents']:.1f}% incidents")
    # Join lines with newline character
        # Join lines with newline character
    report_text = "\n".join(report_lines)
    # Display and download
    st.text_area("Report Preview", report_text, height=200)
    st.download_button("Download report.txt", report_text, file_name="market_report.txt")

    # =================================================
# === AI REPORT GENERATOR (REDESIGNED) ==========
# =================================================

import json

def _snapshot_stats(df: pd.DataFrame) -> dict:
    """Compute essential stats for AI report."""
    total = len(df)
    if total == 0:
        return {}
    num_inc = int((df["exposure"] > 0).sum())
    num_comp = total - num_inc
    compliance_pct = round(num_comp / total * 100, 1)
    incident_pct = round(num_inc / total * 100, 1)
    waste_cost = round(df.loc[(df["exposure"] > 0) & (df["out_of_range"]), "shipment_cost_eur"].sum(), 2)
    co2_saved = round((0.05 - 0.01) * num_inc * df["unit_co2_emitted"].mean(), 1)
    return {
        "total_shipments": total,
        "compliance_pct": float(compliance_pct),
        "incident_pct": float(incident_pct),
        "waste_cost_eur": float(waste_cost),
        "co2_saved": float(co2_saved),
    }

def _draft_report(
    df: pd.DataFrame,
    user_query: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> str:
    """Generate an AI-driven report based on filtered data and a user question."""
    stats = _snapshot_stats(df)
    # Build prompt
    prompt = (
        "You are a data analyst. Given the following dataset summary and a user question, provide a concise report.\n\n"
        f"Dataset summary stats: {json.dumps(stats)}\n\n"
        f"User question: {user_query}\n\n"
        "Provide insights, key anomalies, and actionable recommendations based on the data."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# -------------------- UI ------------------------
st.header("📝 AI-Powered Custom Report")
user_query = st.text_input("Enter your analysis question or insight request:", "")
col1, col2 = st.columns([1, 3])
with col1:
    gen_ai = st.button("Generate AI Report")
with col2:
    temp_ai = st.slider("Cre_
