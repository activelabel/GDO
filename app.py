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
# Base coordinates for each city center
city_coords = {
    "Rome": (41.9028, 12.4964),
    "Florence": (43.7696, 11.2558),
    "Turin": (45.0703, 7.6869)
}
# Initialize folium map centered in Italy
tile_map = folium.Map(location=[42.5, 12.5], zoom_start=5)
# Determine market labels for each city
market_map = {}
for label in filtered["Market Label"].unique():
    city = label.split()[-1]
    market_map.setdefault(city, []).append(label)
# Place a marker for each market with color reflecting % Incidents
for city, labels in market_map.items():
    if city not in city_coords:
        continue
    base_lat, base_lon = city_coords[city]
    n = len(labels)
    for i, label in enumerate(labels):
        angle = 2 * np.pi * i / max(n, 1)
        r = 0.05  # offset radius in degrees (~5 km)
        lat = base_lat + r * np.sin(angle)
        lon = base_lon + r * np.cos(angle)
        # Look up % incidents in summary
        pct_label = float(summary.loc[summary["Market Label"] == label, "Pct Incidents"])
        # Determine marker color
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
# Render the map
folium_static(tile_map)

# =================================================
# === AI REPORT GENERATOR  (NEW SECTION) ==========
# =================================================

def _snapshot_stats(df: pd.DataFrame) -> dict:
    """Essential statistics for model input (reduces token cost)."""
    if df.empty:
        return {}
    return {
        "compliance_pct": float(round(df["in_range"].mean() * 100, 1)),
        "incident_pct": float(round(df["out_of_range"].mean() * 100, 1)),
        "waste_cost_eur": float(round(
            df.loc[df["out_of_range"], "shipment_cost_eur"].sum(), 2
        )),
        "co2_saved": float(round(
            (0.05 - 0.01) * len(df) * df["unit_co2_emitted"].mean(), 1
        )),
    }


def _draft_report(
    df: pd.DataFrame,
    custom_task: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> str:
    """Constructs prompt and requests AI-generated text."""
    # Prepare JSON-safe data sample
    sample_df = df.sample(min(len(df), 50), random_state=42).copy()
    for col in sample_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].apply(lambda x: None if pd.isna(x) else float(x))
        else:
            sample_df[col] = sample_df[col].astype(str).fillna("N/A")
    sample_json = sample_df.to_dict(orient="records")
    # Build prompt with explicit newline escapes
prompt = (
    "You are a data analyst. Write a concise executive summary report in English (max 300 words), "
    "highlighting KPIs, anomalies, and recommendations.\n\n"
    f"Summary statistics: {json.dumps(_snapshot_stats(df))}\n\n"
    f"Sample data rows: {json.dumps(sample_json)[:4000]}\n\n"
    f"Additional request: {custom_task}"
)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# -------------------- UI ------------------------
col_analysis, col_download = st.columns(2)

with col_analysis:
    st.header("📝 AI Analysis")
    with st.expander("Generate a mini-report for filtered data"):
        task_txt = st.text_area(
            "Additional instructions (optional)",
            "Example: Highlight operators with the most incidents and suggest actions.",
        )
        left, right = st.columns([1, 4])
        with left:
            gen_btn = st.button("Generate report")
        with right:
            temp_val = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)
        if gen_btn:
            if filtered.empty:
                st.error("No filtered data – adjust the filters and try again.")
            elif not client.api_key:
                st.error("OPENAI_API_KEY not configured.")
            else:
                with st.spinner("AI analysis in progress..."):
                    report_txt = _draft_report(filtered, task_txt, temperature=temp_val)
                st.success("Report is ready:")
                st.markdown("### Preview")
                st.write(report_txt)
                st.download_button(
                    "Download report.txt", report_txt, file_name="mini_report.txt"
                )

with col_download:
    st.header("📥 Download Data")
    st.download_button(
        "Export CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="report_active_label.csv",
    )
