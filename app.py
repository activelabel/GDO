# ------------------------------------------------
# IMPORT
# ------------------------------------------------
import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv                 #  <-- NEW
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])                              #  <-- NEW
# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Active Label Dashboard")
# ------------------------------------------------
# ENV & OPENAI (NEW)
# ------------------------------------------------
# ------------------------------------------------
# LOGO & TITLE
# ------------------------------------------------
logo = Image.open("assets/ActiveLabel_MARCHIO.png")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, use_container_width=True)
with col2:
    st.title("Active Label Dashboard")

# ------------------------------------------------
# DATA
# ------------------------------------------------
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path, parse_dates=["reading_timestamp"])

data = load_data("italian_shipments_dataset.csv")


# FILTERS (with select-all option)
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Checkbox per selezionare tutti i prodotti
if st.sidebar.checkbox("Select all Products", value=True):
    selected_products = list(data["product"].unique())
else:
    selected_products = st.sidebar.multiselect(
        "Product",
        options=data["product"].unique(),
        default=list(data["product"].unique())
    )

# Checkbox per selezionare tutti gli operatori
if st.sidebar.checkbox("Select all Operators", value=True):
    selected_operators = list(data["operator"].unique())
else:
    selected_operators = st.sidebar.multiselect(
        "Operator",
        options=data["operator"].unique(),
        default=list(data["operator"].unique())
    )

# Checkbox per selezionare tutte le città
if st.sidebar.checkbox("Select all Cities", value=True):
    selected_cities = list(data["city"].unique())
else:
    selected_cities = st.sidebar.multiselect(
        "City",
        options=data["city"].unique(),
        default=list(data["city"].unique())
    )

filtered = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1]))
    & (data["product"].isin(selected_products))
    & (data["operator"].isin(selected_operators))
    & (data["city"].isin(selected_cities))
]
# ------------------------------------------------
# EXECUTIVE SNAPSHOT  (unchanged lines omitted)
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

compliance_pct = filtered["in_range"].mean() * 100 if not filtered.empty else 0
incident_pct = filtered["out_of_range"].mean() * 100 if not filtered.empty else 0
total_shipments = len(filtered)
cost_out_range = (
    filtered.loc[filtered["out_of_range"], "shipment_cost_eur"].sum() if not filtered.empty else 0
)
co2_saved = (
    ((0.05 - 0.01) * len(filtered) * filtered["unit_co2_emitted"].mean())
    if not filtered.empty
    else 0
)

col1.metric("% Compliant Shipments", f"{compliance_pct:.1f}%")
col2.metric("% Shipments with Incidents", f"{incident_pct:.1f}%")
col3.metric("📦 Total Shipments", f"{total_shipments}")
col4.metric("Total Waste Cost (€)", f"{cost_out_range:.2f}")
col5.metric("🌱 CO₂ Saved (kg)", f"{co2_saved:.1f}")

# --- Operational Control ---
# ------------------------------------------------
# 📌 Operational Control
# ------------------------------------------------
st.header("📌 Operational Control")

# Alert Center a tutta pagina
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert from the table below to view further details._")

alert_df = filtered[filtered['out_of_range']].sort_values(
    'reading_timestamp', ascending=False
)

if alert_df.empty:
    st.success("✅ No alerts to show.")
else:
    selection_alert_df = alert_df[["shipment_id", "reading_timestamp", "operator", "product", "severity", "city", "latitude", "longitude"]].copy()
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
# SELECTED ALERTS (dettagli shipment sotto la tabella)
# ------------------------------------------------
selected_alerts = edited_alert_df[edited_alert_df["Select"]]

if len(selected_alerts) == 1:
    shipment_id = selected_alerts.iloc[0]["shipment_id"]
    shipment_details = filtered[filtered["shipment_id"] == shipment_id].iloc[0]

    st.markdown("---")
    st.header("📦 Shipment Details")

    # Due colonne affiancate: dettagli e mappa
    details_col, map_col = st.columns([1, 1])

    with details_col:
        status_text = "✅ In Range" if shipment_details['in_range'] else "❌ Out of Range"
        st.markdown(f"""
        **📅 Timestamp:** {shipment_details['reading_timestamp']}  
        **📍 City:** {shipment_details['city']}  
        **🚚 Operator:** {shipment_details['operator']}  
        **📦 Product:** {shipment_details['product']}  
        **⚠️ Severity:** {shipment_details['severity']}  

        ---

        **🌡️ Actual Temperature:** {shipment_details['actual_temperature']:.2f} °C  
        **🔹 Min Threshold:** {shipment_details['threshold_min_temperature']} °C  
        **🔸 Max Threshold:** {shipment_details['threshold_max_temperature']} °C  

        **💰 Shipment Cost:** € {shipment_details['shipment_cost_eur']}  
        **🍃 CO₂ Emitted:** {shipment_details['unit_co2_emitted']} kg  

        **🚦 Status:** {status_text}

        ---

        **🏷️ Label ID:** {shipment_details['label_id']}  
        **🆔 Shipment ID:** {shipment_details['shipment_id']}  
        **🔅 Exposure:** {shipment_details['exposure']:.2f}  
        """)

        if st.button("🚨 Report Issue", key=f"report_{shipment_id}"):
            st.warning("Reporting feature coming soon!")

    with map_col:
        st.subheader("📍 Shipment Location")
        shipment_map_df = pd.DataFrame({
            'latitude': [shipment_details['latitude']],
            'longitude': [shipment_details['longitude']],
            'product': [shipment_details['product']],
            'operator': [shipment_details['operator']],
            'city': [shipment_details['city']],
            'reading_timestamp': [shipment_details['reading_timestamp']]
        })

        shipment_map = px.scatter_mapbox(
            shipment_map_df,
            lat="latitude",
            lon="longitude",
            zoom=10,
            height=400,
            size_max=25,
            size=[25],
            mapbox_style="open-street-map",
            hover_data=["product", "operator", "city", "reading_timestamp"]
        )
        st.plotly_chart(shipment_map, use_container_width=True)

elif len(selected_alerts) > 1:
    st.warning("⚠️ Please select only one shipment at a time.")

# ------------------------------------------------
# MAPPA COMPLETA sotto ai dettagli
# ------------------------------------------------
st.markdown("---")
st.subheader("🗺️ Complete Shipment Map")

filtered['severity'] = filtered['severity'].fillna('None')
filtered.loc[filtered['severity'].str.strip() == '', 'severity'] = 'None'

size_map = {'High': 20, 'Medium': 10, 'None': 6}
filtered['marker_size'] = filtered['severity'].map(size_map)

fig_map = px.scatter_mapbox(
    filtered,
    lat="latitude",
    lon="longitude",
    color="severity",
    color_discrete_map={"High": "red", "Medium": "yellow", "None": "green"},
    size="marker_size",
    hover_data=["product", "operator", "severity"],
    mapbox_style="open-street-map",
    zoom=4,
    height=500
)
st.plotly_chart(fig_map, use_container_width=True)




# --- Analytics & Trends ---
st.header("📊 Analytics & Trends")

# Logistics Operator Ranking by Incident %
operator_incidents = (
    filtered
    .groupby('operator')['out_of_range']
    .mean()
    .reset_index()
    .sort_values('out_of_range', ascending=False)
)
fig_operator = px.bar(
    operator_incidents,
    x='operator',
    y='out_of_range',
    labels={'out_of_range': '% Incidents'},
    title="Operator Ranking by % Incidents"
)
# Critical Routes
st.subheader("📍 Critical Routes (Top 10 by Incident %)")

# Group by location and calculate total incidents and compliance percentage
location_stats = (
    filtered.groupby(['city', 'latitude', 'longitude'])
    .agg(total_readings=('in_range', 'count'),
         incidents=('out_of_range', 'sum'))
    .reset_index()
)
# Filter locations with at least 10 readings
location_stats = location_stats[location_stats['total_readings'] >= 10]
# Calculate incident percentage relative to total readings in location
location_stats['Incident (%)'] = (
    location_stats['incidents'] / location_stats['total_readings'] * 100
).round(1)

# Sort by Incident percentage (descending order)
critical_routes = location_stats.sort_values('Incident (%)', ascending=False).head(10)

# Display relevant columns
st.dataframe(
    critical_routes[['city', 'latitude', 'longitude', 'total_readings', 'incidents', 'Incident (%)']],
    use_container_width=True
)


# Monthly Compliance Trend
filtered['month'] = filtered['reading_timestamp'].dt.to_period('M').astype(str)

monthly_compliance = filtered.groupby('month')['in_range'].mean().reset_index()
monthly_compliance['Compliance (%)'] = monthly_compliance['in_range'] * 100

fig_monthly = px.line(
    monthly_compliance,
    x='month',
    y='Compliance (%)',
    labels={'Compliance (%)': 'Compliance (%)', 'month': 'Month'},
    title="Monthly % Compliance Trend",
    markers=True,
)

st.plotly_chart(fig_monthly, use_container_width=True)
# --- Product Quality Comparison ---
st.header("📦 Product Quality Comparison")

# Aggregating statistics per product
product_agg = filtered.groupby('product').agg({
    'actual_temperature': 'mean',
    'threshold_min_temperature': 'mean',
    'threshold_max_temperature': 'mean',
    'in_range': 'mean'
}).reset_index()

# Define storage quality based on compliance percentage
def storage_quality(compliance):
    if compliance >= 0.9:
        return '🟢 Good'
    elif compliance >= 0.75:
        return '🟡 Moderate'
    else:
        return '🔴 Poor'

# Apply storage quality indicator
product_agg['Storage Quality'] = product_agg['in_range'].apply(storage_quality)

# Rename columns for clarity
product_agg.columns = [
    'Product Category',
    'Avg Temperature (°C)',
    'Avg Min Temp Limit (°C)',
    'Avg Max Temp Limit (°C)',
    'Compliance (%)',
    'Storage Quality'
]

# Format compliance percentage
product_agg['Compliance (%)'] = (product_agg['Compliance (%)'] * 100).round(1)

# Display aggregated comparison table
st.dataframe(product_agg.sort_values('Compliance (%)', ascending=False), use_container_width=True)




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
