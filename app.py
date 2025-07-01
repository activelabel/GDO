# ------------------------------------------------
# IMPORT
# ------------------------------------------------
import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

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
# GEOCODER SETUP
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def get_geocoder():
    geolocator = Nominatim(user_agent="active_label_app")
    return RateLimiter(geolocator.reverse, min_delay_seconds=1)

reverse = get_geocoder()

@st.cache_data(show_spinner=False)
def reverse_geocode(lat, lon):
    try:
        location = reverse((lat, lon), language='en')
        address = location.raw.get('address', {})
        # Prefer municipality, town, village
        for key in ('municipality','town','city','village'):
            if key in address:
                return address[key]
    except:
        pass
    return "Unknown"

# ------------------------------------------------
# DATA LOADING
# ------------------------------------------------
@st.cache_data
def load_and_annotate(path: str, city_name: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["reading_timestamp"],
        dayfirst=True
    )
    df["Market City"] = city_name
    df["Latitude"] = df["latitude"]
    df["Longitude"] = df["longitude"]
    # Compute comune for each row
    df["Comune"] = df.apply(lambda r: reverse_geocode(r["Latitude"], r["Longitude"]), axis=1)
    df["Location Info"] = df["Latitude"].astype(str) + ", " + df["Longitude"].astype(str) + f" - {city_name}"
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

# Load and merge dataframes
frames = []
for city in selected_cities:
    path = market_files[city]
    if os.path.exists(path):
        df_city = load_and_annotate(path, city)
        frames.append(df_city)
    else:
        st.sidebar.error(f"File not found: {path}")

if not frames:
    st.error("No data loaded. Please upload Market files.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data["reading_timestamp"].dt.date.min()
max_date = data["reading_timestamp"].dt.date.max()
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Product
if st.sidebar.checkbox("Select all Products", value=True):
    sel_prod = data["product"].unique().tolist()
else:
    sel_prod = st.sidebar.multiselect("Product", data["product"].unique(), default=data["product"].unique())
# Operator
if st.sidebar.checkbox("Select all Operators", value=True):
    sel_op = data["operator"].unique().tolist()
else:
    sel_op = st.sidebar.multiselect("Operator", data["operator"].unique(), default=data["operator"].unique())
# Comune
if st.sidebar.checkbox("Select all Comuni", value=True):
    sel_com = data["Comune"].unique().tolist()
else:
    sel_com = st.sidebar.multiselect("Comune", data["Comune"].unique(), default=data["Comune"].unique())

# Apply filters
df_filt = data[
    (data["reading_timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (data["product"].isin(sel_prod)) &
    (data["operator"].isin(sel_op)) &
    (data["Comune"].isin(sel_com))
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)
if not df_filt.empty:
    comp = df_filt["in_range"].mean()*100
    inc = df_filt["out_of_range"].mean()*100
    tot = len(df_filt)
    waste = df_filt.loc[df_filt["out_of_range"],"shipment_cost_eur"].sum()
    saved = (0.05-0.01)*tot*df_filt["unit_co2_emitted"].mean()
else:
    comp=inc=tot=waste=saved=0
col1.metric("% Compliant", f"{comp:.1f}%")
col2.metric("% Incidents", f"{inc:.1f}%")
col3.metric("Total", f"{tot}")
col4.metric("Waste (€)", f"{waste:.2f}")
col5.metric("CO₂ Saved", f"{saved:.1f}")

# ------------------------------------------------
# OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert to view details._")
alerts = df_filt[df_filt["out_of_range"]].sort_values("reading_timestamp",ascending=False)
if alerts.empty:
    st.success("No alerts.")
else:
    disp = alerts[["shipment_id","reading_timestamp","operator","product","severity","Comune"]].copy()
    disp.insert(0,"Select",False)
    st.data_editor(disp,hide_index=True,use_container_width=True,
                   column_config={"Select":st.column_config.CheckboxColumn(required=True)},key="alerts")

# ------------------------------------------------
# ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments list with Comune._")
cols = ["shipment_id","reading_timestamp","operator","product",
       "actual_temperature","threshold_min_temperature","threshold_max_temperature",
       "in_range","out_of_range","shipment_cost_eur","unit_co2_emitted","Comune"]
all_s = df_filt[cols].copy()
all_s.columns=["Shipment ID","Timestamp","Operator","Product",
               "Temp (°C)","Min Temp","Max Temp",
               "In Range","Out of Range","Cost (€)","CO₂ Emitted","Comune"]
st.dataframe(all_s.sort_values("Timestamp",ascending=False),use_container_width=True)

