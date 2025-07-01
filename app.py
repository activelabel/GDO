import os, json
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# OPENAI CLIENT
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
# GEOCODER SETUP
# ------------------------------------------------
geolocator = Nominatim(user_agent="active_label_app")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# ------------------------------------------------
# DATA LOADING
# ------------------------------------------------
@st.cache_data
def load_data(path: str, market_city: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["reading_timestamp"],
        dayfirst=True
    )
    df['Market City'] = market_city
    df['Latitude'] = df['latitude']
    df['Longitude'] = df['longitude']
    # store raw df without Comune
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
    "Choose cities to include", list(market_files.keys()),
    default=list(market_files.keys())
)

frames = []
for city in selected_cities:
    file = market_files[city]
    if os.path.exists(file):
        df_city = load_data(file, city)
        frames.append(df_city)
    else:
        st.sidebar.error(f"File not found: {file}")

if not frames:
    st.error("No data loaded. Please upload Market files.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

# ------------------------------------------------
# REVERSE GEOCODING (unique coords)
# ------------------------------------------------
@st.cache_data
def compute_comuni(df: pd.DataFrame) -> dict:
    mapping = {}
    coords = df[['Latitude','Longitude']].drop_duplicates()
    for _, row in coords.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        try:
            loc = reverse((lat, lon), language='en')
            addr = loc.raw.get('address', {})
            for fld in ('municipality','city','town','village'):
                if fld in addr:
                    mapping[(lat, lon)] = addr[fld]
                    break
            else:
                mapping[(lat, lon)] = 'Unknown'
        except:
            mapping[(lat, lon)] = 'Unknown'
    return mapping

# Compute Comune mapping once
dict_comuni = compute_comuni(data)
# Annotate Comune and Location Info
data['Comune'] = data.apply(lambda r: dict_comuni[(r['Latitude'], r['Longitude'])], axis=1)
data['Location Info'] = data['Latitude'].astype(str) + ", " + data['Longitude'].astype(str) + " - " + data['Market City']

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data['reading_timestamp'].dt.date.min()
max_date = data['reading_timestamp'].dt.date.max()
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

if st.sidebar.checkbox("Select all Products", True):
    sel_prod = data['product'].unique().tolist()
else:
    sel_prod = st.sidebar.multiselect(
        "Product", data['product'].unique(), default=data['product'].unique()
    )

if st.sidebar.checkbox("Select all Operators", True):
    sel_op = data['operator'].unique().tolist()
else:
    sel_op = st.sidebar.multiselect(
        "Operator", data['operator'].unique(), default=data['operator'].unique()
    )

if st.sidebar.checkbox("Select all Comuni", True):
    sel_com = data['Comune'].unique().tolist()
else:
    sel_com = st.sidebar.multiselect(
        "Comune", data['Comune'].unique(), default=data['Comune'].unique()
    )

# Apply filters
df_filt = data[
    data['reading_timestamp'].dt.date.between(date_range[0], date_range[1]) &
    data['product'].isin(sel_prod) &
    data['operator'].isin(sel_op) &
    data['Comune'].isin(sel_com)
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
metrics = st.columns(5)
if not df_filt.empty:
    comp = df_filt['in_range'].mean()*100
    inc = df_filt['out_of_range'].mean()*100
    tot = len(df_filt)
    waste = df_filt.loc[df_filt['out_of_range'], 'shipment_cost_eur'].sum()
    saved = (0.05-0.01)*tot*df_filt['unit_co2_emitted'].mean()
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
alerts = df_filt[df_filt['out_of_range']].sort_values('reading_timestamp', ascending=False)
if alerts.empty:
    st.success("No alerts.")
else:
    disp = alerts[['shipment_id','reading_timestamp','operator','product','severity','Comune']].copy()
    disp.insert(0, 'Select', False)
    st.data_editor(
        disp, hide_index=True, use_container_width=True,
        column_config={'Select': st.column_config.CheckboxColumn(required=True)}, key='alerts'
    )

# ------------------------------------------------
# ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments with Comune._")
cols = ['shipment_id','reading_timestamp','operator','product',
        'actual_temperature','threshold_min_temperature','threshold_max_temperature',
        'in_range','out_of_range','shipment_cost_eur','unit_co2_emitted','Comune']
all_ship = df_filt[cols].copy()
all_ship.columns = ['Shipment ID','Timestamp','Operator','Product',
                    'Actual Temp (°C)','Min Temp','Max Temp',
                    'In Range','Out of Range','Cost (€)','CO₂ Emitted (kg)','Comune']
st.dataframe(all_ship.sort_values('Timestamp', ascending=False), use_container_width=True)
