# ------------------------------------------------
# GEOCODER SETUP
# ------------------------------------------------
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Instantiate geocoder and rate limiter
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
    # annotate market city
    df['Market City'] = market_city
    # reverse-geocode Comune for each coordinate
    df['Comune'] = df.apply(lambda row: reverse_geocode(row['latitude'], row['longitude']), axis=1)
    # store location info string
    df['Location Info'] = df['latitude'].astype(str) + ", " + df['longitude'].astype(str) + f" - {market_city}"
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

# load and concatenate data
frames = []
for city in selected_cities:
    file = market_files[city]
    if os.path.exists(file):
        df_city = load_data(file, city)
        frames.append(df_city)
    else:
        st.sidebar.error(f"File not found: {file}")

if not frames:
    st.error("No data loaded. Please check your Market files.")
    st.stop()

data = pd.concat(frames, ignore_index=True)

# ------------------------------------------------
# FILTERS
# ------------------------------------------------
min_date = data['reading_timestamp'].dt.date.min()
max_date = data['reading_timestamp'].dt.date.max()
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Period", [min_date, max_date])

# Product filter
if st.sidebar.checkbox("Select all Products", True):
    sel_prod = data['product'].unique().tolist()
else:
    sel_prod = st.sidebar.multiselect("Product", data['product'].unique(), default=data['product'].unique())

# Operator filter
if st.sidebar.checkbox("Select all Operators", True):
    sel_op = data['operator'].unique().tolist()
else:
    sel_op = st.sidebar.multiselect("Operator", data['operator'].unique(), default=data['operator'].unique())

# Comune filter
if st.sidebar.checkbox("Select all Comuni", True):
    sel_com = data['Comune'].unique().tolist()
else:
    sel_com = st.sidebar.multiselect("Comune", data['Comune'].unique(), default=data['Comune'].unique())

# Apply filters
filtered = data[
    (data['reading_timestamp'].dt.date.between(date_range[0], date_range[1])) &
    data['product'].isin(sel_prod) &
    data['operator'].isin(sel_op) &
    data['Comune'].isin(sel_com)
]

# ------------------------------------------------
# EXECUTIVE SNAPSHOT
# ------------------------------------------------
st.header("🚦 Executive Snapshot")
metrics = st.columns(5)
if not filtered.empty:
    comp = filtered['in_range'].mean() * 100
    inc = filtered['out_of_range'].mean() * 100
    total = len(filtered)
    waste = filtered.loc[filtered['out_of_range'], 'shipment_cost_eur'].sum()
    saved = (0.05 - 0.01) * total * filtered['unit_co2_emitted'].mean()
else:
    comp = inc = total = waste = saved = 0
metrics[0].metric("% Compliant", f"{comp:.1f}%")
metrics[1].metric("% Incidents", f"{inc:.1f}%")
metrics[2].metric("Total Shipments", f"{total}")
metrics[3].metric("Waste Cost (€)", f"{waste:.2f}")
metrics[4].metric("CO₂ Saved (kg)", f"{saved:.1f}")

# ------------------------------------------------
# OPERATIONAL CONTROL
# ------------------------------------------------
st.header("📌 Operational Control")
st.subheader("🚨 Alert Center")
st.markdown("_Select an alert to view details._")
alerts = filtered[filtered['out_of_range']].sort_values('reading_timestamp', ascending=False)
if alerts.empty:
    st.success("No alerts.")
else:
    alert_disp = alerts[['shipment_id','reading_timestamp','operator','product','severity','Comune']].copy()
    alert_disp.insert(0, 'Select', False)
    st.data_editor(alert_disp, hide_index=True, use_container_width=True,
                   column_config={'Select': st.column_config.CheckboxColumn(required=True)}, key='alerts')

# ------------------------------------------------
# ALL SHIPMENTS
# ------------------------------------------------
st.subheader("📋 All Shipments")
st.markdown("_Filtered shipments with Comune._")
cols = ['shipment_id','reading_timestamp','operator','product',
        'actual_temperature','threshold_min_temperature','threshold_max_temperature',
        'in_range','out_of_range','shipment_cost_eur','unit_co2_emitted','Comune']
all_ship = filtered[cols].copy()
all_ship.columns = [
    'Shipment ID','Timestamp','Operator','Product',
    'Actual Temp (°C)','Min Temp','Max Temp',
    'In Range','Out of Range','Cost (€)','CO₂ Emitted (kg)','Comune'
]
st.dataframe(all_ship.sort_values('Timestamp', ascending=False), use_container_width=True)
