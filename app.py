import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import requests
from datetime import timedelta
import plotly.graph_objects as go

# -------------------------------
# 1. Page Config & CSS (unchanged)
# -------------------------------
st.set_page_config(page_title="Dalma Mall | Energy Intelligence", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] { 
        background-color: #000000; 
        font-family: 'Inter', sans-serif; 
        color: #FFFFFF; 
    }
    
    [data-testid="stHorizontalBlock"] { 
        background-color: #111111; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #222222; 
    }
    
    .result-card { 
        border-bottom: 1px solid #333333; 
        padding: 40px 0;
        overflow: hidden;
    }
    
    .label { 
        font-size: 0.85rem; 
        letter-spacing: 4px; 
        text-transform: uppercase; 
        color: #888888; 
        margin-bottom: 10px;
    }
    
    .value { 
        font-size: 5vw;
        font-weight: 800; 
        color: #FFFFFF; 
        line-height: 1.1; 
        display: inline-block;
        white-space: nowrap;
    }
    
    .unit { 
        font-size: 1.2rem; 
        color: #00FF6A; 
        margin-left: 10px; 
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 400;
    }
    
    input { background-color: #222222 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 2. Load models
# -------------------------------
@st.cache_resource
def load_energy_model():
    return joblib.load('dalma_energy_sim_model.pkl')

@st.cache_resource
def load_historical_params():
    return joblib.load('historical_parameters.pkl')

@st.cache_resource
def load_temp_correction():
    return joblib.load('temperature_correction.pkl')

model = load_energy_model()
hist_params = load_historical_params()
temp_corr = load_temp_correction()

# -------------------------------
# 3. Helper functions for auto mode
# -------------------------------
def stull_wet_bulb(temp_c, rh_pct):
    """Stull (2011) approximation for wet-bulb temperature."""
    rh_frac = rh_pct / 100.0
    term1 = temp_c * np.arctan(0.151977 * (rh_frac * 100 + 8.313659) ** 0.5)
    term2 = np.arctan(temp_c + rh_frac * 100)
    term3 = np.arctan(rh_frac * 100 - 1.67631)
    term4 = 0.00391838 * (rh_frac * 100) ** 1.5 * np.arctan(0.023101 * (rh_frac * 100))
    return term1 + term2 - term3 + term4

def fetch_openmeteo(lat, lon, start_date, end_date):
    """Fetch daily dry bulb and relative humidity from Open-Meteo."""
    today = datetime.date.today()
    if start_date <= today and end_date <= today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if "daily" not in data:
        st.error(f"Open-Meteo error: {data.get('reason', 'Unknown')}")
        return None
    df = pd.DataFrame(data['daily'])
    df['date'] = pd.to_datetime(df['time'])
    df.rename(columns={'temperature_2m_mean': 'dry_bulb_api',
                       'relative_humidity_2m_mean': 'rh_api'}, inplace=True)
    return df

def correct_temperature(api_dry, api_wet):
    """Apply linear correction to API temperatures."""
    dry_corr = temp_corr['dry_bulb']['intercept'] + temp_corr['dry_bulb']['slope'] * api_dry
    wet_corr = temp_corr['wet_bulb']['intercept'] + temp_corr['wet_bulb']['slope'] * api_wet
    return dry_corr, wet_corr

def predict_footfall(date):
    weekday = date.weekday()
    return hist_params['weekday_footfall_avg'].get(weekday, 25000.0)

def predict_occupancy(date):
    month = date.month
    return hist_params['monthly_occ_median'].get(month, hist_params['default_occ'])

def generate_energy_prediction(date, dry_bulb, wet_bulb, footfall, occupancy):
    """Compute all features and call energy model."""
    is_weekend = 1 if date.weekday() in [4,5] else 0
    month = date.month
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)
    enthalpy = 1.006 * dry_bulb + 0.05 * wet_bulb
    
    input_df = pd.DataFrame([[
        enthalpy, dry_bulb, wet_bulb, footfall, occupancy,
        is_weekend, 1, m_sin, m_cos
    ]], columns=['enthalpy', 'dry_bulb', 'wet_bulb', 'footfall', 'occupancy_pct',
                 'Is_weekend', 'is_optimized', 'month_sin', 'month_cos'])
    
    pred_log = model.predict(input_df)[0]
    kw, ton = np.expm1(pred_log[0]), np.expm1(pred_log[1])
    return kw, ton

# -------------------------------
# 4. Mode selection (Manual vs Auto)
# -------------------------------
mode = st.radio("Prediction Mode", ["Manual (enter parameters)", "Auto (date range)"], horizontal=True)

if mode == "Auto (date range)":
    st.subheader("Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.date.today())
    with col2:
        end_date = st.date_input("End date", datetime.date.today() + timedelta(days=7))
    
    max_future = datetime.date.today() + timedelta(days=16)
    if end_date > max_future:
        st.warning(f"Open-Meteo forecast only provides 16 days ahead. Limit end date to {max_future}.")
        end_date = max_future
        st.info(f"Adjusted end date to {end_date}")
    
    if st.button("Generate Forecast", use_container_width=True):
        lat, lon = "24.333333466152357", "54.52473814417924"
        weather_df = fetch_openmeteo(lat, lon, start_date, end_date)
        if weather_df is None:
            st.stop()
        
        results = []
        progress = st.progress(0)
        for i, row in weather_df.iterrows():
            date = row['date']
            dry_api = row['dry_bulb_api']
            rh = row['rh_api']
            wet_api_raw = stull_wet_bulb(dry_api, rh)
            dry_corr, wet_corr = correct_temperature(dry_api, wet_api_raw)
            foot = predict_footfall(date)
            occ = predict_occupancy(date)
            kw, ton = generate_energy_prediction(date, dry_corr, wet_corr, foot, occ)
            results.append({"date": date, "KWs": kw, "Tons": ton})
            progress.progress((i+1)/len(weather_df))
        
        df_res = pd.DataFrame(results)
        st.success("Forecast complete!")
        
        # --- Plotly Chart (black background, red/blue, clean hover) ---
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_res['date'],
            y=df_res['KWs'],
            mode='lines+markers',
            name='Power Demand (kWs)',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(size=4, color='#FF4B4B'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>KWs: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_res['date'],
            y=df_res['Tons'],
            mode='lines+markers',
            name='Cooling Load (Tons)',
            line=dict(color='#4B9EFF', width=2),
            marker=dict(size=4, color='#4B9EFF'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Tons: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Energy Forecast Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='#FFFFFF'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#333333')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#333333')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Data Table ---
        st.subheader("Daily Predictions")
        display_df = df_res.copy()
        display_df['KWs'] = display_df['KWs'].map(lambda x: f"{x:,.0f}")
        display_df['Tons'] = display_df['Tons'].map(lambda x: f"{x:,.0f}")
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df.columns = ['Date', 'Power Demand (kWs)', 'Cooling Load (Tons)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download CSV
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")
    st.stop()

# -------------------------------
# 5. ORIGINAL MANUAL MODE (unchanged)
# -------------------------------
def sync_val(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]

if 'db' not in st.session_state: st.session_state.db = 32.0
if 'wb' not in st.session_state: st.session_state.wb = 24.0
if 'ff' not in st.session_state: st.session_state.ff = 25000.0
if 'occ' not in st.session_state: st.session_state.occ = 90.0

st.title("DALMA MALL")
st.markdown("<p style='color:#888888; margin-bottom:40px;'>SIMULATION ENGINE v6.0 / LAG-INDEPENDENT</p>", unsafe_allow_html=True)

target_date = st.date_input("SELECT DATE", datetime.date.today())
is_weekend = 1 if target_date.weekday() in [4, 5] else 0

for label, key, min_v, max_v in [
    ("DRY BULB (°C)", "db", 10.0, 55.0),
    ("WET BULB (°C)", "wb", 5.0, 50.0),
    ("FOOTFALL", "ff", 5000, 80000),
    ("OCCUPANCY %", "occ", 50.0, 100.0)
]:
    c1, c2 = st.columns([1, 4])
    with c1: st.number_input(f"{key}_box", min_v, max_v, key=f"{key}_box", on_change=sync_val, args=(f"{key}_box", key), label_visibility="collapsed")
    with c2: st.slider(label, min_v, max_v, key=key, on_change=sync_val, args=(key, f"{key}_box"))

month = target_date.month
m_sin, m_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
h = (1.006 * st.session_state.db) + (st.session_state.wb * 0.05)

input_data = pd.DataFrame([[
    h, st.session_state.db, st.session_state.wb, st.session_state.ff, st.session_state.occ,
    is_weekend, 1, m_sin, m_cos
]], columns=['enthalpy', 'dry_bulb', 'wet_bulb', 'footfall', 'occupancy_pct',
             'Is_weekend', 'is_optimized', 'month_sin', 'month_cos'])

pred = np.expm1(model.predict(input_data))
k_val, t_val = pred[0][0], pred[0][1]

res1, res2 = st.columns(2)
with res1: 
    st.markdown(f'<div class="result-card"><p class="label">Power Demand</p><span class="value">{k_val:,.0f}</span><span class="unit">kWs</span></div>', unsafe_allow_html=True)
with res2: 
    st.markdown(f'<div class="result-card"><p class="label">Cooling Load</p><span class="value">{t_val:,.0f}</span><span class="unit">Tons</span></div>', unsafe_allow_html=True)