import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime

# 1. Page Config & CSS
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
        overflow: hidden; /* Prevents text from spilling out */
    }
    
    .label { 
        font-size: 0.85rem; 
        letter-spacing: 4px; 
        text-transform: uppercase; 
        color: #888888; 
        margin-bottom: 10px;
    }
    
    /* --- RESPONSIVE DYNAMIC FONT SIZE --- */
    .value { 
        font-size: 5vw; /* Scales based on screen width (approx 5-6rem) */
        font-weight: 800; 
        color: #FFFFFF; 
        line-height: 1.1; 
        display: inline-block;
        white-space: nowrap; /* Forces number to stay on one line */
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

# 2. Load Simulation Model
@st.cache_resource
def load_model():
    return joblib.load('dalma_energy_sim_model.pkl')

model = load_model()

# 3. Logic Utilities
def sync_val(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]

if 'db' not in st.session_state: st.session_state.db = 32.0
if 'wb' not in st.session_state: st.session_state.wb = 24.0
if 'ff' not in st.session_state: st.session_state.ff = 25000.0
if 'occ' not in st.session_state: st.session_state.occ = 90.0

# --- HEADER ---
st.title("DALMA MALL")
st.markdown("<p style='color:#888888; margin-bottom:40px;'>SIMULATION ENGINE v6.0 / LAG-INDEPENDENT</p>", unsafe_allow_html=True)

# --- INPUTS ---
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

# --- CALCULATION ---
month = target_date.month
m_sin, m_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
h = (1.006 * st.session_state.db) + (st.session_state.wb * 0.05)

# Input Row (No Lags)
input_data = pd.DataFrame([[
    h, st.session_state.db, st.session_state.wb, 
    st.session_state.ff, st.session_state.occ, 
    is_weekend, 1, m_sin, m_cos
]], columns=['enthalpy', 'dry_bulb', 'wet_bulb', 'footfall', 'occupancy_pct', 'Is_weekend', 'is_optimized', 'month_sin', 'month_cos'])

# Predict & Inverse Log
pred = np.expm1(model.predict(input_data))
k_val, t_val = pred[0][0], pred[0][1]


# --- CALCULATION (Add this line) ---
# ... (existing prediction code) ...
k_val, t_val = pred[0][0], pred[0][1]

# Calculate COP: (Tons * 3.517) / kWs
# Use a small epsilon (1e-6) to prevent division by zero
cop_val = (t_val * 3.517) / max(k_val, 1e-6)

# --- DISPLAY (Update this section) ---
res1, res2, res3 = st.columns(3) # Changed to 3 columns

with res1: 
    st.markdown(f'<div class="result-card"><p class="label">Power Demand</p><span class="value">{k_val:,.0f}</span><span class="unit">kWs</span></div>', unsafe_allow_html=True)
    
with res2: 
    st.markdown(f'<div class="result-card"><p class="label">Cooling Load</p><span class="value">{t_val:,.0f}</span><span class="unit">Tons</span></div>', unsafe_allow_html=True)

# with res3:
#     # Highlighting COP in a different color (Gold/Orange) to show Efficiency
#     st.markdown(f'''
#         <div class="result-card">
#             <p class="label">System Efficiency</p>
#             <span class="value" style="color: #FFD700;">{cop_val:.2f}</span><span class="unit">COP</span>
#         </div>
#     ''', unsafe_allow_html=True)