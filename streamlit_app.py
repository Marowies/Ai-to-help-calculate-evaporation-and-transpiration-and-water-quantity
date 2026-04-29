import streamlit as st
import pandas as pd
import joblib
import os

# Set Page Config
st.set_page_config(page_title="Cosmic Irrigation System", layout="wide")

# Styling
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; width: 100%; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌌 Cosmic Irrigation Prediction")
st.write("Futuristic AI System for Water Management")

# Load Models
@st.cache_resource
def load_models():
    try:
        etc_model = joblib.load('saved_models_v2/pipeline_etc.pkl')
        water_model = joblib.load('saved_models_v2/water_model.pkl')
        return etc_model, water_model
    except:
        return None, None

etc_model, water_model = load_models()

if etc_model is None:
    st.error("⚠️ Model files not found! Please ensure 'saved_models_v2' folder exists.")
else:
    # Sidebar Inputs
    st.sidebar.header("📡 Environmental Inputs")
    t_max = st.sidebar.number_input("Max Temp (°C)", 0.0, 60.0, 30.0)
    t_min = st.sidebar.number_input("Min Temp (°C)", 0.0, 60.0, 20.0)
    rh = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    ws = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.5)
    ps = st.sidebar.number_input("Pressure (kPa)", 50.0, 110.0, 101.0)
    rn = st.sidebar.number_input("Net Radiation (MJ)", 0.0, 50.0, 15.0)

    st.sidebar.header("🌿 Crop & Soil Settings")
    crop = st.sidebar.selectbox("Crop Type", ["Spinach", "Other"])
    stage = st.sidebar.selectbox("Growth Stage", ["Seedling", "Vegetative", "Flowering", "Maturity"])
    soil = st.sidebar.selectbox("Soil Type", ["Clay", "Sandy", "Loam"])
    root_depth = st.sidebar.number_input("Root Depth (m)", 0.1, 2.0, 0.5)

    # Kc Mapping
    kc_map = {"Seedling": 0.7, "Vegetative": 1.0, "Flowering": 1.05, "Maturity": 0.95}
    kc = kc_map.get(stage, 0.85)

    # Main Area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧱 Soil Properties")
        fc = st.number_input("Field Capacity (FC)", 0.0, 1.0, 0.35)
        wp = st.number_input("Wilting Point (WP)", 0.0, 1.0, 0.15)

    with col2:
        st.subheader("ℹ️ Selected Kc")
        st.info(f"Automatic Kc for {stage}: **{kc}**")

    # Predict Button
    if st.button("🚀 Calculate Irrigation Requirements"):
        # Prepare Data
        input_data = pd.DataFrame([{
            "T2M_MAX": t_max, "T2M_MIN": t_min, "RH2M": rh, "WS2M": ws,
            "PS": ps, "Rn": rn, "G": 0.0, "CROP": crop, "GrowthStage": stage,
            "SOIL_TEX": soil, "RootDepth": root_depth, "FieldCapacity": fc,
            "WiltingPoint": wp, "Kc": kc, "T_mean": (t_max + t_min) / 2
        }])

        # Prediction
        etc_pred = etc_model.predict(input_data)[0]
        water_pred = water_model.predict(input_data)[0]

        # Results
        st.success("✅ Prediction Complete!")
        res1, res2 = st.columns(2)
        res1.metric("ETc (Evapotranspiration)", f"{etc_pred:.4f} mm/day")
        res2.metric("Water Requirement", f"{water_pred:.2f} m³/ha")

        st.balloons()

# Footer
st.markdown("---")
st.markdown("<center>Developed by <b>Marwan _ewis</b> | © 2026 Cosmic Irrigation</center>", unsafe_allow_html=True)
