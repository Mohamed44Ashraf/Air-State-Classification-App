import streamlit as st
import pandas as pd
import joblib
import time
import os

# Page name
st.set_page_config(page_title="Air Quality Classifier", layout="wide")

try:
    model = joblib.load( "trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load( "encoder.pkl")
except Exception as e:
    st.error(f"Error loading model, scaler, or encoder: {e}")
    st.stop()
    
    

# App Title and Description
st.title("Air :red[Pollution] Prediction App")
st.image("D:\Data Analysis\GDG-CoreTeam\Air pollution\src\pexels-pixabay-459728.jpg", caption="Air Quality Monitoring", use_column_width=True)
st.subheader("Welcome to our air quality prediction tool!")
st.markdown("Use the **sidebar** to input environmental factors, and click **Predict** to determine the air quality status.")

# Sidebar Inputs
st.sidebar.title("Air Quality Input Features")

st.sidebar.subheader("Gas Levels")
co = st.sidebar.slider("Enter CO Level", min_value=0.0, max_value=20.0, step=0.1 ,
                       help="Carbon Monoxide: High levels can indicate pollution from vehicles or burning fuels.")
NO2 = st.sidebar.slider("Enter NO2 Level", min_value=0.0, max_value=100.0, step=0.1 ,
                        help="Nitrogen Dioxide: Elevated levels are often linked to industrial emissions and traffic.")
SO2 = st.sidebar.slider("Enter SO2 Level", min_value=-10.0, max_value=100.0, step=0.1,
                        help="Sulfur Dioxide: A key pollutant from burning fossil fuels, affecting respiratory health.")

st.sidebar.subheader("Particulate Matter")
Fine = st.sidebar.slider("Enter Fine Particulate Matter (µg/m³)", min_value=0.0, max_value=500.0, step=0.1,
                         help="Fine : Tiny particles that penetrate deep into the lungs, primarily from combustion.")
Coarse = st.sidebar.slider("Enter Coarse Particulate Matter (µg/m³)", min_value=-5.00, max_value=500.0, step=0.1 ,
                               help="Coarse: Larger particles that can irritate the respiratory tract, often from dust and construction."
)

st.sidebar.subheader("Environmental Factors")
Temperature = st.sidebar.slider("Enter Temperature (°C)", min_value= -30	, max_value=65, step=1,
                                help="Temperature: Can influence chemical reactions in the atmosphere, affecting pollution levels.")
Humidity = st.sidebar.slider("Enter Humidity (%)", min_value=30	, max_value=140, step=1 ,
                                 help="Humidity: High humidity can trap pollutants close to the ground, worsening air quality."
)
Nearest = st.sidebar.slider("Enter Distance to Nearest Industrial Area (km)", min_value=0.0, max_value=50.0, step=0.1 ,
                                help="Proximity to Industry: Closer distances may correlate with higher pollutant exposure."
)
Population_Density = st.sidebar.slider("Enter Population Density (people/km²)", min_value=0.0, max_value=10000.0, step=1.0 ,
                                           help="Population Density: Denser areas often experience more pollution from traffic and energy use."
)


# Load Feature Names
try:
    with open('feature_names.txt', 'r') as f:
        feature_names = f.read().splitlines()
except FileNotFoundError:
    st.error("Feature names file not found. Ensure 'feature_names.txt' is in the correct location.")
    st.stop()

# Prepare Input Data
input_data = pd.DataFrame({
    'CO': [co],
    'NO2': [NO2],
    'Temperature': [Temperature],
    'Humidity': [Humidity],
    'Fine particulate matter': [Fine],
    'Coarse particulate matter': [Coarse],
    'Nearest Industrial Areas': [Nearest],
    'Population_Density': [Population_Density],
    'SO2': [SO2],
})

# Ensure All Features Are Present
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

# Reorder Columns to Match Model Input
input_data = input_data[feature_names]

# Scale Data
try:
    scaled_data = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

# Prediction
if st.sidebar.button("Predict"):
    try:
        # Make Prediction
        classify_air_status = model.predict(scaled_data)

        # Reverse Transform Prediction to Original Label
        predicted_label = encoder.inverse_transform(classify_air_status)[0]

        # Display Results
        with st.spinner("Analyzing the data..."):
            time.sleep(1)
        st.success(f"Predicted Air Quality: **{predicted_label}**")
        
        if(predicted_label=='Good'):
            st.caption("### Great! It's essential to continue maintaining these efforts to keep the air quality at this excellent level.")
        else:
            st.caption("### Some Advice to Improve Air Quality:")
            st.markdown("""
            - Reduce emissions from vehicles by using public transport or carpooling.
            - Avoid burning waste or using wood-burning stoves.
            - Plant more trees and green spaces in urban areas.
            - Use energy-efficient appliances to reduce energy consumption.
            - Advocate for stricter industrial regulations to limit pollution.
            """)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

        
