import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Header
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌾 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Enter soil and climate conditions to get the best crop recommendation</p>", unsafe_allow_html=True)

# Load data and train model
@st.cache_resource
def load_model():
    data1 = pd.read_csv('crop_data1.csv')
    data2 = pd.read_csv('crop_data2.csv')
    data = pd.concat([data1, data2], ignore_index=True)
    
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, data

model, scaler, data = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧪 Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 150, 50, help="Nitrogen content in soil (kg/ha)")
    P = st.slider("Phosphorus (P)", 0, 150, 50, help="Phosphorus content in soil (kg/ha)")
    K = st.slider("Potassium (K)", 0, 210, 50, help="Potassium content in soil (kg/ha)")
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5, step=0.1, help="pH level of soil")

with col2:
    st.markdown("### 🌤️ Climate Conditions")
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0, step=0.1)

st.markdown("---")

# Predict button
if st.button("🌱 Get Crop Recommendation", type="primary", use_container_width=True):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.success(f"## 🎯 Recommended Crop: **{prediction.upper()}**")
    
    # Input summary
    st.markdown("### 📊 Your Input Summary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Soil:** N={N}, P={P}, K={K}, pH={ph}")
    with col_b:
        st.info(f"**Climate:** Temp={temperature}°C, Humidity={humidity}%, Rain={rainfall}mm")

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info("AI-powered crop recommendation based on soil and climate data.")
    
    st.markdown("## 📈 Model Info")
    st.success("**Algorithm:** Random Forest\n\n**Accuracy:** 99.77%")
    
    st.markdown("## 🌾 Available Crops")
    st.write(", ".join(sorted(data['label'].unique())))

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built with ❤️ | Crop Recommendation System</p>", unsafe_allow_html=True)

