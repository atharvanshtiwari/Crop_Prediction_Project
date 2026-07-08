import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import plotly.express as px
from google import genai

# --- DEVELOPER API CONFIGURATION ---
# Securely fetching the API key from your .streamlit/secrets.toml file!
# Ensure your secrets.toml has this exact line: GEMINI_API_KEY = "your_actual_key"
API_KEY = st.secrets["GEMINI_API_KEY"]

# 1. Page Configuration
st.set_page_config(page_title="AI Crop Predictor", page_icon="🌱", layout="wide")

# 2. Helper Function to load Lottie Animations from a URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 3. Load the pre-trained Machine Learning Model's "Brain"
@st.cache_resource
def load_model():
    with open('crop_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run `model_training.py` first to generate `crop_model.pkl`.")
    st.stop()

# --- MAIN UI: Header & Animation ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🌱 Smart Crop Predictor & AI Advisor")
    st.write("""
    Enter your farm's soil metrics and environmental conditions below. 
    Our Machine Learning model will predict the best crop to grow, and our AI bot will give you tailored advice!
    """)

with col2:
    # Load a free farming/plant animation from LottieFiles
    lottie_plant = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_Zz37yH.json")
    if lottie_plant:
        st_lottie(lottie_plant, height=150, key="plant_anim")

# --- USER INPUT SECTION ---
st.markdown("### 📊 Enter Your Farm Data")
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    N = st.number_input("Nitrogen (N) ratio", min_value=0.0, max_value=150.0, value=90.0)
    P = st.number_input("Phosphorus (P) ratio", min_value=0.0, max_value=150.0, value=42.0)
    K = st.number_input("Potassium (K) ratio", min_value=0.0, max_value=210.0, value=43.0)

with input_col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=55.0, value=20.8)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)

with input_col3:
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=202.9)

# --- PREDICTION & AI BOT SECTION ---
if st.button("🔮 Predict Best Crop", use_container_width=True):
    
    # Package the user's inputs
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    
    # Get the ML Prediction & Probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    crop_classes = model.classes_ 
    
    prob_df = pd.DataFrame({
        'Crop': crop_classes,
        'Probability': probabilities
    })
    
    top_5_crops = prob_df.sort_values(by='Probability', ascending=False).head(5)
    
    # --- DISPLAY RESULTS ---
    st.markdown("---")
    st.success(f"### 🏆 The Model Predicts: **{prediction.capitalize()}**")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("#### 📈 Prediction Probabilities")
        fig = px.pie(
            top_5_crops, 
            values='Probability', 
            names='Crop', 
            hole=0.4, 
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with res_col2:
        st.markdown("#### 🤖 AI Expert Advisor Note")
        
        with st.spinner("Analyzing farm data and generating expert advice..."):
            try:
                client = genai.Client(api_key=API_KEY)
                
                prompt = f"""
                You are an expert agricultural advisor. A farmer's soil has the following metrics: 
                Nitrogen: {N}, Phosphorus: {P}, Potassium: {K}, Temp: {temp}°C, Humidity: {humidity}%, pH: {ph}, Rainfall: {rainfall}mm. 
                Our Machine Learning model has predicted that '{prediction}' is the best crop to grow. 
                Write a highly encouraging, professional 3-4 line note to the farmer explaining what to expect with this crop and giving one quick, specific farming tip based on their exact metrics. Do not use markdown formatting.
                """
                
                # Fetch response from AI using the new client format
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                
                st.info(response.text)
                
            except Exception as e:
                st.error(f"Could not connect to the AI Advisor. Please check your API key. Error Details: {e}")
