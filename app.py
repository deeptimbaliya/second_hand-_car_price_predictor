import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: -webkit-linear-gradient(#38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    transition: 0.3s;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.card:hover {
    transform: scale(1.02);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(99,102,241,0.7);
}

/* Result animation */
.success-box {
    animation: pop 0.5s ease-in-out;
}

@keyframes pop {
    0% {transform: scale(0.8); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}

/* Loader text animation */
.loader-text {
    text-align: center;
    font-size: 20px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {opacity: 0.3;}
    50% {opacity: 1;}
    100% {opacity: 0.3;}
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<p class="title">🚗 AI Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown("### Enter details and get instant valuation 💰")

# -----------------------------
# DATA
# -----------------------------
brand_model_map = {
                   'audi': ['a4', 'a6', 'q7', 'a8'], 
                   'bentley': ['continental'], 
                   'bmw': ['5', '3', 'z4', '6', 'x5', 'x1', '7', 'x3', 'x4'], 
                   'datsun': ['redigo', 'go', 'redi-go'], 
                   'ferrari': ['gtc4lusso'], 
                   'force': ['gurkha'], 
                   'ford': ['ecosport', 'aspire', 'figo', 'endeavour', 'freestyle'], 
                   'honda': ['city', 'amaze', 'cr-v', 'jazz', 'civic', 'wr-v', 'cr'], 
                   'hyundai': ['grand', 'i20', 'i10', 'venue', 'verna', 'creta', 'santro', 'elantra', 'aura', 'tucson'], 
                   'isuzu': ['d-max', 'mux'], 
                   'jaguar': ['xf', 'f-pace', 'xe'], 
                   'jeep': ['wrangler', 'compass'], 
                   'kia': ['seltos', 'carnival'], 
                   'land rover': ['rover'], 
                   'lexus': ['es', 'nx', 'rx'], 
                   'mahindra': ['bolero', 'xuv500', 'kuv100', 'scorpio', 'marazzo', 'kuv', 'thar', 'xuv300', 'alturas'], 
                   'maruti': ['alto', 'wagon r', 'swift', 'ciaz', 'baleno', 'swift dzire', 'ignis', 'vitara', 'celerio', 'ertiga', 'eeco', 'dzire vxi', 'xl6', 's-presso', 'dzire lxi', 'dzire zxi'], 
                   'maserati': ['ghibli', 'quattroporte'], 
                   'mercedes-amg': ['c'], 
                   'mercedes-benz': ['c-class', 'e-class', 'gl-class', 's-class', 'cls', 'gls'], 
                   'mg': ['hector'], 'mini': ['cooper'], 
                   'nissan': ['kicks', 'x-trail'], 
                   'porsche': ['cayenne', 'macan', 'panamera'], 
                   'renault': ['duster', 'kwid', 'triber'], 
                   'rolls-royce': ['ghost'], 
                   'skoda': ['rapid', 'superb', 'octavia'], 
                   'tata': ['tiago', 'tigor', 'safari', 'hexa', 'nexon', 'harrier', 'altroz'], 
                   'toyota': ['innova', 'fortuner', 'camry', 'yaris', 'glanza'], 
                   'volkswagen': ['vento', 'polo'], 
                   'volvo': ['s90', 'xc', 'xc90', 'xc60']
                   }

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['km_per_year'] = X['km_driven'] / (X['vehicle_age'] + 1)
        X['power_to_engine'] = X['max_power'] / (X['engine'] + 1)
        return X

# -----------------------------
# LOAD MODEL
# -----------------------------
pipeline = joblib.load("car_price_pipeline.pkl")

# -----------------------------
# GAUGE FUNCTION
# -----------------------------
def show_price_gauge(price):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        title={'text': "Estimated Price (₹)"},
        gauge={
            'axis': {'range': [0, price * 1.8]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [0, price * 0.5], 'color': "#ef4444"},
                {'range': [price * 0.5, price * 1.2], 'color': "#facc15"},
                {'range': [price * 1.2, price * 1.8], 'color': "#22c55e"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'value': price
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=350
    )

    return fig

def animate_gauge(price):
    placeholder = st.empty()

    max_range = price * 1.8
    steps = 60  # more steps = smoother

    for i in range(steps + 1):
        value = price * i / steps

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': "Estimated Price (₹)"},
            gauge={
                'axis': {'range': [0, max_range]},
                'bar': {'color': "#22c55e"},
                'steps': [
                    {'range': [0, price * 0.5], 'color': "#ef4444"},
                    {'range': [price * 0.5, price * 1.2], 'color': "#facc15"},
                    {'range': [price * 1.2, max_range], 'color': "#22c55e"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.01)  # 🔥 smoother animation
        
# -----------------------------
# UI LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    brand = st.selectbox("🚘 Brand", list(brand_model_map.keys()))
    model_name = st.selectbox("📌 Model", brand_model_map[brand])
    vehicle_age = st.slider("📅 Vehicle Age", 0, 20, 5)
    engine = st.number_input("⚙ Engine (CC)", 800, 5000, 1200)
    max_power = st.number_input("🔥 Max Power", 40.0, 500.0, 80.0)
    seats = st.selectbox("💺 Seats", [2, 4, 5, 7, 8])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fuel_type = st.selectbox("⛽ Fuel Type", ['petrol', 'diesel', 'cng'])
    transmission_type = st.selectbox("⚡ Transmission", ['manual', 'automatic'])
    seller_type = st.selectbox("👤 Seller Type", ['dealer', 'individual'])
    km_driven = st.number_input("📍 KM Driven", 0, 300000, 50000)
    mileage = st.number_input("📊 Mileage", 10.0, 40.0, 18.0)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PREDICTION
# -----------------------------
st.markdown("###")

if st.button("🚀 Predict Price"):

    input_data = pd.DataFrame([{
        'brand': brand.lower(),
        'model': model_name.lower(),
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'fuel_type': fuel_type.lower(),
        'seller_type': seller_type.lower(),
        'transmission_type': transmission_type.lower(),
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }])

    try:
        # ⏳ LOADER
        with st.spinner(""):
            st.markdown('<p class="loader-text">🚗 AI is analyzing your car...</p>', unsafe_allow_html=True)
            time.sleep(1.2)

            prediction = pipeline.predict(input_data)
            prediction = np.expm1(prediction)
            price = int(prediction[0])

        # 💰 RESULT CARD
        st.markdown(
            f'<div class="card success-box"><h2>💰 Estimated Price: ₹ {price:,}</h2></div>',
            unsafe_allow_html=True
        )

        # 📊 ANIMATED GAUGE
        animate_gauge(price)

    except Exception as e:
        st.error("❌ Error in prediction")
        st.write(e)