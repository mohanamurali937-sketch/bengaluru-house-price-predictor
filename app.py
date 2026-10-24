import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-title {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        font-weight: 700;
    }
    .sub-text {
        text-align: center;
        color: #cccccc;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">🏠 Bengaluru Smart Home Predictor</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-text">Powered by your trained Machine Learning Model</p>',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model_and_columns():
  model = joblib.load("house_price_model.pkl")
  data_columns = joblib.load("columns.pkl")
  return model, data_columns


try:
  model, data_columns = load_model_and_columns()
except Exception as e:
  st.error(
      f"Error loading model or columns: {e}. Make sure you ran train_model.py"
      " first!"
  )
  st.stop()
loc_columns = [
    col[9:] for col in data_columns if col.startswith("location_")
]
if not loc_columns:
  loc_columns = ["Whitefield", "Indira Nagar", "Electronic City", "Koramangala"]
col1, col2 = st.columns(2)

with col1:
  sqft = st.number_input(
      "📐 Total Square Feet", min_value=300, max_value=10000, value=1200, step=50
  )
  bath = st.selectbox(
      "🛁 Number of Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1
  )

with col2:
  bhk = st.selectbox("🛏️ BHK (Bedrooms)", options=[1, 2, 3, 4, 5], index=1)
  location = st.selectbox("📍 Location", options=sorted(loc_columns))

st.markdown("---")
if st.button("✨ Predict Market Price"):
  with st.spinner("Crunching real estate data with your ML model..."):
    try:
      x = np.zeros(len(data_columns))
      x[0] = sqft
      x[1] = bath
      x[2] = bhk
      loc_key = f"location_{location}"
      if loc_key in data_columns:
        loc_index = np.where(data_columns == loc_key)[0][0]
        x[loc_index] = 1
      predicted_price = model.predict([x])[0]

      st.success("🎉 Valuation Complete!")
      st.metric(
          label=f"Estimated Market Price in {location}",
          value=f"₹ {predicted_price:.2f} Lakhs",
          delta="Real-time ML Inference",
      )
    except Exception as e:
      st.error(f"Prediction error: {e}")