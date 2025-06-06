import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(layout="wide")
st.title("🌾 Crop Production Analysis & Predictor")

le_area = joblib.load("le_area.pkl")
le_item = joblib.load("le_item.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

with st.sidebar:
    st.header("🔍 Filter Parameters")
    area = st.selectbox("Select Area:", le_area.classes_)
    item = st.selectbox("Select Crop:", le_item.classes_)
    year = st.slider("Select Year:", 1960, 2025, 2020)
    area_harvested = st.number_input("Area Harvested (ha):", min_value=0.0)
    yield_ = st.number_input("Yield (kg/ha):", min_value=0.0)

if st.button("🚀 Predict Production"):
    input_data = pd.DataFrame([[le_area.transform([area])[0],
                                le_item.transform([item])[0],
                                year, area_harvested, yield_]],
                              columns=['Area_Code', 'Item_Code', 'Year', 'Area_Harvested', 'Yield'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌽 Predicted Production: {prediction:.2f} tons")

    st.subheader("📊 Visual Summary")
    chart_data = pd.DataFrame({
        'Category': ['Area Harvested', 'Yield', 'Predicted Production'],
        'Value': [area_harvested, yield_, prediction]
    })
    st.bar_chart(chart_data.set_index('Category'))

st.markdown("---")
st.subheader("📈 Insights Preview (Static)")
col1, col2 = st.columns(2)
with col1:
    st.image(r"D:\Anaconda\Project\Project_II\top_crops.png",caption="Top Crops")
with col2:
    st.image(r"D:\Anaconda\Project\Project_II\top_regions.png", caption="Top Regions")
