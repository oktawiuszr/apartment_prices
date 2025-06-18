import streamlit as st
import pandas as pd
import pickle

with open("./../ML_models/XGB_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Apartment Price Prediction")
st.markdown("Fill in the details below to estimate the apartment price.")

# Inputs
buliding_type = st.selectbox("Building Type", ["NoData", "blockOfFlats", "tenement", "apartmentBuilding"])
material = st.selectbox("Building Material", ["NoData", "brick", "concreteSlab"])
parking = st.checkbox("Parking Space")
balcony = st.checkbox("Balcony")
elevator = st.checkbox("Elevator")
security = st.checkbox("Security")
storage = st.checkbox("Storage")

square_meters = st.number_input("Area (m2)", min_value=1.0, max_value=500.00, value=50.0)
rooms = st.slider("Number of rooms", 1, 10, 2)
floor = st.number_input("Floor", min_value=0, max_value=50, value=4)
floor_count = st.number_input("Total number of floors in building", min_value=0, max_value=50, value=4)
build_year = st.number_input("Year of building", min_value=1800, max_value=2025, value=2000)
centre_distance = st.slider("Distance to center (km)", 0.0, 20.0, 2.0)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "type": buliding_type,
        "buildingMaterial": material,
        "squareMeters": square_meters,
        "rooms": rooms,
        "floor": floor,
        "floorCount": floor_count,
        "buildYear": build_year,
        "centreDistance": centre_distance,
        "hasParkingSpace": parking,
        "hasBalcony": balcony,
        "hasElevator": elevator,
        "hasSecurity": security,
        "hasStorageRoom": storage,
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: {prediction:,.0f} PLN")

