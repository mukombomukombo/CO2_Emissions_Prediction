import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model without compiling it to avoid the warnings
model = load_model('CO2_emissions_model.h5', compile=False)

# Title of the web app
st.title('CO2 Emissions Prediction')

# Input fields for the vehicle parameters
engine_size = st.number_input('Engine Size (L)', min_value=0.0, format="%.1f")
cylinders = st.number_input('Cylinders', min_value=1, step=1)
fuel_city = st.number_input('Fuel Consumption City (L/100 km)', min_value=0.0, format="%.1f")
fuel_hwy = st.number_input('Fuel Consumption Hwy (L/100 km)', min_value=0.0, format="%.1f")
fuel_comb = st.number_input('Fuel Consumption Comb (L/100 km)', min_value=0.0, format="%.1f")
fuel_comb_mpg = st.number_input('Fuel Consumption Comb (mpg)', min_value=1, step=1)

# Button to trigger prediction
if st.button('Predict'):
    # Collect input features into a numpy array and ensure they are not empty
    features = np.array([engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg]).reshape((1, -1))

    if features.size == 0:
        st.error("One or more inputs are empty. Please provide valid inputs.")
    else:
        # Make a prediction using the pre-trained model
        prediction = model.predict(features)

        # Display the prediction result
        st.write(f"Predicted CO2 Emissions: {float(prediction[0][0])} g/km")
