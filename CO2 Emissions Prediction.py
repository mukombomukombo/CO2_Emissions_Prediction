import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('CO2_emissions_model.h5')

# Title of the web app
st.title('CO2 Emissions Prediction')

# Input fields for the vehicle parameters
engine_size = st.number_input('Engine Size (L)')
cylinders = st.number_input('Cylinders')
fuel_city = st.number_input('Fuel Consumption City (L/100 km)')
fuel_hwy = st.number_input('Fuel Consumption Hwy (L/100 km)')
fuel_comb = st.number_input('Fuel Consumption Comb (L/100 km)')
fuel_comb_mpg = st.number_input('Fuel Consumption Comb (mpg)')

# Button to trigger prediction
if st.button('Predict'):
    # Collect input features into a numpy array
    features = np.array([engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg]).reshape((1, -1, 1))
    
    # Make a prediction using the pre-trained model
    prediction = model.predict(features)
    
    # Display the prediction result
    st.write(f"Predicted CO2 Emissions: {float(prediction[0][0])} g/km")
