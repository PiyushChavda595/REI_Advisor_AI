import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="REI-Advisor AI",
    page_icon="🏢",
    layout="wide"
)

# --- 2. DEBUG LOGGING ---
print("--- App Starting ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir()}")

# --- 3. LOAD MODELS (With Error Handling) ---
@st.cache_resource
def load_artifacts():
    print("--- Loading Artifacts... ---")
    try:
        # Load one by one to see which fails
        print("Loading Preprocessor...")
        preprocessor = joblib.load('rei_preprocessor.joblib')
        
        print("Loading Classifier...")
        classifier = joblib.load('rei_classifier_model.joblib')
        
        print("Loading Regressor...")
        regressor = joblib.load('rei_regressor_model.joblib')
        
        print("--- All Artifacts Loaded Successfully ---")
        return preprocessor, classifier, regressor
    except Exception as e:
        print(f"ERROR LOADING ARTIFACTS: {e}")
        return None, None, None

# --- 4. UI STRUCTURE ---
st.title("🏡 REI-Advisor AI: Real Estate Analytics")

# Load models in the background
preprocessor, classifier, regressor = load_artifacts()

if preprocessor is None:
    st.error("⚠️ System Error: Could not load model files.")
    st.warning(
        """
        **Troubleshooting:**
        1. Check that the .joblib files are in the *root* folder of your GitHub repo.
        2. Check the logs (Manage App -> Logs) to see the specific error.
        3. Ensure you are using Python 3.10 on Streamlit Cloud.
        """
    )
    st.stop() # Stop execution here if models failed

# --- 5. SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Configuration")
    st.success("Models Loaded & Ready")

# Main Form
with st.form("valuation_form"):
    st.subheader("Enter Property Details")
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", ["Mumbai", "Bangalore", "Pune", "Delhi", "Dehradun"])
        size_sqft = st.number_input("Size (Sq. Ft.)", 100, 20000, 1200)
        bhk = st.number_input("BHK", 1, 10, 2)
        locality = st.text_input("Locality", "Andheri East")
        
    with col2:
        prop_type = st.selectbox("Type", ["Apartment", "Villa", "Independent House"])
        amenities_count = st.slider("Amenity Score", 0, 20, 5)
        floor = st.number_input("Floor", 0, 100, 5)
        
    submitted = st.form_submit_button("Predict Value")

# --- 6. PREDICTION LOGIC ---
if submitted:
    # Prepare input data (Using dummy values for non-visible fields to prevent crash)
    input_df = pd.DataFrame({
        'State': ['Maharashtra'], # Default
        'City': [city],
        'Locality': [locality],
        'Property_Type': [prop_type],
        'BHK': [bhk],
        'Size_in_SqFt': [size_sqft],
        'Year_Built': [2015],
        'Furnished_Status': ['Semi-Furnished'],
        'Floor_No': [floor],
        'Total_Floors': [15],
        'Age_of_Property': [10],
        'Nearby_Schools': [2],
        'Nearby_Hospitals': [2],
        'Public_Transport_Accessibility': ['Medium'],
        'Parking_Space': ['Yes'],
        'Security': ['Yes'],
        'Facing': ['East'],
        'Owner_Type': ['Owner'],
        'Availability_Status': ['Ready_to_Move'],
        'Amenity_Score': [amenities_count]
    })
    
    try:
        processed_data = preprocessor.transform(input_df)
        price = regressor.predict(processed_data)[0]
        category = classifier.predict(processed_data)[0]
        
        st.success(f"Predicted Price: ₹ {price:.2f} Lakhs")
        st.info(f"Investment Potential: {category}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
