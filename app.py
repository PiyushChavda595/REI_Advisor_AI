import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="REI-Advisor AI | Premium Real Estate Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #1e40af; color: #ffffff; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; }
    .highlight-good { color: #059669; font-weight: bold; }
    .highlight-bad { color: #dc2626; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load('rei_preprocessor.joblib')
        classifier = joblib.load('rei_classifier_model.joblib')
        regressor = joblib.load('rei_regressor_model.joblib')
        return preprocessor, classifier, regressor
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure .joblib files are in the same directory.")
        return None, None, None

preprocessor, classifier, regressor = load_artifacts()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1256/1256650.png", width=100)
    st.title("REI-Advisor AI")
    st.markdown("### Intelligent Valuation Engine")
    st.info("System Status: 🟢 Online")
    st.markdown("---")
    st.caption("v1.1.0 | Updated Location Lists")

# --- 5. MAIN FORM ---
st.title("🏡 Property Investment Analysis")
st.markdown("Enter property details below to generate a real-time valuation report.")

with st.form("valuation_form"):
    
    st.subheader("1. Location & Layout")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # UPDATED LISTS: Added Dehradun, Uttarakhand, etc.
        city = st.selectbox("City", [
            "Mumbai", "Bangalore", "Pune", "Delhi", "Hyderabad", "Chennai", 
            "Kolkata", "Ahmedabad", "Dehradun", "Jaipur", "Lucknow", "Nagpur", 
            "Indore", "Chandigarh", "Kochi"
        ])
        # Text input allows you to type "Locality_429" exactly as it appears in your data
        locality = st.text_input("Locality", value="Locality_429") 
        
        state = st.selectbox("State", [
            "Maharashtra", "Karnataka", "Delhi", "Telangana", "Tamil Nadu", 
            "West Bengal", "Gujarat", "Uttarakhand", "Rajasthan", "Uttar Pradesh",
            "Punjab", "Kerala", "Haryana", "Madhya Pradesh"
        ])
        
    with col2:
        prop_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa", "Penthouse"])
        bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=3)
        size_sqft = st.number_input("Size (Sq. Ft.)", min_value=100, max_value=20000, value=4564)
        
    with col3:
        floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=4)
        total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=12)
        facing = st.selectbox("Facing Direction", ["East", "West", "North", "South", "North-East", "North-West", "South-East", "South-West"])

    st.markdown("---")
    
    st.subheader("2. Features & Condition")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2004)
        furnished = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
        owner_type = st.selectbox("Seller Type", ["Owner", "Dealer", "Builder"])
        
    with col5:
        # Mapped your "Yes/No" inputs to match typical user selection
        parking = st.selectbox("Parking Available?", ["Yes", "No"])
        security = st.selectbox("Gated Security?", ["Yes", "No"])
        availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        
    with col6:
        # Defaulting these to 'Medium' based on your input 'Medium'
        transport = st.select_slider("Public Transport Access", options=["Low", "Medium", "High"], value="Medium")
        # Your input data had '2' for schools and '1' for hospitals
        schools = st.slider("Nearby Schools (Count)", 0, 10, 2)
        hospitals = st.slider("Nearby Hospitals (Count)", 0, 10, 1)

    st.markdown("---")
    
    st.subheader("3. Luxury Amenities")
    # Your input had amenity_score = 4. 
    # Select any 4 items here to match that score.
    amenities_list = [
        "Swimming Pool", "Gym", "Clubhouse", "Jogging Track", "Power Backup", 
        "Garden", "Intercom", "Lift", "CCTV", "Wifi", "Party Hall"
    ]
    selected_amenities = st.multiselect("Select all available amenities (Select 4 to match your data):", amenities_list)
    
    # --- FORM SUBMIT BUTTON ---
    submitted = st.form_submit_button("🚀 Generate Analysis Report")

# --- 6. PREDICTION LOGIC & OUTPUT ---
if submitted:
    if preprocessor is None:
        st.error("Artifacts not loaded. Cannot predict.")
    else:
        # A. Prepare Input Data
        amenity_score = len(selected_amenities)
        # Calculate Age based on Year Built (Current Year - Year Built)
        # Input 2004 -> Age approx 21 years
        current_year = 2025
        age_of_property = current_year - year_built
        
        # Mapping UI "Yes/No" to dataset format if needed, 
        # but your categorical encoder likely handles "Yes"/"No" strings directly if they were in training data.
        
        input_data = pd.DataFrame({
            'State': [state],
            'City': [city],
            'Locality': [locality],
            'Property_Type': [prop_type],
            'BHK': [bhk],
            'Size_in_SqFt': [size_sqft],
            'Year_Built': [year_built],
            'Furnished_Status': [furnished],
            'Floor_No': [floor_no],
            'Total_Floors': [total_floors],
            'Age_of_Property': [age_of_property],
            'Nearby_Schools': [schools],
            'Nearby_Hospitals': [hospitals],
            'Public_Transport_Accessibility': [transport],
            'Parking_Space': [parking],
            'Security': [security],
            'Facing': [facing],
            'Owner_Type': [owner_type],
            'Availability_Status': [availability],
            'Amenity_Score': [amenity_score]
        })

        # B. Preprocess
        with st.spinner("Analyzing market trends..."):
            try:
                # Transform data using the loaded pipeline
                processed_input = preprocessor.transform(input_data)
                
                # C. Predict
                pred_class = classifier.predict(processed_input)[0]
                pred_price = regressor.predict(processed_input)[0]
                
                time.sleep(0.5) 
                
                # D. Display Results
                st.markdown("### 📊 Valuation Result")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### Estimated Market Price")
                    st.markdown(f"<h1 style='color: #1E3A8A;'>₹ {pred_price:.2f} Lakhs</h1>", unsafe_allow_html=True)
                    price_per_sqft = (pred_price * 100000) / size_sqft
                    st.caption(f"Avg. ₹ {price_per_sqft:.0f} / Sq. Ft.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### Investment Verdict")
                    if pred_class == "High_Potential":
                        st.markdown(f"<h1 class='highlight-good'>🌟 HIGH POTENTIAL</h1>", unsafe_allow_html=True)
                        st.caption("Undervalued relative to locality average.")
                    else:
                        st.markdown(f"<h1 class='highlight-bad'>⚠️ LOW POTENTIAL</h1>", unsafe_allow_html=True)
                        st.caption("Priced at a premium / Market Average.")
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Tip: Ensure the inputs (like 'Locality') match the format seen during training.")