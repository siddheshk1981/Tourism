import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="siddhesh1981/tourism-package-predict-model", filename="best_tourism_package_predict_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("The Tourism Package Purchase Prediction App is an internal tool for Visit with Us,a leading travel company, that predicts  whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them")
st.write("Kindly enter the customer details to check whether they are likely to purchase the newly introduced Wellness Tourism Package before contacting them.")

# Collect user input

Age = st.number_input("Age",min_value=18,max_value=92,value=45)
TypeofContact = st.selectbox("TypeofContact",["Self Enquiry","Company Invited"])
CityTier = st.number_input("CityTier",min_value=1,max_value=3,step=1)
DurationOfPitch = st.number_input("DurationOfPitch",min_value=5,max_value=130,value=15)
Occupation = st.selectbox("Occupation",["Free Lancer","Salaried","Small Business","Large Business"])
Gender = st.selectbox("Gender",["Male","Female"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting",min_value=1,max_value=5,step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups",min_value=1,max_value=6,step=1)
ProductPitched = st.selectbox("ProductPitched",["Basic","Standard","King","Deluxe","Super Deluxe"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar",min_value=1,max_value=5,step=1)
MaritalStatus = st.selectbox("MaritalStatus",["Unmarried","Married","Divorced"])
NumberOfTrips = st.number_input("NumberOfTrips",min_value=1,max_value=22,step=1)
Passport = st.selectbox("Passport",["Yes","No"])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore",min_value=1,max_value=5,step=1)
OwnCar = st.selectbox("OwnCar",["Yes","No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting",min_value=0,max_value=3,step=1)
Designation = st.selectbox("Designation",["AVP","VP","Manager","Senior Manager","Executive"])
MonthlyIncome = st.number_input("MonthlyIncome",min_value=1000.0,max_value=100000.0,value=10000.0)


input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba > 0.6).astype(int)
    result = "Purchase" if prediction == 1 else "not Purchase"
    st.write(f"Based on the information provided, the customer is likely to {result} the newly introduced Wellness Tourism Package before contacting them.")
