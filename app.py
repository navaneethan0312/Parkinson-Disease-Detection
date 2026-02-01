import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("parkinson_model.pkl")

st.set_page_config(page_title="Parkinson's Disease Detection")

st.title("Parkinson's Disease Detection using Voice Frequency")
st.write("Enter patient voice features and click Predict")

feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

inputs = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_data = np.array([inputs])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🟥 Parkinson’s Disease Detected")
    else:
        st.success("🟩 Healthy Person")
