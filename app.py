import streamlit as st
from valid_data_prediction import xgb_model, smiles_to_morgan_fp, compute_descriptors
import numpy as np

st.title("Bio Marker Value Predictor")

smiles = st.text_input("Enter SMILES string:")

if st.button("Predict"):
    fp = smiles_to_morgan_fp(smiles)
    desc = compute_descriptors(smiles)
    if fp is not None and None not in desc:
        # Combine fingerprint and descriptors
        features = np.hstack((fp, desc)).reshape(1, -1)
        prediction = xgb_model.predict(features)[0]
        st.success(f"Predicted Bio Marker Value: {prediction:.4f}")
    else:
        st.error("Invalid SMILES string. Please try again.")