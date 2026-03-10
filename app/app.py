import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. Load the Model & Features ---
model = joblib.load('app/rf_model.pkl')
model_columns = joblib.load('app/model_columns.pkl')

# --- 2. Interface Setup ---
st.set_page_config(page_title="BMT Predictor", layout="wide")
st.title("🩺 Pediatric BMT Success Predictor")
st.write("A decision-support application for physicians predicting bone marrow transplant success.")

# --- 3. Physician Input Sidebar ---
st.sidebar.header("Patient Medical Data")
st.sidebar.write("Please input the top critical features identified by SHAP:")

relapse = st.sidebar.selectbox("Relapse History (1 = Yes, 0 = No)", [0, 1])
plt_recovery = st.sidebar.number_input("Platelet Recovery Time (Days)", value=20.0, step=1.0)
cd34 = st.sidebar.number_input("CD34+ Cell Dose (x10^6/kg)", value=5.0, step=0.1)

# --- 4. Prediction & Explainability Logic ---
if st.sidebar.button("Predict Transplant Success"):
    
    # Create patient profile
    patient_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    patient_data['Relapse'] = relapse
    patient_data['PLTrecovery'] = plt_recovery
    patient_data['CD34kgx10d6'] = cd34
    
    # Make prediction
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]

    # Display results
    st.subheader("Prediction Results")
    if prediction == 1:
        st.success(f"**High Likelihood of Survival** (Confidence: {probability:.1%})")
    else:
        st.error(f"**High Risk / Lower Likelihood of Survival** (Confidence: {probability:.1%})")
        
    st.info("Note: This prediction emphasizes the primary driving factors (Relapse, PLTrecovery, and CD34).")

    # --- NEW: SHAP Integration for the Interface ---
    st.subheader("🧠 Model Explainability (Why did the AI choose this?)")
    st.write("This chart shows exactly how this specific patient's data pushed the model's decision.")
    
    # Calculate SHAP values for this single patient
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(patient_data)
    
    # Safely extract the data depending on the SHAP version
    if isinstance(shap_vals, list):
        patient_shap = shap_vals[1][0]
    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        patient_shap = shap_vals[0, :, 1]
    else:
        patient_shap = shap_vals[0]

    # Extract the impact scores of the 3 inputted features
    impacts = {
        'Relapse History': patient_shap[model_columns.index('Relapse')],
        'Platelet Recovery Time': patient_shap[model_columns.index('PLTrecovery')],
        'CD34+ Cell Dose': patient_shap[model_columns.index('CD34kgx10d6')]
    }

    # Draw a clean, physician-friendly bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['green' if val > 0 else 'red' for val in impacts.values()]
    ax.barh(list(impacts.keys()), list(impacts.values()), color=colors)
    ax.set_xlabel("SHAP Value (Green = Increases Survival, Red = Decreases Survival)")
    ax.set_title("Feature Impact for this Patient")
    
    # Display the plot in Streamlit
    st.pyplot(fig)