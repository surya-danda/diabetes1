import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
with open("diabetes_predictor.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.title("Diabetes Prediction App")

# Store previous predictions in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Input fields with default values
st.header("Enter Patient Details:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=2)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, format="%.1f", value=28.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, format="%.2f", value=0.47)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)[0]
    
    # If model supports probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1] * 100
        st.info(f"🧠 Probability of being Diabetic: **{prob:.2f}%**")
    else:
        prob = None
    
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    emoji = "🟥" if prediction == 1 else "🟩"
    st.success(f"The model predicts: **{emoji} {result}**")
    
    # Save result to session
    st.session_state.history.append({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Result": result,
        "Probability": prob if prob is not None else "N/A"
    })

# Display past predictions
if st.session_state.history:
    st.subheader("📊 Past Predictions")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # Plotting bar chart of results
    result_counts = df["Result"].value_counts()
    fig, ax = plt.subplots()
    result_counts.plot(kind='bar', color=["green", "red"], ax=ax)
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Prediction Results Summary")
    st.pyplot(fig)
