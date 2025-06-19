import streamlit as st
import pandas as pd
import requests

# Title and description
st.set_page_config(page_title="Obesity Level Prediction", layout="centered", initial_sidebar_state="auto")
st.title("Obesity Level Prediction Web App")
st.write("🔎 Masukkan data di bawah ini untuk memprediksi tingkat obesitas Anda!")

# Input form

with st.sidebar:
    st.title("Obesity Prediction")
    st.markdown("---")
    page = st.radio("Navigation", ["Predict Obesity Level", "Test Cases"])
    st.markdown("---")
    st.info("This application predicts booking status based on input parameters.")
    
if page == "Predict Obesity Level":
    with st.form("prediction_form"):
        
        st.markdown("### Section 1: Demographic & Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age    = st.number_input("Age",   min_value=1, max_value=100, value=25)
            height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
            weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
        with col2:
            family_history = st.selectbox(
                "Family history with overweight",
                ["yes", "no"]
            )
            favc = st.selectbox(
                "Frequently consumes high caloric food (FAVC)",
                ["yes", "no"]
            )
            fcvc = st.slider(
                "Frequency of vegetable consumption (FCVC)",
                0, 3, 1
            )
            ncp = st.slider(
                "Number of main meals per day (NCP)",
                1, 5, 3
            )

        st.markdown("---")
        st.markdown("### Section 2: Lifestyle & Habits")
        
        col3, col4 = st.columns(2)
        with col3:
            caec  = st.selectbox(
                "Consumption between meals (CAEC)",
                ["no", "Sometimes", "Frequently", "Always"]
            )
            smoke = st.selectbox("Do you smoke? (SMOKE)", ["yes", "no"])
            ch2o  = st.number_input(
                "Daily water consumption (liters) (CH2O)",
                min_value=0.0, max_value=5.0, value=1.0, step=0.1
            )
            scc   = st.selectbox(
                "Monitor calorie consumption? (SCC)",
                ["yes", "no"]
            )
        with col4:
            faf   = st.slider(
                "Physical activity (days/week) (FAF)",
                0, 7, 3
            )
            tue   = st.slider(
                "Tech use per day (hours) (TUE)",
                0.0, 24.0, 2.0, step=0.5
            )
            calc  = st.selectbox(
                "Alcohol consumption (CALC)",
                ["no", "Sometimes", "Frequently"]
            )
            mtrans = st.selectbox(
                "Means of transportation (MTRANS)",
                ["Walking", "Bike", "Motorbike", "Car", "Public_Transportation"]
            )

        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Construct payload
            payload = {
                "Gender": gender,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "family_history_with_overweight": family_history,
                "FAVC": favc,
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": caec,
                "SMOKE": smoke,
                "CH2O": ch2o,
                "SCC": scc,
                "FAF": faf,
                "TUE": tue,
                "CALC": calc,
                "MTRANS": mtrans
            }
            with st.spinner("Predicting..."):
                try:
                    response = requests.post("http://localhost:8000/predict", json=payload)
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.HTTPError as e:
                    err_body = e.response.json() if e.response.headers.get("content-type","").startswith("application/json") else e.response.text
                    st.error(f"Error {e.response.status_code}: {err_body}")
                except Exception as e:
                    st.error(f"Error: {e}")
                
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Proobability: {result['probability']:.2f}")
        
if page == "Test Cases":
    st.title("📋 Test Cases")

    # 2 contoh payload template
    sample_payloads = [
        {
            "Gender": "Male",
            "Age": 30,
            "Height": 1.80,
            "Weight": 85.0,
            "family_history_with_overweight": "yes",
            "FAVC": "yes",
            "FCVC": 1.0,
            "NCP": 3.0,
            "CAEC": "Sometimes",
            "SMOKE": "no",
            "CH2O": 2.0,
            "SCC": "no",
            "FAF": 2.0,
            "TUE": 3.0,
            "CALC": "no",
            "MTRANS": "Car"
        },
        {
            "Gender": "Female",
            "Age": 25,
            "Height": 1.65,
            "Weight": 60.0,
            "family_history_with_overweight": "no",
            "FAVC": "no",
            "FCVC": 2.0,
            "NCP": 4.0,
            "CAEC": "Frequently",
            "SMOKE": "no",
            "CH2O": 3.0,
            "SCC": "yes",
            "FAF": 5.0,
            "TUE": 1.0,
            "CALC": "Sometimes",
            "MTRANS": "Public_Transportation"
        }
    ]

    # Tampilkan sebagai DataFrame
    df = pd.DataFrame(sample_payloads)
    st.dataframe(df.T, width=600, height=500)
    
    if st.button("Run Test Cases"):
        with st.spinner("Predicting test cases..."):
            preds = []
            for rec in sample_payloads:
                try:
                    r = requests.post("http://localhost:8000/predict", json=rec)
                    r.raise_for_status()
                    preds.append(r.json()["prediction"])
                except Exception as e:
                    preds.append(f"Error")
        df["prediction"] = preds
        pred_df = df[['prediction']]
        st.dataframe(pred_df.T, width=600)
